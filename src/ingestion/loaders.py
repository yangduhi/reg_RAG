import asyncio
import re
from abc import ABC, abstractmethod
from pathlib import Path
from functools import wraps

import aiofiles
import pdfplumber
from bs4 import BeautifulSoup

from src.core.logging import logger
from src.domain.schemas import DocumentMetadata, IngestedDocument, RegulationRegion
from src.ingestion.utils import clean_korean_text, html_to_markdown_table


class BaseLoader(ABC):
    @abstractmethod
    async def load(self, file_path: Path) -> list[IngestedDocument]:
        pass


class LoaderFactory:
    """A factory class that creates the appropriate loader based on the file extension."""

    _loaders = {}

    @classmethod
    def register(cls, extension: str):
        """A decorator to register loader classes for a given file extension."""
        def decorator(loader_class: type[BaseLoader]):
            @wraps(loader_class)
            def wrapper(*args, **kwargs):
                return loader_class(*args, **kwargs)
            cls._loaders[extension.lower()] = loader_class
            logger.debug(f"🔩 Loader registered: '{extension}' -> {loader_class.__name__}")
            return wrapper
        return decorator

    @classmethod
    def create(cls, file_path: Path) -> BaseLoader:
        """Creates and returns a loader instance suitable for the file path."""
        extension = file_path.suffix.lower()
        loader_class = cls._loaders.get(extension)

        if not loader_class:
            logger.error(f"❌ Unsupported file extension: {extension}")
            raise ValueError(f"Unsupported file extension: {extension}")

        logger.debug(f"🏭 Creating loader for '{file_path.name}': {loader_class.__name__}")
        return loader_class()


@LoaderFactory.register(".xml")
class UniversalXmlLoader(BaseLoader):
    async def load(self, file_path: Path) -> list[IngestedDocument]:
        # [1] Encoding (UTF-8 -> CP949 -> EUC-KR)
        content = ""
        for encoding in ["utf-8", "cp949", "euc-kr"]:
            try:
                async with aiofiles.open(file_path, "r", encoding=encoding) as f:
                    content = await f.read()
                break
            except UnicodeDecodeError:
                continue
        if not content:
            return []

        try:
            # [2] Parsing XML
            try:
                soup = BeautifulSoup(content, "lxml-xml")
            except Exception:
                soup = BeautifulSoup(content, "xml")

            # [3] Metadata and ID Extraction
            file_name = file_path.stem
            id_match = re.search(r"(\d+)", file_name)
            raw_id = id_match.group(1) if id_match else file_name

            subject = soup.find(["SUBJECT", "Subject", "편장절관", "조문제목", "법령명"])
            title = subject.get_text(strip=True) if subject else file_name

            # [4] Korean Regulation Identification
            is_korean = (
                "KMVSS" in file_path.name.upper()
                or "KOR" in file_path.name.upper()
                or re.search(r"[가-힣]", file_name)
                or soup.find("조문")
            )
            region = RegulationRegion.KMVSS if is_korean else RegulationRegion.FMVSS
            std_id = f"KMVSS {raw_id}" if is_korean and not str(raw_id).startswith("KMVSS") else raw_id

            # [5] Content Extraction
            full_text = self._extract_text(soup, is_korean)

            if len(full_text.strip()) < 10:
                return []

            if region == RegulationRegion.KMVSS:
                logger.info(f"🇰🇷 [LOAD] {std_id} loaded successfully ({len(full_text)} chars)")

            return [
                IngestedDocument(
                    content=full_text,
                    metadata=DocumentMetadata(
                        source_file=file_path.name,
                        region=region,
                        standard_id=std_id,
                        title=title,
                    ),
                )
            ]
        except Exception as e:
            logger.error(f"Error processing XML file ({file_path.name}): {e}", exc_info=True)
            return []

    def _extract_text(self, soup: BeautifulSoup, is_korean: bool) -> str:
        """Extracts the main text content from the parsed XML."""
        if is_korean:
            content_tag = soup.find(["CONTENT", "Content", "본문"])
            if content_tag:
                try:
                    return html_to_markdown_table(content_tag.decode_contents())
                except Exception:
                    pass

            korean_tags = soup.find_all(["조문", "article", "조"])
            if korean_tags:
                texts = []
                for tag in korean_tags:
                    jo_no = tag.find(["조문번호", "jomunno", "조번호"])
                    jo_content = tag.find(["조문내용", "jomuncontent", "항", "호"])
                    part = f"제{jo_no.get_text(strip=True)}조 " if jo_no else ""
                    if jo_content:
                        try:
                            part += "\n" + html_to_markdown_table(jo_content.decode_contents())
                        except Exception:
                            part += "\n" + clean_korean_text(jo_content.get_text())
                    texts.append(part)
                return "\n\n".join(texts)
        
        # Default for non-Korean or as a fallback
        return clean_korean_text(soup.get_text(separator="\n", strip=True))


@LoaderFactory.register(".pdf")
class ECEPDFLoader(BaseLoader):
    
    def _process_pdf(self, file_path: Path) -> list[IngestedDocument]:
        """Synchronous function to process PDF files."""
        docs = []
        try:
            match = re.search(r"ECE_R(\d+)", file_path.name)
            std_id = f"R{match.group(1)}" if match else file_path.stem
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    tables = page.extract_tables()
                    md_tables = []
                    if tables:
                        for table in tables:
                            clean_tbl = [
                                [str(c).replace("\n", " ") if c else "" for c in r]
                                for r in table
                            ]
                            if not clean_tbl:
                                continue
                            
                            header = "| " + " | ".join(clean_tbl[0]) + " |"
                            sep = "| " + " | ".join(["---"] * len(clean_tbl[0])) + " |"
                            body = "\n".join(["| " + " | ".join(r) + " |" for r in clean_tbl[1:]])
                            md_tables.append(f"\n{header}\n{sep}\n{body}\n")

                    full_content = text + "\n".join(md_tables)
                    if len(full_content.strip()) < 50:
                        continue
                    
                    docs.append(
                        IngestedDocument(
                            content=full_content,
                            metadata=DocumentMetadata(
                                source_file=file_path.name,
                                region=RegulationRegion.ECE,
                                standard_id=std_id,
                                title="ECE Regulation",
                                page=i + 1,
                            ),
                        )
                    )
            return docs
        except Exception as e:
            logger.error(f"Error processing PDF file ({file_path.name}): {e}", exc_info=True)
            return []

    async def load(self, file_path: Path) -> list[IngestedDocument]:
        """Asynchronously loads a PDF file by running the synchronous processing in a separate thread."""
        return await asyncio.to_thread(self._process_pdf, file_path)
