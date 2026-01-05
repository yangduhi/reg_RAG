import re

import markdownify
from bs4 import BeautifulSoup


def clean_korean_text(text: str) -> str:
    """
    한국 법규 텍스트 정제
    1. 박스 드로잉 문자(┌─┐) 제거
    2. 과도한 공백 제거
    """
    if not text: return ""

    # 박스 문자 제거
    text = re.sub(r'[┌┐└┘├┤┬┴┼━┃┏┓┗┛┣┫┳┻╋─│┯┠┨]', ' ', text)

    # 연속된 공백 축소 (줄바꿈은 유지)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        clean_line = re.sub(r'\s+', ' ', line).strip()
        if clean_line:
            cleaned_lines.append(clean_line)

    return "\n".join(cleaned_lines)

def html_to_markdown_table(html_text: str) -> str:
    """
    HTML 테이블을 마크다운 표로 변환 (표 구조 보존)
    """
    if not html_text: return ""

    # HTML 태그가 없으면 일반 텍스트로 간주하고 정제
    if not ("<table" in html_text or "<tr" in html_text):
        return clean_korean_text(html_text)

    try:
        # markdownify를 사용하여 변환
        # strip=['a', 'img'] 등으로 불필요한 태그 제거 가능
        md = markdownify.markdownify(html_text, heading_style="ATX")
        return clean_korean_text(md)
    except Exception:
        # 변환 실패 시 텍스트만 추출
        return clean_korean_text(BeautifulSoup(html_text, "lxml").get_text())
