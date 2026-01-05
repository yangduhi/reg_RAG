from src.ingestion.utils import clean_korean_text, html_to_markdown_table


def test_clean_korean_text_removes_box_chars():
    """박스 드로잉 문자(┌─┐)가 잘 제거되는지 테스트"""
    dirty_text = "┌── 항목 ──┐\n│  내용  │\n└──────┘"
    cleaned = clean_korean_text(dirty_text)

    # 박스 문자가 없어야 함
    assert "┌" not in cleaned
    assert "│" not in cleaned
    # 내용은 남아있어야 함
    assert "항목" in cleaned
    assert "내용" in cleaned

def test_html_table_conversion():
    """HTML 테이블이 Markdown 표로 잘 변환되는지 테스트"""
    html_input = """
    <table>
        <tr><th>구분</th><th>기준</th></tr>
        <tr><td>속도</td><td>50km/h</td></tr>
    </table>
    """

    markdown_output = html_to_markdown_table(html_input)

    # 마크다운 표 문법(|)이 존재해야 함
    assert "|" in markdown_output
    assert "---" in markdown_output
    assert "구분" in markdown_output
    assert "50km/h" in markdown_output
