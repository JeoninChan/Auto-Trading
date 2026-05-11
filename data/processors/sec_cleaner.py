"""
SEC 공시 HTML 클리닝
XBRL 태그, 인라인 CSS, 스크립트 제거 후 순수 텍스트 추출
"""
import re
from bs4 import BeautifulSoup, Comment


def clean_sec_html(raw: str) -> str:
    """
    SEC EDGAR HTML/XBRL 원문 → 읽기 가능한 순수 텍스트
    """
    import warnings
    from bs4 import XMLParsedAsHTMLWarning
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

    # XBRL이면 xml 파서, 일반 HTML이면 lxml
    parser = "lxml-xml" if raw.strip().startswith("<?xml") else "lxml"
    soup = BeautifulSoup(raw, parser)

    # 1. 불필요한 태그 완전 제거
    for tag in soup(["script", "style", "meta", "link", "head"]):
        tag.decompose()

    # 2. HTML 주석 제거
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # 3. XBRL/iXBRL 네임스페이스 태그 → 내용만 남기기
    for tag in soup.find_all(re.compile(r"^ix:|^xbrl|^dei:|^us-gaap:")):
        tag.unwrap()

    # 4. EDGAR 헤더 섹션 (<DOCUMENT>, <TYPE> 등) 제거
    text = soup.get_text(separator="\n")

    # 5. SGML 헤더 라인 제거 (<DOCUMENT>, <TYPE>, <SEQUENCE> 등)
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("<") and stripped.endswith(">") and len(stripped) < 100:
            continue
        cleaned.append(stripped)

    # 6. 빈 줄 3개 이상 → 1개로 압축
    result = "\n".join(cleaned)
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


def clean_file(path: str) -> str:
    """파일 경로에서 읽어 클리닝된 텍스트 반환 + _clean.txt 저장"""
    from pathlib import Path
    p = Path(path)
    raw = p.read_text(encoding="utf-8", errors="ignore")
    cleaned = clean_sec_html(raw)

    clean_path = p.parent / (p.stem + "_clean.txt")
    clean_path.write_text(cleaned, encoding="utf-8")
    return cleaned


def clean_all_existing():
    """기존에 수집된 모든 SEC .txt 파일 일괄 클리닝"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parents[2]))

    from loguru import logger

    # 루트 datasets 와 data/datasets 둘 다 탐색
    base = Path(__file__).parents[2]
    patterns = [
        base / "datasets" / "us" / "sec",
        base / "data" / "datasets" / "us" / "sec",
    ]

    for sec_dir in patterns:
        if not sec_dir.exists():
            continue
        files = [f for f in sec_dir.rglob("*.txt") if not f.name.endswith("_clean.txt")]
        logger.info(f"{sec_dir}: {len(files)}개 파일 클리닝")
        for f in files:
            try:
                clean_file(str(f))
            except Exception as e:
                logger.warning(f"{f.name} 실패: {e}")

    logger.info("완료")


if __name__ == "__main__":
    clean_all_existing()
