import json
import re
import time
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from loguru import logger
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from theme_tracker.config import HOT_THEMES, THEME_KEYWORDS

HEADERS = {"User-Agent": "stock-research-bot hideinthecodes@gmail.com"}
CACHE_PATH = Path(__file__).parent / "cache" / "subsidiaries.json"
RELEVANCE_LEVELS = ["직접 연관", "간접 연관", "테마 주변", "연관 없음"]


# ──────────────────────────────────────────────
# 캐시 헬퍼
# ──────────────────────────────────────────────

def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


# ──────────────────────────────────────────────
# EDGAR Exhibit 21 파싱
# ──────────────────────────────────────────────

def _fetch_subsidiaries(ticker: str) -> list[str]:
    """
    SEC 10-K Exhibit 21에서 자회사 이름 목록 반환.
    캐시 우선; 없으면 EDGAR 조회 후 저장.
    """
    cache = _load_cache()
    if ticker in cache:
        return cache[ticker]

    names: list[str] = []
    try:
        names = _edgar_exhibit21(ticker)
    except Exception as e:
        logger.debug(f"{ticker} Exhibit 21 조회 실패: {e}")

    cache[ticker] = names
    _save_cache(cache)
    return names


def _get_cik(ticker: str) -> str:
    """EDGAR company_tickers.json에서 CIK 조회."""
    try:
        r = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=HEADERS, timeout=15,
        )
        r.raise_for_status()
        for entry in r.json().values():
            if entry.get("ticker", "").upper() == ticker.upper():
                return str(entry["cik_str"])
    except Exception:
        pass
    return ""


def _get_latest_10k_accession(cik: str) -> str:
    """submissions JSON에서 최신 10-K accession number 반환."""
    r = requests.get(
        f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json",
        headers=HEADERS, timeout=15,
    )
    r.raise_for_status()
    filings = r.json().get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    accs = filings.get("accessionNumber", [])
    for form, acc in zip(forms, accs):
        if form == "10-K":
            return acc
    return ""


def _find_exhibit21_url(cik: str, accession: str) -> str:
    """
    EDGAR 디렉토리 HTML에서 Exhibit 21 파일 URL 찾기.
    exhibit21*.htm / ex21*.htm 등 매칭, exhibit31*/32* 등 제외.
    """
    base = accession.replace("-", "")
    dir_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{base}/"
    r = requests.get(dir_url, headers=HEADERS, timeout=15)
    r.raise_for_status()

    # 파일 링크 추출
    hrefs = re.findall(r'href="(/Archives/edgar/data/[^"]+)"', r.text)

    candidates = []
    for href in hrefs:
        fname = href.split("/")[-1].lower()
        # exhibit21 이지만 exhibit312 같은 숫자 이어지는 것 제외
        if re.match(r"(ex-?21|exhibit21)[^3-9]", fname) or fname == "ex21.htm":
            candidates.append("https://www.sec.gov" + href)

    # 가장 짧은 이름 (단순한 게 정답일 가능성 높음)
    if candidates:
        return sorted(candidates, key=len)[0]
    return ""


def _parse_exhibit21_html(url: str) -> list[str]:
    """Exhibit 21 HTML에서 회사명 파싱 (BeautifulSoup)."""
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    names = []
    seen = set()

    # 테이블 파싱 — 첫 번째 컬럼이 Name
    _header_words = {"name", "subsidiary", "subsidiaries", "jurisdiction",
                     "incorporation", "state", "country", "percent", "ownership"}

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        for row in rows:
            cells = row.find_all(["td", "th"])
            if not cells:
                continue
            name = cells[0].get_text(strip=True)
            if len(name) < 3:
                continue
            low = name.lower()
            # 헤더 행: 단어 대부분이 헤더 단어이면 스킵
            words = set(re.split(r"[\s,]+", low))
            if words & _header_words and len(words) <= 4:
                continue
            key = low.strip()
            if key not in seen:
                seen.add(key)
                names.append(name)

    # 테이블 없으면 단락에서 텍스트 추출
    if not names:
        for tag in soup.find_all(["p", "div", "li", "span"]):
            txt = tag.get_text(strip=True)
            if 4 < len(txt) < 100 and txt.lower() not in seen:
                seen.add(txt.lower())
                names.append(txt)

    return names[:200]


def _edgar_exhibit21(ticker: str) -> list[str]:
    """CIK 조회 → 최신 10-K → Exhibit 21 URL → 파싱."""
    cik = _get_cik(ticker)
    if not cik:
        return []

    accession = _get_latest_10k_accession(cik)
    if not accession:
        return []

    ex21_url = _find_exhibit21_url(cik, accession)
    if not ex21_url:
        return []

    return _parse_exhibit21_html(ex21_url)


# ──────────────────────────────────────────────
# 자회사 연관도 체크
# ──────────────────────────────────────────────

def check_subsidiary_relevance(ticker: str, theme: str) -> str:
    """
    자회사 이름이 테마 키워드와 얼마나 매칭되는지 판단.
    Returns: "직접 연관" / "간접 연관" / "연관 없음"
    """
    if theme not in THEME_KEYWORDS:
        return "연관 없음"

    subsidiaries = _fetch_subsidiaries(ticker)
    if not subsidiaries:
        return "연관 없음"

    sub_text = " ".join(subsidiaries).lower()
    subs = THEME_KEYWORDS[theme]
    keywords = [k.lower() for sub_kws in subs.values() for k in sub_kws]

    hits = sum(1 for k in keywords if k in sub_text)
    if hits >= 2:
        return "직접 연관"
    if hits >= 1:
        return "간접 연관"
    return "연관 없음"


# ──────────────────────────────────────────────
# 메인 연관도 체크 (모회사 + 자회사 통합)
# ──────────────────────────────────────────────

def check_relevance(ticker: str, theme: str) -> str:
    """
    HOT_THEMES 중 하나에 대해 이 종목이 얼마나 연관됐는지 판단.
    모회사 설명 + 자회사 이름 모두 포함해 키워드 매칭.
    Returns: "직접 연관" / "간접 연관" / "테마 주변" / "연관 없음"
    """
    if theme not in THEME_KEYWORDS:
        return "연관 없음"

    try:
        info = yf.Ticker(ticker).info
        desc = " ".join([
            info.get("longBusinessSummary", ""),
            info.get("longName", ""),
            info.get("industry", ""),
        ]).lower()
    except Exception:
        return "연관 없음"

    subsidiaries = _fetch_subsidiaries(ticker)
    sub_text = " ".join(subsidiaries).lower()

    sec_desc = _fetch_sec_text(ticker)
    full_text = desc + " " + sec_desc + " " + sub_text

    subs = THEME_KEYWORDS[theme]
    direct_kws = [k.lower() for sub_kws in subs.values() for k in sub_kws]

    direct_hit = sum(1 for k in direct_kws if k in full_text)

    if direct_hit >= 3:
        return "직접 연관"
    if direct_hit >= 1:
        return "간접 연관"

    indirect_terms = {
        "AI인프라": ["power", "electric", "compute", "infrastructure", "data"],
        "우주":     ["aerospace", "defense", "materials", "propulsion", "avionics"],
        "원자력":   ["energy", "power generation", "fuel", "reactor"],
        "바이오":   ["pharmaceutical", "research", "laboratory", "CRO", "CMO"],
        "방산":     ["defense", "aerospace", "government contract", "DoD"],
        "EV자율주행": ["automotive", "electronics", "battery", "sensor"],
    }
    ind_kws = indirect_terms.get(theme, [])
    if any(k in full_text for k in ind_kws):
        return "테마 주변"

    return "연관 없음"


def _fetch_sec_text(ticker: str) -> str:
    """SEC EDGAR에서 최근 8-K 메타데이터 일부 가져와 텍스트에 추가."""
    try:
        search_url = (
            f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22"
            f"&dateRange=custom&startdt=2025-01-01&forms=8-K"
        )
        r = requests.get(search_url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            return ""
        hits = r.json().get("hits", {}).get("hits", [])
        if not hits:
            return ""
        source = hits[0].get("_source", {})
        return source.get("file_date", "") + " " + source.get("period_of_report", "")
    except Exception:
        return ""


def batch_relevance(tickers: list[str]) -> dict[str, dict[str, str]]:
    """
    {ticker: {"AI인프라": "직접 연관", "우주": "연관 없음"}}
    """
    from theme_tracker.config import DELAY
    results = {}
    for ticker in tickers:
        rel = {}
        for theme in HOT_THEMES:
            rel[theme] = check_relevance(ticker, theme)
            time.sleep(0.2)
        results[ticker] = rel
        time.sleep(DELAY)
    return results
