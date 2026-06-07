"""
EDGAR 실시간 스캐너 — 최근 8-K/6-K 공시에서 섹터 키워드 감지.

사용:
  python3 -m theme_predictor.collectors.edgar_scanner
  python3 -m theme_predictor.collectors.edgar_scanner --days 3
  python3 -m theme_predictor.collectors.edgar_scanner --theme 우주 AI인프라
"""

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import requests
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
from theme_predictor.config import SECTOR_KEYWORDS, OUTPUT_DIR

HEADERS = {"User-Agent": "stock-research-bot hideinthecodes@gmail.com"}
BASE_URL = "https://efts.sec.gov/LATEST/search-index"
MAX_HITS = 40  # 섹터당 최대 수집 건수
SAVE_PATH = Path(__file__).parents[1] / OUTPUT_DIR / "edgar_scanner.json"


def _build_query(keywords: list[str]) -> str:
    """키워드 리스트 → EDGAR 검색 쿼리 문자열."""
    quoted = [f'"{k}"' if " " in k else k for k in keywords]
    return " OR ".join(quoted)


def scan_sector(
    sector: str,
    keywords: list[str],
    start: str,
    end: str,
    forms: str = "8-K,6-K",
) -> list[dict]:
    """단일 섹터 키워드로 EDGAR 공시 검색 후 결과 반환."""
    query = _build_query(keywords)
    params = {
        "q":         query,
        "forms":     forms,
        "dateRange": "custom",
        "startdt":   start,
        "enddt":     end,
        "_source":   "file_date,period_of_report,entity_name,file_num,form_type,biz_location",
        "hits.hits.total.value": MAX_HITS,
    }
    try:
        r = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        hits = r.json().get("hits", {}).get("hits", [])
    except Exception as e:
        logger.warning(f"[{sector}] 조회 실패: {e}")
        return []

    results = []
    for h in hits:
        src = h.get("_source", {})
        names = src.get("display_names", [])
        entity = names[0].split("(")[0].strip() if names else ""
        ticker = ""
        if names:
            m = __import__("re").search(r"\(([A-Z]{1,5})\)", names[0])
            if m:
                ticker = m.group(1)
        results.append({
            "섹터":     sector,
            "기업":     entity,
            "티커":     ticker,
            "종류":     src.get("form", ""),
            "공시일":   src.get("file_date", ""),
            "보고기간": src.get("period_ending", ""),
            "위치":     (src.get("biz_locations") or [""])[0],
            "accession": src.get("adsh", ""),
        })
    return results


def run(days: int = 1, themes: list[str] | None = None) -> dict:
    """
    전체 섹터(또는 지정 섹터) 스캔.
    Returns: {sector: [filing_dict, ...], ...}
    """
    end_date   = date.today().isoformat()
    start_date = (date.today() - timedelta(days=days)).isoformat()
    logger.info(f"EDGAR 스캔: {start_date} ~ {end_date}")

    target_sectors = themes if themes else list(SECTOR_KEYWORDS.keys())
    all_results: dict[str, list[dict]] = {}

    for sector in target_sectors:
        kws = SECTOR_KEYWORDS.get(sector)
        if not kws:
            logger.warning(f"섹터 없음: {sector}")
            continue
        hits = scan_sector(sector, kws, start_date, end_date)
        all_results[sector] = hits
        logger.info(f"  [{sector}] {len(hits)}건")

    # 저장
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scan_date": end_date,
        "from":      start_date,
        "to":        end_date,
        "results":   all_results,
    }
    SAVE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.success(f"저장 완료 → {SAVE_PATH}")
    return all_results


def print_summary(results: dict):
    """스캔 결과 콘솔 출력."""
    print(f"\n{'='*60}")
    print(f"  EDGAR 공시 스캔 결과")
    print(f"{'='*60}")
    for sector, filings in results.items():
        if not filings:
            continue
        print(f"\n[{sector}] {len(filings)}건")
        for f in filings[:5]:  # 섹터당 최대 5건 출력
            print(f"  {f['공시일']}  {f['종류']:5s}  {f['기업']}")
        if len(filings) > 5:
            print(f"  ... 외 {len(filings)-5}건")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDGAR 실시간 공시 스캐너")
    parser.add_argument("--days",  "-d", type=int, default=1,
                        help="몇 일 전부터 스캔할지 (기본: 1)")
    parser.add_argument("--theme", "-t", nargs="+", default=None,
                        help="특정 섹터만 (예: --theme 우주 AI인프라)")
    parser.add_argument("--forms", "-f", default="8-K,6-K",
                        help="공시 종류 (기본: 8-K,6-K)")
    args = parser.parse_args()

    results = run(days=args.days, themes=args.theme)
    print_summary(results)
