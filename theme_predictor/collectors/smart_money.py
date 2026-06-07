import json
import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime
from loguru import logger

import sys; sys.path.insert(0, str(Path(__file__).parents[2]))
from theme_predictor.config import SECTOR_KEYWORDS, DELAY

OUT = Path(__file__).parents[1] / "signals" / "smart_money.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def fetch_unusual_whales_sectors() -> dict[str, int]:
    """Unusual Whales 옵션 플로우 — 섹터별 콜옵션 집계."""
    sector_count: dict[str, int] = {}
    try:
        r = requests.get("https://unusualwhales.com/flow", headers=HEADERS, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        rows = soup.select("table tbody tr")
        for row in rows[:200]:
            cells = row.find_all("td")
            if len(cells) < 4:
                continue
            option_type = cells[2].get_text(strip=True).upper() if len(cells) > 2 else ""
            ticker = cells[0].get_text(strip=True).upper() if cells else ""
            if "CALL" not in option_type:
                continue

            import yfinance as yf
            try:
                info = yf.Ticker(ticker).info
                desc = (info.get("longBusinessSummary", "") + " " + info.get("industry", "")).lower()
                for sector, kws in SECTOR_KEYWORDS.items():
                    if any(k.lower() in desc for k in kws):
                        sector_count[sector] = sector_count.get(sector, 0) + 1
                time.sleep(0.2)
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"Unusual Whales 실패: {e}")

    return dict(sorted(sector_count.items(), key=lambda x: x[1], reverse=True))


def fetch_finviz_option_unusual() -> dict[str, int]:
    """Finviz 옵션 탭에서 이상 거래 감지."""
    sector_count: dict[str, int] = {}
    try:
        r = requests.get("https://finviz.com/screener.ashx?v=111&s=ta_unusualvolume",
                         headers=HEADERS, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        import yfinance as yf
        tickers = [a.get_text(strip=True) for a in soup.select("a.screener-link-primary")][:50]
        for ticker in tickers:
            try:
                info = yf.Ticker(ticker).info
                desc = (info.get("longBusinessSummary", "") + " " + info.get("industry", "")).lower()
                for sector, kws in SECTOR_KEYWORDS.items():
                    if any(k.lower() in desc for k in kws):
                        sector_count[sector] = sector_count.get(sector, 0) + 1
                time.sleep(0.2)
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"Finviz 이상거래 실패: {e}")
    return sector_count


def fetch_form4_insider_buys() -> dict[str, int]:
    """SEC Form 4 내부자 매수 → 섹터별 집계."""
    sector_count: dict[str, int] = {}
    try:
        base = "https://www.sec.gov"
        r = requests.get(
            f"{base}/cgi-bin/browse-edgar?action=getcompany&type=4&dateb=&owner=include&count=100&search_text=",
            headers={"User-Agent": "stock-research-bot hideinthecodes@gmail.com"},
            timeout=20,
        )
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        import yfinance as yf
        for link in soup.select("a[href*='/cgi-bin/browse-edgar?action=getcompany']")[:30]:
            company = link.get_text(strip=True)
            if not company:
                continue
            try:
                info = yf.Ticker(company).info
                desc = (info.get("longBusinessSummary", "") + " " + info.get("industry", "")).lower()
                for sector, kws in SECTOR_KEYWORDS.items():
                    if any(k.lower() in desc for k in kws):
                        sector_count[sector] = sector_count.get(sector, 0) + 1
                time.sleep(0.2)
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"Form 4 수집 실패: {e}")
    return sector_count


def fetch_crunchbase_sectors() -> dict[str, int]:
    """Crunchbase 트렌딩 분야 스크래핑 (공개 페이지)."""
    sector_count: dict[str, int] = {}
    try:
        r = requests.get("https://www.crunchbase.com/discover/funding_rounds", headers=HEADERS, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text().lower()
        for sector, kws in SECTOR_KEYWORDS.items():
            count = sum(text.count(k.lower()) for k in kws)
            if count > 0:
                sector_count[sector] = count
    except Exception as e:
        logger.debug(f"Crunchbase 실패: {e}")
    return dict(sorted(sector_count.items(), key=lambda x: x[1], reverse=True))


def run() -> dict:
    logger.info("=== 스마트머니 선행 지표 수집 ===")

    logger.info("  Unusual Whales 옵션 플로우...")
    unusual = fetch_unusual_whales_sectors()

    logger.info("  Finviz 이상 거래량...")
    finviz_opt = fetch_finviz_option_unusual()

    logger.info("  SEC Form 4 내부자 매수...")
    form4 = fetch_form4_insider_buys()

    logger.info("  Crunchbase VC 펀딩 트렌드...")
    vc = fetch_crunchbase_sectors()

    all_sectors = set(list(unusual.keys()) + list(finviz_opt.keys()) + list(form4.keys()) + list(vc.keys()))
    combined: dict[str, dict] = {}
    for s in all_sectors:
        combined[s] = {
            "options_unusual": unusual.get(s, 0),
            "finviz_vol": finviz_opt.get(s, 0),
            "insider_buy": form4.get(s, 0),
            "vc_funding": vc.get(s, 0),
            "total": unusual.get(s, 0) + finviz_opt.get(s, 0) + form4.get(s, 0),
        }

    sorted_combined = dict(sorted(combined.items(), key=lambda x: x[1]["total"], reverse=True))

    output = {
        "updated": datetime.now().isoformat(),
        "smart_money_signals": sorted_combined,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    logger.success(f"저장 → {OUT}")
    return output


if __name__ == "__main__":
    run()
