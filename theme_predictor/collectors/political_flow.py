import json
import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger

import sys; sys.path.insert(0, str(Path(__file__).parents[2]))
from theme_predictor.config import SECTOR_KEYWORDS, DELAY

OUT = Path(__file__).parents[1] / "signals" / "political_flow.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def fetch_capitol_trades(days: int = 90) -> list[dict]:
    """Capitol Trades에서 최근 의원 거래 내역 파싱."""
    trades = []
    page = 1
    cutoff = datetime.now() - timedelta(days=days)

    while page <= 5:
        url = f"https://www.capitoltrades.com/trades?page={page}&pageSize=96"
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            rows = soup.select("table tbody tr") or soup.select(".trade-row")

            if not rows:
                break

            for row in rows:
                cells = row.find_all("td")
                if len(cells) < 5:
                    continue
                trade_date_str = cells[0].get_text(strip=True) if cells else ""
                try:
                    trade_date = datetime.strptime(trade_date_str[:10], "%Y-%m-%d")
                    if trade_date < cutoff:
                        return trades
                except ValueError:
                    pass

                trades.append({
                    "date": trade_date_str,
                    "politician": cells[1].get_text(strip=True) if len(cells) > 1 else "",
                    "ticker": cells[2].get_text(strip=True) if len(cells) > 2 else "",
                    "action": cells[3].get_text(strip=True) if len(cells) > 3 else "",
                    "amount": cells[4].get_text(strip=True) if len(cells) > 4 else "",
                })
            page += 1
            time.sleep(DELAY)
        except Exception as e:
            logger.debug(f"Capitol Trades 페이지 {page} 실패: {e}")
            break

    return trades


def fetch_quiver_congress() -> list[dict]:
    """Quiver Quantitative 무료 티어 의회 거래 데이터."""
    try:
        url = "https://www.quiverquant.com/sources/congresstrading"
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        rows = soup.select("table tbody tr")
        result = []
        for row in rows[:100]:
            cells = row.find_all("td")
            if len(cells) >= 5:
                result.append({
                    "date": cells[0].get_text(strip=True),
                    "politician": cells[1].get_text(strip=True),
                    "ticker": cells[2].get_text(strip=True),
                    "action": cells[3].get_text(strip=True),
                    "amount": cells[4].get_text(strip=True),
                })
        return result
    except Exception as e:
        logger.debug(f"Quiver Congress 실패: {e}")
        return []


def classify_ticker_sector(ticker: str) -> list[str]:
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        desc = (info.get("longBusinessSummary", "") + " " + info.get("industry", "")).lower()
        matched = [s for s, kws in SECTOR_KEYWORDS.items() if any(k.lower() in desc for k in kws)]
        return matched or ["기타"]
    except Exception:
        return ["기타"]


def aggregate_sector_buys(trades: list[dict]) -> dict[str, int]:
    sector_count: dict[str, int] = {}
    buy_trades = [t for t in trades if "purchase" in t.get("action", "").lower() or "buy" in t.get("action", "").lower()]
    tickers_seen = set()

    for t in buy_trades:
        ticker = t.get("ticker", "").upper().strip()
        if not ticker or ticker in tickers_seen or len(ticker) > 6:
            continue
        tickers_seen.add(ticker)
        sectors = classify_ticker_sector(ticker)
        for s in sectors:
            sector_count[s] = sector_count.get(s, 0) + 1
        time.sleep(0.3)

    return dict(sorted(sector_count.items(), key=lambda x: x[1], reverse=True))


def run() -> dict:
    logger.info("=== 의회/정치 자금 흐름 수집 ===")

    trades = fetch_capitol_trades(days=90)
    logger.info(f"Capitol Trades: {len(trades)}건")

    if len(trades) < 10:
        logger.info("Capitol Trades 부족 → Quiver 보조 수집")
        trades += fetch_quiver_congress()

    sector_buys = aggregate_sector_buys(trades)

    output = {
        "updated": datetime.now().isoformat(),
        "period_days": 90,
        "total_trades": len(trades),
        "sector_buy_count": sector_buys,
        "raw_trades": trades[:50],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    logger.success(f"저장 → {OUT}")
    return output


if __name__ == "__main__":
    run()
