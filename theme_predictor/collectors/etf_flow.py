import json
import time
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime
from loguru import logger

import sys; sys.path.insert(0, str(Path(__file__).parents[2]))
from theme_predictor.config import SECTOR_ETFS, DELAY

OUT = Path(__file__).parents[1] / "signals" / "etf_inflow.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def fetch_etf_aum_yfinance(ticker: str) -> dict | None:
    """yfinance로 ETF 기본 데이터 (AUM, 최근 거래량) 수집."""
    try:
        info = yf.Ticker(ticker).info
        hist = yf.Ticker(ticker).history(period="1mo")
        avg_vol_1m = float(hist["Volume"].mean()) if not hist.empty else 0
        avg_vol_1w = float(hist["Volume"].iloc[-5:].mean()) if len(hist) >= 5 else avg_vol_1m
        return {
            "ticker": ticker,
            "aum": info.get("totalAssets"),
            "price": info.get("navPrice") or info.get("regularMarketPrice"),
            "avg_vol_1m": avg_vol_1m,
            "avg_vol_1w": avg_vol_1w,
            "vol_surge": round(avg_vol_1w / avg_vol_1m, 2) if avg_vol_1m > 0 else 1.0,
        }
    except Exception as e:
        logger.debug(f"{ticker} yfinance 실패: {e}")
        return None


def fetch_etfdb_flows(ticker: str) -> dict | None:
    """ETFdb.com에서 ETF 자금 유입 스크래핑."""
    try:
        url = f"https://etfdb.com/etf/{ticker}/#etf-ticker-profile"
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        flow_data = {}
        for row in soup.select("table tr"):
            cells = row.find_all("td")
            if len(cells) >= 2:
                label = cells[0].get_text(strip=True)
                val = cells[1].get_text(strip=True)
                if "Fund Flows" in label or "1 Week" in label or "1 Month" in label:
                    flow_data[label] = val
        return flow_data if flow_data else None
    except Exception as e:
        logger.debug(f"{ticker} ETFdb 스크래핑 실패: {e}")
        return None


def analyze_sector_flows() -> dict[str, dict]:
    """섹터별 ETF 자금 유입 집계."""
    sector_results: dict[str, dict] = {}

    for sector, etfs in SECTOR_ETFS.items():
        sector_vol_surge = []
        sector_aum_total = 0
        etf_details = []

        for ticker in etfs:
            data = fetch_etf_aum_yfinance(ticker)
            if data:
                sector_vol_surge.append(data["vol_surge"])
                if data["aum"]:
                    sector_aum_total += data["aum"]
                etf_details.append(data)
            time.sleep(DELAY)

        avg_surge = round(sum(sector_vol_surge) / len(sector_vol_surge), 2) if sector_vol_surge else 1.0
        sector_results[sector] = {
            "avg_vol_surge": avg_surge,
            "total_aum_usd": sector_aum_total,
            "etf_count": len(etfs),
            "etfs": etf_details,
        }
        logger.info(f"  {sector:12s}: 거래량 배수 {avg_surge:.2f}x, AUM ${sector_aum_total/1e9:.1f}B")

    return dict(sorted(sector_results.items(), key=lambda x: x[1]["avg_vol_surge"], reverse=True))


def run() -> dict:
    logger.info("=== ETF 자금 유입 분석 ===")
    flows = analyze_sector_flows()

    output = {
        "updated": datetime.now().isoformat(),
        "sector_etf_flows": flows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    logger.success(f"저장 → {OUT}")
    return output


if __name__ == "__main__":
    run()
