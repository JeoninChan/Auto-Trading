import json
import time
import requests
import yfinance as yf
import pandas as pd
from pathlib import Path
from loguru import logger
from datetime import datetime

sys_path = Path(__file__).parents[2]
import sys; sys.path.insert(0, str(sys_path))
from theme_predictor.config import TRACKED_FUNDS, SECTOR_ETFS, DELAY

HEADERS = {"User-Agent": "stock-research-bot hideinthecodes@gmail.com"}
BASE = "https://data.sec.gov"
OUT = Path(__file__).parents[1] / "signals" / "13f_sector.json"


def get_latest_filing(cik: str) -> dict | None:
    url = f"{BASE}/submissions/CIK{cik.zfill(10)}.json"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        acc_nums = filings.get("accessionNumber", [])
        dates = filings.get("filingDate", [])
        for i, form in enumerate(forms):
            if form == "13F-HR":
                return {"accession": acc_nums[i].replace("-", ""), "date": dates[i], "cik": cik}
        return None
    except Exception as e:
        logger.debug(f"CIK {cik} 조회 실패: {e}")
        return None


def fetch_holdings(cik: str, accession: str) -> list[dict]:
    cik_pad = cik.zfill(10)
    idx_url = f"{BASE}/Archives/edgar/data/{int(cik)}/{accession}/index.json"
    try:
        r = requests.get(idx_url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        files = r.json().get("directory", {}).get("item", [])
        xml_file = next((f["name"] for f in files if f["name"].endswith(".xml") and "infotable" in f["name"].lower()), None)
        if not xml_file:
            xml_file = next((f["name"] for f in files if f["name"].endswith(".xml") and f["name"] != "primary_doc.xml"), None)
        if not xml_file:
            return []
        xml_url = f"{BASE}/Archives/edgar/data/{int(cik)}/{accession}/{xml_file}"
        rx = requests.get(xml_url, headers=HEADERS, timeout=30)
        rx.raise_for_status()
        return _parse_xml(rx.text)
    except Exception as e:
        logger.debug(f"holdings 파싱 실패 ({cik}): {e}")
        return []


def _parse_xml(text: str) -> list[dict]:
    import xml.etree.ElementTree as ET
    holdings = []
    try:
        root = ET.fromstring(text)
        ns = {"ns": root.tag.split("}")[0].lstrip("{")} if "}" in root.tag else {}
        tag = lambda t: f"{{ns}}{t}" if ns else t
        for info in root.iter():
            if info.tag.split("}")[-1] == "infoTable":
                name = info.find(".//{*}nameOfIssuer")
                cusip = info.find(".//{*}cusip")
                val = info.find(".//{*}value")
                shares = info.find(".//{*}sshPrnamt")
                holdings.append({
                    "name": name.text.strip() if name is not None else "",
                    "cusip": cusip.text.strip() if cusip is not None else "",
                    "value_usd": int(val.text) * 1000 if val is not None and val.text else 0,
                    "shares": int(shares.text) if shares is not None and shares.text else 0,
                })
    except Exception as e:
        logger.debug(f"XML 파싱 오류: {e}")
    return holdings


def ticker_to_sector(ticker: str) -> str:
    try:
        info = yf.Ticker(ticker).info
        return info.get("sector", "") or info.get("industry", "") or "Unknown"
    except Exception:
        return "Unknown"


def classify_by_sector_etf(ticker: str) -> list[str]:
    matched = []
    try:
        info = yf.Ticker(ticker).info
        desc = (info.get("longBusinessSummary", "") + " " + info.get("industry", "")).lower()
        from theme_predictor.config import SECTOR_KEYWORDS
        for sector, kws in SECTOR_KEYWORDS.items():
            if any(k.lower() in desc for k in kws):
                matched.append(sector)
    except Exception:
        pass
    return matched or ["기타"]


def run() -> dict:
    logger.info("=== 13F 섹터 자금 이동 분석 ===")
    results = {}
    sector_flow: dict[str, int] = {}

    for fund_name, cik in TRACKED_FUNDS.items():
        logger.info(f"  {fund_name} 조회 중...")
        filing = get_latest_filing(cik)
        if not filing:
            logger.warning(f"  {fund_name}: 13F 없음")
            continue
        time.sleep(DELAY)

        holdings = fetch_holdings(cik, filing["accession"])
        logger.info(f"  {fund_name}: {len(holdings)}개 종목, 기준일 {filing['date']}")

        results[fund_name] = {"date": filing["date"], "holdings_count": len(holdings), "top": holdings[:20]}

        for h in holdings[:50]:
            sectors = classify_by_sector_etf(h.get("name", ""))
            for s in sectors:
                sector_flow[s] = sector_flow.get(s, 0) + h.get("value_usd", 0)
        time.sleep(DELAY)

    sorted_flow = dict(sorted(sector_flow.items(), key=lambda x: x[1], reverse=True))

    output = {
        "updated": datetime.now().isoformat(),
        "sector_flow_usd": sorted_flow,
        "fund_detail": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    logger.success(f"저장 → {OUT}")
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cik", default=None, help="특정 CIK만 조회")
    args = parser.parse_args()

    if args.cik:
        filing = get_latest_filing(args.cik)
        if filing:
            h = fetch_holdings(args.cik, filing["accession"])
            print(f"보유 종목 {len(h)}개, 최신 10개:")
            for item in h[:10]:
                print(f"  {item['name']:40s} ${item['value_usd']:>15,}")
    else:
        run()
