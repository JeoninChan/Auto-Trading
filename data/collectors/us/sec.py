import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))

"""
SEC EDGAR 공시 수집 (10-K, 10-Q, 8-K, S-3, S-8)
희석 위험(dilution risk) 포함
API 키 불필요, 10 req/sec 제한
"""
import time
import requests
import pandas as pd
from pathlib import Path
from loguru import logger

from config.settings import SEC_RATE_LIMIT

HEADERS  = {"User-Agent": "AutoTrading contact@autotrading.com"}
BASE_URL = "https://data.sec.gov"
SAVE_DIR = Path(__file__).parents[3] / "datasets" / "us" / "sec"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 텐버거 분석에 중요한 공시 유형
TARGET_FORMS = {"10-K", "10-Q", "8-K", "S-3", "S-8", "DEF 14A"}


def get_cik(ticker: str) -> str | None:
    """ticker → CIK 변환 (EDGAR company_tickers 활용)"""
    from data.collectors.us.universe import load_universe
    universe = load_universe()
    row = universe[universe["ticker"] == ticker.upper()]
    if row.empty:
        return None
    return str(row.iloc[0]["cik"]).zfill(10)


def get_filings(cik: str, forms: set = TARGET_FORMS) -> pd.DataFrame:
    """CIK의 공시 목록 가져오기"""
    url = f"{BASE_URL}/submissions/CIK{cik}.json"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"CIK {cik} 공시 목록 실패: {e}")
        return pd.DataFrame()

    recent = data.get("filings", {}).get("recent", {})
    if not recent:
        return pd.DataFrame()

    df = pd.DataFrame(recent)
    if "form" not in df.columns:
        return pd.DataFrame()

    df = df[df["form"].isin(forms)].copy()
    df["cik"] = cik
    return df[["cik", "form", "filingDate", "accessionNumber", "primaryDocument"]]


def download_filing_text(cik: str, accession: str, doc: str) -> str | None:
    """공시 원문 텍스트 다운로드"""
    accession_fmt = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_fmt}/{doc}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        logger.warning(f"공시 원문 실패 ({accession}): {e}")
        return None


def collect_ticker(ticker: str, max_filings: int = 10) -> pd.DataFrame:
    """
    특정 종목의 최근 공시 수집
    - 공시 목록 저장
    - 8-K(중요 이벤트), S-3/S-8(희석 위험) 원문 저장
    """
    cik = get_cik(ticker)
    if not cik:
        logger.warning(f"{ticker}: CIK 없음")
        return pd.DataFrame()

    save_ticker_dir = SAVE_DIR / ticker
    save_ticker_dir.mkdir(exist_ok=True)

    filings = get_filings(cik)
    if filings.empty:
        return pd.DataFrame()

    filings = filings.head(max_filings)
    filings.to_parquet(save_ticker_dir / "filings.parquet", index=False)

    # 희석 위험 관련 공시(S-3, S-8) + 중요 이벤트(8-K) 원문 → 클리닝 후 저장
    from data.processors.sec_cleaner import clean_sec_html
    priority = filings[filings["form"].isin({"8-K", "S-3", "S-8"})]
    for _, row in priority.iterrows():
        time.sleep(SEC_RATE_LIMIT)
        text = download_filing_text(cik, row["accessionNumber"], row["primaryDocument"])
        if text:
            cleaned = clean_sec_html(text)
            fname = f"{row['filingDate']}_{row['form'].replace(' ','_')}.txt"
            (save_ticker_dir / fname).write_text(cleaned[:30000], encoding="utf-8")

    logger.info(f"{ticker}: {len(filings)}개 공시 수집 완료")
    time.sleep(SEC_RATE_LIMIT)
    return filings


def collect_all(tickers: list[str]) -> None:
    logger.info(f"SEC 공시 수집 시작: {len(tickers)}개 종목")
    for ticker in tickers:
        existing = SAVE_DIR / ticker / "filings.parquet"
        if existing.exists():
            continue
        collect_ticker(ticker)


if __name__ == "__main__":
    test = ["NVDA", "IONQ", "SHLS", "DAVE"]
    for t in test:
        df = collect_ticker(t)
        print(f"\n{t}: {len(df)}개 공시")
        if not df.empty:
            print(df[["form", "filingDate"]].head())
