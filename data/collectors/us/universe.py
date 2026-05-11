"""
미국 주식 유니버스 관리
SEC EDGAR에서 전체 상장 종목 목록 수집
"""
import time
import requests
import pandas as pd
from pathlib import Path
from loguru import logger

EDGAR_URL = "https://www.sec.gov/files/company_tickers.json"
HEADERS   = {"User-Agent": "AutoTrading contact@autotrading.com"}
SAVE_PATH = Path(__file__).parents[3] / "datasets" / "us" / "universe.parquet"


def fetch() -> pd.DataFrame:
    logger.info("SEC EDGAR 전체 종목 목록 다운로드...")
    resp = requests.get(EDGAR_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    df = pd.DataFrame.from_dict(resp.json(), orient="index")
    df.columns = ["cik", "ticker", "company"]
    df["ticker"] = df["ticker"].str.upper().str.strip()
    logger.info(f"{len(df):,}개 수신")
    return df


def get_universe(save: bool = True) -> pd.DataFrame:
    """
    텐버거 탐색용 유니버스
    - 점(.) 또는 하이픈(-) 포함 티커 제외 (OTC, 우선주 등)
    """
    df = fetch()
    df = df[~df["ticker"].str.contains(r"[.\-]", regex=True)]
    df = df.reset_index(drop=True)
    logger.info(f"OTC 제외 후 {len(df):,}개")

    if save:
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(SAVE_PATH, index=False)
        logger.info(f"저장 완료: {SAVE_PATH}")

    return df


def load_universe() -> pd.DataFrame:
    if SAVE_PATH.exists():
        return pd.read_parquet(SAVE_PATH)
    return get_universe()


if __name__ == "__main__":
    df = get_universe()
    print(df.head())
    print(f"\n총 {len(df):,}개 종목")
