"""
한국 주식 유니버스 관리 (pykrx)
KOSPI + KOSDAQ 전 종목
"""
import pandas as pd
from pathlib import Path
from pykrx import stock
from loguru import logger

SAVE_PATH = Path(__file__).parents[3] / "datasets" / "kr" / "universe.parquet"


def get_universe(save: bool = True) -> pd.DataFrame:
    logger.info("KRX 전 종목 목록 수집...")

    kospi  = stock.get_market_ticker_list(market="KOSPI")
    kosdaq = stock.get_market_ticker_list(market="KOSDAQ")

    rows = []
    for ticker in kospi:
        rows.append({"ticker": ticker, "name": stock.get_market_ticker_name(ticker), "market": "KOSPI"})
    for ticker in kosdaq:
        rows.append({"ticker": ticker, "name": stock.get_market_ticker_name(ticker), "market": "KOSDAQ"})

    df = pd.DataFrame(rows)
    logger.info(f"KOSPI {len(kospi)}개 + KOSDAQ {len(kosdaq)}개 = 총 {len(df)}개")

    if save:
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(SAVE_PATH, index=False)
        logger.info(f"저장: {SAVE_PATH}")

    return df


def load_universe() -> pd.DataFrame:
    if SAVE_PATH.exists():
        return pd.read_parquet(SAVE_PATH)
    return get_universe()


if __name__ == "__main__":
    df = get_universe()
    print(df.head(10))
    print(f"\n총 {len(df):,}개")
