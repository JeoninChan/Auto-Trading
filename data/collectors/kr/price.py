import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))

"""
한국 주식 가격·거래량 수집 (pykrx)
KOSPI + KOSDAQ 일봉
"""
import time
import pandas as pd
from pathlib import Path
from pykrx import stock
from loguru import logger
from tqdm import tqdm

from config.settings import KR_PRICE_START
from data.collectors.kr.universe import load_universe

SAVE_DIR = Path(__file__).parents[3] / "datasets" / "kr" / "price"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

import datetime
END_DATE = datetime.date.today().strftime("%Y%m%d")


def download_ticker(ticker: str) -> pd.DataFrame | None:
    try:
        df = stock.get_market_ohlcv(KR_PRICE_START, END_DATE, ticker)
        if df.empty:
            return None
        df.index.name = "date"
        # pykrx 버전에 따라 컬럼 수 다를 수 있어 동적으로 매핑
        col_map = {
            "시가": "open", "고가": "high", "저가": "low",
            "종가": "close", "거래량": "volume",
            "거래대금": "trading_value", "등락률": "price_change_pct"
        }
        df = df.rename(columns=col_map)
        return df
    except Exception as e:
        logger.warning(f"{ticker} 가격 실패: {e}")
        return None


def download_all(tickers: list[str] | None = None) -> None:
    if tickers is None:
        universe = load_universe()
        tickers = universe["ticker"].tolist()

    existing = {p.stem for p in SAVE_DIR.glob("*.parquet")}
    tickers  = [t for t in tickers if t not in existing]
    logger.info(f"KR 가격 수집 대상: {len(tickers):,}개 (기존 {len(existing):,}개 스킵)")

    failed = []
    for ticker in tqdm(tickers, desc="KR 가격 수집"):
        df = download_ticker(ticker)
        if df is not None:
            df.to_parquet(SAVE_DIR / f"{ticker}.parquet")
        else:
            failed.append(ticker)
        time.sleep(0.1)

    logger.info(f"완료. 실패: {len(failed)}개")
    if failed:
        pd.Series(failed).to_csv(SAVE_DIR / "failed.csv", index=False)


def load_price(ticker: str) -> pd.DataFrame | None:
    path = SAVE_DIR / f"{ticker}.parquet"
    return pd.read_parquet(path) if path.exists() else None


if __name__ == "__main__":
    test = ["005930", "000660", "035720"]  # 삼성전자, SK하이닉스, 카카오
    download_all(tickers=test)
    print(load_price("005930").tail())
