import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))

"""
미국 주식 가격·거래량 수집 (yfinance)
일봉 기준, 전체 유니버스 대상
"""
import time
import yfinance as yf
import pandas as pd
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from config.settings import US_PRICE_PERIOD, US_PRICE_INTERVAL, YFINANCE_DELAY
from data.collectors.us.universe import load_universe

SAVE_DIR = Path(__file__).parents[3] / "datasets" / "us" / "price"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def download_ticker(ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(
            ticker,
            period=US_PRICE_PERIOD,
            interval=US_PRICE_INTERVAL,
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            return None
        df.index.name = "date"
        return df
    except Exception as e:
        logger.warning(f"{ticker} 실패: {e}")
        return None


def download_all(tickers: list[str] | None = None, batch_size: int = 50) -> None:
    """
    전체 또는 지정 티커 일봉 다운로드
    배치 단위로 저장 → 중간에 끊겨도 재시작 가능
    """
    if tickers is None:
        universe = load_universe()
        tickers = universe["ticker"].tolist()

    # 이미 다운로드된 티커 건너뛰기
    existing = {p.stem for p in SAVE_DIR.glob("*.parquet")}
    tickers = [t for t in tickers if t not in existing]
    logger.info(f"다운로드 대상: {len(tickers):,}개 (기존 {len(existing):,}개 스킵)")

    failed = []
    for i, ticker in enumerate(tqdm(tickers, desc="US 가격 수집")):
        df = download_ticker(ticker)
        if df is not None and not df.empty:
            df.to_parquet(SAVE_DIR / f"{ticker}.parquet")
        else:
            failed.append(ticker)

        # 배치마다 짧은 대기
        if (i + 1) % batch_size == 0:
            time.sleep(YFINANCE_DELAY * 3)
        else:
            time.sleep(YFINANCE_DELAY)

    logger.info(f"완료. 실패: {len(failed)}개")
    if failed:
        pd.Series(failed).to_csv(SAVE_DIR / "failed.csv", index=False)


def load_price(ticker: str) -> pd.DataFrame | None:
    path = SAVE_DIR / f"{ticker}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return None


if __name__ == "__main__":
    # 특정 종목만 테스트
    test = ["AAPL", "NVDA", "IONQ", "NVDL", "SOXL"]
    download_all(tickers=test)
