import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))

"""
한국 주식 공시 수집 (DART API)
수시공시 + 정기공시 (주요사항보고서, 사업보고서 등)
"""
import time
import OpenDartReader as dart_reader
import pandas as pd
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from config.settings import DART_API_KEY
from data.collectors.kr.universe import load_universe

SAVE_DIR = Path(__file__).parents[3] / "datasets" / "kr" / "disclosure"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

dart = dart_reader.OpenDartReader(DART_API_KEY)

# 텐버거/단타에 중요한 공시 유형
IMPORTANT_REPORTS = [
    "주요사항보고서",  # 유상증자, 전환사채 → 희석 위험
    "사업보고서",      # 연간 실적
    "반기보고서",
    "분기보고서",
    "공정공시",        # 실적 전망 등
]


def get_disclosures(ticker: str, start: str = "20220101") -> pd.DataFrame | None:
    try:
        df = dart.list(ticker, start=start, kind="A")  # 정기공시
        df2 = dart.list(ticker, start=start, kind="B") # 주요사항
        combined = pd.concat([df, df2], ignore_index=True) if df is not None and df2 is not None else (df or df2)
        return combined
    except Exception as e:
        logger.warning(f"{ticker} 공시 목록 실패: {e}")
        return None


def download_all(tickers: list[str] | None = None) -> None:
    if tickers is None:
        universe = load_universe()
        tickers = universe["ticker"].tolist()

    existing = {p.stem for p in SAVE_DIR.glob("*.parquet")}
    tickers  = [t for t in tickers if t not in existing]
    logger.info(f"KR 공시 수집 대상: {len(tickers):,}개")

    failed = []
    for ticker in tqdm(tickers, desc="KR 공시 수집"):
        df = get_disclosures(ticker)
        if df is not None and not df.empty:
            df["ticker"] = ticker
            df.to_parquet(SAVE_DIR / f"{ticker}.parquet", index=False)
        else:
            failed.append(ticker)
        time.sleep(0.3)

    logger.info(f"완료. 실패: {len(failed)}개")


def load_disclosures(ticker: str) -> pd.DataFrame | None:
    path = SAVE_DIR / f"{ticker}.parquet"
    return pd.read_parquet(path) if path.exists() else None


if __name__ == "__main__":
    test = ["005930", "000660", "035720"]
    download_all(tickers=test)
    print(load_disclosures("005930"))
