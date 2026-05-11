"""
한국 주식 재무제표 수집 (DART API + OpenDartReader)
PBR, PSR, ROE 등 텐버거 핵심 지표
"""
import time
import OpenDartReader as dart_reader
import pandas as pd
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from config.settings import DART_API_KEY
from data.collectors.kr.universe import load_universe

SAVE_DIR = Path(__file__).parents[3] / "datasets" / "kr" / "financial"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

dart = dart_reader.OpenDartReader(DART_API_KEY)

# 수집할 재무 항목 (IFRS 기준)
FINANCIAL_ITEMS = [
    "매출액", "영업이익", "당기순이익",
    "자산총계", "부채총계", "자본총계",
    "현금및현금성자산",
]


def get_financials(corp_code: str, year: int = None) -> pd.DataFrame | None:
    import datetime
    if year is None:
        year = datetime.date.today().year - 1  # 전년도

    try:
        df = dart.finstate(corp_code, year)
        if df is None or df.empty:
            return None
        return df
    except Exception as e:
        logger.warning(f"{corp_code} 재무 실패: {e}")
        return None


def get_corp_code(ticker: str) -> str | None:
    try:
        info = dart.find_corp_code(ticker)
        return info if info else None
    except Exception:
        return None


def download_all(tickers: list[str] | None = None) -> pd.DataFrame:
    if tickers is None:
        universe = load_universe()
        tickers = universe["ticker"].tolist()

    existing = {p.stem for p in SAVE_DIR.glob("*.parquet")}
    tickers  = [t for t in tickers if t not in existing]
    logger.info(f"KR 재무 수집 대상: {len(tickers):,}개")

    rows = []
    failed = []

    for ticker in tqdm(tickers, desc="KR 재무 수집"):
        corp_code = get_corp_code(ticker)
        if not corp_code:
            failed.append(ticker)
            continue

        df = get_financials(corp_code)
        if df is not None and not df.empty:
            df["ticker"] = ticker
            df.to_parquet(SAVE_DIR / f"{ticker}.parquet", index=False)
            rows.append({"ticker": ticker, "corp_code": corp_code, "rows": len(df)})
        else:
            failed.append(ticker)

        time.sleep(0.2)

    logger.info(f"완료. 실패: {len(failed)}개")
    return pd.DataFrame(rows)


def load_financials(ticker: str) -> pd.DataFrame | None:
    path = SAVE_DIR / f"{ticker}.parquet"
    return pd.read_parquet(path) if path.exists() else None


if __name__ == "__main__":
    test = ["005930", "000660"]  # 삼성전자, SK하이닉스
    download_all(tickers=test)
    print(load_financials("005930"))
