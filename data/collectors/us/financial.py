import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))

"""
미국 주식 재무제표 수집 (yfinance)
PBR, PSR, ROE, 매출 성장률 등 텐버거 핵심 지표 계산
"""
import time
import yfinance as yf
import pandas as pd
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from config.settings import YFINANCE_DELAY
from data.collectors.us.universe import load_universe

SAVE_DIR = Path(__file__).parents[3] / "datasets" / "us" / "financial"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def get_financials(ticker: str) -> dict | None:
    try:
        t = yf.Ticker(ticker)
        info = t.info

        income = t.financials          # 손익계산서
        balance = t.balance_sheet      # 대차대조표
        cashflow = t.cashflow          # 현금흐름

        ratios = _calc_ratios(info, income, balance)
        return {
            "ticker": ticker,
            "info": info,
            "ratios": ratios,
            "income": income,
            "balance": balance,
            "cashflow": cashflow,
        }
    except Exception as e:
        logger.warning(f"{ticker} 재무 실패: {e}")
        return None


def _calc_ratios(info: dict, income: pd.DataFrame, balance: pd.DataFrame) -> dict:
    """핵심 지표 계산"""
    def safe(key, default=None):
        return info.get(key, default)

    ratios = {
        "market_cap":        safe("marketCap"),
        "price":             safe("currentPrice") or safe("regularMarketPrice"),
        "pbr":               safe("priceToBook"),
        "per":               safe("trailingPE"),
        "psr":               safe("priceToSalesTrailing12Months"),
        "roe":               safe("returnOnEquity"),
        "roa":               safe("returnOnAssets"),
        "debt_to_equity":    safe("debtToEquity"),
        "current_ratio":     safe("currentRatio"),
        "revenue_growth":    safe("revenueGrowth"),
        "earnings_growth":   safe("earningsGrowth"),
        "gross_margin":      safe("grossMargins"),
        "operating_margin":  safe("operatingMargins"),
        "profit_margin":     safe("profitMargins"),
        "sector":            safe("sector"),
        "industry":          safe("industry"),
        "country":           safe("country"),
        "employees":         safe("fullTimeEmployees"),
        "short_ratio":       safe("shortRatio"),        # 공매도 비율
        "shares_outstanding":safe("sharesOutstanding"),  # 발행 주식수
        "float_shares":      safe("floatShares"),        # 유통 주식수
    }

    # 매출 성장률 (직접 계산, 최근 2년)
    if income is not None and "Total Revenue" in income.index and income.shape[1] >= 2:
        rev = income.loc["Total Revenue"]
        if len(rev) >= 2 and rev.iloc[1] and rev.iloc[1] != 0:
            ratios["revenue_growth_calc"] = (rev.iloc[0] - rev.iloc[1]) / abs(rev.iloc[1])

    return ratios


def download_all(tickers: list[str] | None = None) -> pd.DataFrame:
    if tickers is None:
        universe = load_universe()
        tickers = universe["ticker"].tolist()

    existing = {p.stem for p in SAVE_DIR.glob("*.parquet")}
    tickers = [t for t in tickers if t not in existing]
    logger.info(f"재무 수집 대상: {len(tickers):,}개")

    rows = []
    failed = []
    for ticker in tqdm(tickers, desc="US 재무 수집"):
        result = get_financials(ticker)
        if result:
            rows.append(result["ratios"] | {"ticker": ticker})
            # 개별 상세 저장
            pd.DataFrame([result["ratios"]]).to_parquet(SAVE_DIR / f"{ticker}.parquet")
        else:
            failed.append(ticker)
        time.sleep(YFINANCE_DELAY)

    if rows:
        summary = pd.DataFrame(rows)
        summary.to_parquet(SAVE_DIR / "_summary.parquet", index=False)
        logger.info(f"요약 저장: {SAVE_DIR}/_summary.parquet")

    logger.info(f"완료. 실패: {len(failed)}개")
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_summary() -> pd.DataFrame:
    path = SAVE_DIR / "_summary.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


if __name__ == "__main__":
    test = ["AAPL", "NVDA", "IONQ", "SHLS", "DAVE"]
    df = download_all(tickers=test)
    print(df[["ticker", "pbr", "psr", "roe", "revenue_growth"]].to_string())
