"""
US 우량주 스크리너 — 시총 5,000억원 ~ 28조원

필터 기준:
  - 시총: KRW 5,000억 ~ 28조 (환율 기준 USD 변환)
  - 부채비율(D/E): ≤ 100
  - 유동비율: ≥ 1.0
  - 매출총이익률: ≥ 10%
  - 영업이익률: ≥ 10%
  - 순이익률: ≥ 5%
  - 매출성장률: ≥ 20%

사용법:
    python screener/us_smallmid_screen.py
    python screener/us_smallmid_screen.py --sample 100
"""
import sys
import argparse
import time
import random
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parents[1]))

import yfinance as yf
import pandas as pd
from loguru import logger
from tqdm import tqdm

MARKET_CAP_MIN_KRW = 500_000_000_000     # 5,000억원
MARKET_CAP_MAX_KRW = 28_000_000_000_000  # 28조원
DELAY              = 0.3
OUTPUT_DIR         = Path(__file__).parent / "output"
UNIVERSE_PATH      = Path(__file__).parents[1] / "datasets" / "us" / "universe.parquet"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fmt_usd(amount) -> str:
    if amount is None or (isinstance(amount, float) and pd.isna(amount)):
        return "N/A"
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return "N/A"
    eok = int(amount // 100_000_000)
    man = int((amount % 100_000_000) // 10_000)
    if eok > 0:
        return f"{eok}억 {man:,}만달러" if man > 0 else f"{eok}억달러"
    return f"{man:,}만달러"


def fmt_pct(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def fmt_peg(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "N/A"


def fmt_ratio(numerator, denominator) -> str:
    if numerator is None or denominator is None or denominator == 0:
        return "N/A"
    try:
        return f"{float(numerator) / float(denominator) * 100:.1f}%"
    except (TypeError, ValueError):
        return "N/A"


def get_krw_rate() -> float:
    try:
        rate = yf.Ticker("KRW=X").fast_info["lastPrice"]
        logger.info(f"USD/KRW 환율: {rate:,.0f}")
        return float(rate)
    except Exception:
        logger.warning("환율 조회 실패 → 기본값 1,400 사용")
        return 1400.0


def load_universe() -> list[str]:
    df = pd.read_parquet(UNIVERSE_PATH)
    return df["ticker"].tolist()


def filter_by_market_cap(tickers: list[str], krw_rate: float) -> list[tuple[str, float]]:
    min_usd = MARKET_CAP_MIN_KRW / krw_rate
    max_usd = MARKET_CAP_MAX_KRW / krw_rate
    logger.info(f"시총 필터 범위: ${min_usd/1e6:.0f}M ~ ${max_usd/1e9:.1f}B (USD)")

    passed = []
    for ticker in tqdm(tickers, desc="1단계 시총 필터"):
        try:
            mc = yf.Ticker(ticker).fast_info["market_cap"]
            if mc and min_usd <= mc <= max_usd:
                passed.append((ticker, float(mc)))
        except Exception:
            pass
        time.sleep(DELAY)
    return passed


def apply_quality_filters(info: dict) -> bool:
    filters = [
        ("debtToEquity",    info.get("debtToEquity"),    lambda v: v <= 100),
        ("currentRatio",    info.get("currentRatio"),     lambda v: v >= 1.0),
        ("grossMargins",    info.get("grossMargins"),     lambda v: v >= 0.1),
        ("operatingMargins",info.get("operatingMargins"), lambda v: v >= 0.1),
        ("profitMargins",   info.get("profitMargins"),    lambda v: v >= 0.05),
        ("revenueGrowth",   info.get("revenueGrowth"),   lambda v: v >= 0.2),
    ]
    for name, val, cond in filters:
        if val is None:
            return False
        if not cond(val):
            return False
    return True


def fetch_and_filter(ticker: str, mc: float) -> dict | None:
    try:
        info = yf.Ticker(ticker).info

        if not apply_quality_filters(info):
            return None

        officers = info.get("companyOfficers", [])
        ceo = "N/A"
        for o in officers:
            title = o.get("title", "")
            if "CEO" in title or "Chief Executive" in title:
                ceo = o.get("name", "N/A")
                break
        if ceo == "N/A" and officers:
            ceo = officers[0].get("name", "N/A")

        cash = info.get("totalCash")
        debt = info.get("totalDebt")

        return {
            "회사명":          info.get("longName") or info.get("shortName", "N/A"),
            "티커":            ticker,
            "CEO":             ceo,
            "_mc":             mc,
            "시가총액":        fmt_usd(mc),
            "현금비율":        fmt_ratio(cash, mc),
            "현금표시":        fmt_usd(cash),
            "부채비율":        fmt_ratio(debt, mc),
            "부채표시":        fmt_usd(debt),
            "매출총이익률":    fmt_pct(info.get("grossMargins")),
            "영업이익률":      fmt_pct(info.get("operatingMargins")),
            "순이익률":        fmt_pct(info.get("profitMargins")),
            "매출성장률":      fmt_pct(info.get("revenueGrowth")),
            "PEG":             fmt_peg(info.get("trailingPegRatio")),
            "인비저블 썸띵(텐버거)": "",
            "인비저블 썸띵(리스크)": "",
            "제미나이 점수":   "",
            "클로드 점수":     "",
            "그록 점수":       "",
            "점수 합":         "",
        }
    except Exception as e:
        logger.debug(f"{ticker} 수집 실패: {e}")
        return None


def build_dataframe(passed: list[tuple[str, float]]) -> pd.DataFrame:
    rows = []
    for ticker, mc in tqdm(passed, desc="2단계 품질 필터 + 상세 수집"):
        result = fetch_and_filter(ticker, mc)
        if result:
            rows.append(result)
        time.sleep(DELAY)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("_mc", ascending=False).reset_index(drop=True)
    df = df.drop(columns=["_mc"])
    return df


def save_all(df: pd.DataFrame, krw_rate: float):
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    base = OUTPUT_DIR / f"us_quality_{date_str}"

    # Excel
    with pd.ExcelWriter(base.with_suffix(".xlsx"), engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="스크리닝결과")
        ws = writer.sheets["스크리닝결과"]
        for cell in ws[1]:
            cell.font = cell.font.copy(bold=True)
        for col in ws.columns:
            width = max(len(str(c.value or "")) for c in col) + 4
            ws.column_dimensions[col[0].column_letter].width = min(width, 45)
    logger.info(f"Excel → {base.with_suffix('.xlsx')}")

    # JSON
    df.to_json(base.with_suffix(".json"), orient="records", force_ascii=False, indent=2)
    logger.info(f"JSON  → {base.with_suffix('.json')}")

    # TXT
    with open(base.with_suffix(".txt"), "w", encoding="utf-8") as f:
        f.write("US 우량주 스크리닝 결과\n")
        f.write(f"수집일시 : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"환율     : 1 USD = {krw_rate:,.0f} KRW\n")
        f.write(f"필터     : 시총 {MARKET_CAP_MIN_KRW/1e8:.0f}억원 ~ {MARKET_CAP_MAX_KRW/1e12:.0f}조원\n")
        f.write(f"종목수   : {len(df):,}개\n")
        f.write("=" * 120 + "\n")
        f.write(df.to_string(index=False))
    logger.info(f"TXT   → {base.with_suffix('.txt')}")


def run(sample: int | None = None):
    logger.info("=== US 우량주 스크리너 시작 ===")

    tickers = load_universe()
    if sample:
        random.seed(42)
        tickers = random.sample(tickers, min(sample, len(tickers)))
        logger.info(f"샘플 {sample}개 랜덤 선택")
    logger.info(f"처리 대상: {len(tickers):,}개")

    krw_rate = get_krw_rate()

    passed = filter_by_market_cap(tickers, krw_rate)
    logger.info(f"시총 필터 통과: {len(passed):,}개")

    if not passed:
        logger.warning("시총 조건에 맞는 종목 없음.")
        return

    df = build_dataframe(passed)

    if df.empty:
        logger.warning("품질 필터 통과 종목 없음.")
        return

    save_all(df, krw_rate)
    logger.success(f"완료. {len(df):,}개 종목 저장 → {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    run(sample=args.sample)
