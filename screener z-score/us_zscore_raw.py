"""
US Z-Score 스크리너 — Raw 데이터 전용 (점수 없음)

유니버스 전체 종목을 수집, 유동비율 < 1.3 제외 후
순수 지표값만 저장. 점수 칼럼 없음.

유일한 하드 필터: 유동비율 < 1.3 (좀비기업 컷오프)

출력:
  us_zscore_raw_YYYYMMDD_HHMM.{xlsx,json,txt}

사용법:
    python3 "screener z-score/us_zscore_raw.py"
    python3 "screener z-score/us_zscore_raw.py" --sample 100
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

CURRENT_RATIO_MIN = 1.3
DELAY             = 0.3
OUTPUT_DIR        = Path(__file__).parent / "output"
UNIVERSE_PATH     = Path(__file__).parents[1] / "datasets" / "us" / "universe.parquet"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLS_ORDER = [
    "회사명", "티커", "CEO",
    "시가총액",
    "현금비율", "현금표시",
    "유동비율",
    "부채비율", "부채표시",
    "매출총이익률",
    "영업이익률",
    "순이익률",
    "매출성장률",
    "PEG",
    "인비저블 썸띵(텐버거)", "인비저블 썸띵(리스크)",
    "제미나이 점수", "클로드 점수", "그록 점수", "점수 합",
]


# ---------------------------------------------------------------------------
# 포맷 헬퍼
# ---------------------------------------------------------------------------

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


def fmt_float(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "N/A"


# ---------------------------------------------------------------------------
# 환율 / 유니버스
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 종목별 수집
# ---------------------------------------------------------------------------

def fetch_raw(ticker: str) -> dict | None:
    try:
        info = yf.Ticker(ticker).info

        current_ratio = info.get("currentRatio")
        if current_ratio is None or float(current_ratio) < CURRENT_RATIO_MIN:
            return None

        mc   = info.get("marketCap")
        cash = info.get("totalCash")
        debt = info.get("totalDebt")

        officers = info.get("companyOfficers", [])
        ceo = "N/A"
        for o in officers:
            title = o.get("title", "")
            if "CEO" in title or "Chief Executive" in title:
                ceo = o.get("name", "N/A")
                break
        if ceo == "N/A" and officers:
            ceo = officers[0].get("name", "N/A")

        return {
            "회사명":                  info.get("longName") or info.get("shortName", "N/A"),
            "티커":                    ticker,
            "CEO":                     ceo,
            "_mc_raw":                 float(mc) if mc else None,
            "시가총액":                fmt_usd(mc),
            "현금비율":                fmt_ratio(cash, mc),
            "현금표시":                fmt_usd(cash),
            "유동비율":                fmt_float(current_ratio),
            "부채비율":                fmt_ratio(debt, mc),
            "부채표시":                fmt_usd(debt),
            "매출총이익률":            fmt_pct(info.get("grossMargins")),
            "영업이익률":              fmt_pct(info.get("operatingMargins")),
            "순이익률":                fmt_pct(info.get("profitMargins")),
            "매출성장률":              fmt_pct(info.get("revenueGrowth")),
            "PEG":                     fmt_peg(info.get("trailingPegRatio")),
            "인비저블 썸띵(텐버거)":   "",
            "인비저블 썸띵(리스크)":   "",
            "제미나이 점수":           "",
            "클로드 점수":             "",
            "그록 점수":               "",
            "점수 합":                 "",
        }
    except Exception as e:
        logger.debug(f"{ticker} 수집 실패: {e}")
        return None


# ---------------------------------------------------------------------------
# DataFrame 빌드
# ---------------------------------------------------------------------------

def build_dataframe(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for ticker in tqdm(tickers, desc="Raw 수집 (유동비율 ≥ 1.3 필터)"):
        result = fetch_raw(ticker)
        if result:
            rows.append(result)
        time.sleep(DELAY)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("_mc_raw", ascending=False, na_position="last").reset_index(drop=True)
    df = df.drop(columns=["_mc_raw"])
    return df


# ---------------------------------------------------------------------------
# 저장
# ---------------------------------------------------------------------------

def _to_excel(df: pd.DataFrame, path: Path):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="스크리닝결과")
        ws = writer.sheets["스크리닝결과"]
        for cell in ws[1]:
            cell.font = cell.font.copy(bold=True)
        for col in ws.columns:
            width = max(len(str(c.value or "")) for c in col) + 4
            ws.column_dimensions[col[0].column_letter].width = min(width, 45)
    logger.info(f"Excel → {path}")


def save_all(df: pd.DataFrame, krw_rate: float):
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    cols = [c for c in COLS_ORDER if c in df.columns]
    df_out = df[cols]

    base = OUTPUT_DIR / f"us_zscore_raw_{date_str}"

    _to_excel(df_out, base.with_suffix(".xlsx"))

    df_out.to_json(base.with_suffix(".json"), orient="records", force_ascii=False, indent=2)
    logger.info(f"JSON  → {base.with_suffix('.json')}")

    with open(base.with_suffix(".txt"), "w", encoding="utf-8") as f:
        f.write("US Z-Score 스크리닝 결과 [Raw 지표값]\n")
        f.write(f"수집일시 : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"환율     : 1 USD = {krw_rate:,.0f} KRW\n")
        f.write(f"유동비율 : ≥ {CURRENT_RATIO_MIN} 통과 기준\n")
        f.write(f"종목수   : {len(df_out):,}개\n")
        f.write("=" * 130 + "\n")
        f.write(df_out.to_string(index=False))
    logger.info(f"TXT   → {base.with_suffix('.txt')}")


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

def run(sample: int | None = None):
    logger.info("=== US Z-Score Raw 스크리너 시작 ===")

    tickers = load_universe()
    if sample:
        random.seed(42)
        tickers = random.sample(tickers, min(sample, len(tickers)))
        logger.info(f"샘플 {sample}개 랜덤 선택")
    logger.info(f"처리 대상: {len(tickers):,}개")

    krw_rate = get_krw_rate()
    df = build_dataframe(tickers)

    if df.empty:
        logger.warning("유동비율 조건 통과 종목 없음.")
        return

    save_all(df, krw_rate)
    logger.success(f"완료. {len(df):,}개 종목 저장 → {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    run(sample=args.sample)
