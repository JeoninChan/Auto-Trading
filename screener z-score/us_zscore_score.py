"""
US Z-Score 스크리너 — 점수 포함 버전

유니버스 전체 종목을 수집, 유동비율 < 1.3 제외 후
각 지표를 연속 점수(0~100)로 환산.

종합점수: 가중합 Σ(점수×가중치)
평균점수: 유효 점수들의 단순 산술평균 (None 제외)
  ※ 평균이 높아도 종합이 높은건 아님 (가중치 차이 때문)

가중치:
  시가총액   20%  — Plateau(5000억~28조 = 100점) + Gaussian Falloff
  PEG        30%  — 선형 매핑 (≤0.5→100, 1.5→50, 2.5+→0)
  매출성장률 20%  — CDF σ=15, μ=20%
  순이익률   15%  — CDF σ=10, μ=5%
  부채비율   10%  — 역방향 CDF σ=40, μ=100%
  매출총이익률 5% — CDF σ=15, μ=10%

유일한 하드 필터: 유동비율 < 1.3

출력:
  us_zscore_scored_YYYYMMDD_HHMM.{xlsx,json,txt}
  마지막 행: [평균] — 모든 점수 칼럼의 전 종목 평균

사용법:
    python3 "screener z-score/us_zscore_score.py"
    python3 "screener z-score/us_zscore_score.py" --sample 100
"""
import sys
import math
import argparse
import time
import random
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parents[1]))

import scipy.stats as stats
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
    "시가총액", "시총점수(20%)",
    "현금비율", "현금표시",
    "유동비율",
    "부채비율", "부채점수(10%)", "부채표시",
    "매출총이익률", "매출총이익점수(5%)",
    "영업이익률",
    "순이익률", "순이익점수(15%)",
    "매출성장률", "매출성장점수(20%)",
    "PEG", "PEG점수(30%)",
    "종합점수", "평균점수",
    "인비저블 썸띵(텐버거)", "인비저블 썸띵(리스크)",
    "제미나이 점수", "클로드 점수", "그록 점수", "점수 합",
]

# 점수 칼럼명 → 내부 숫자 키 매핑 (평균행 계산용)
SCORE_COL_TO_KEY = {
    "시총점수(20%)":       "_s_mc",
    "부채점수(10%)":       "_s_de",
    "매출총이익점수(5%)":  "_s_gm",
    "순이익점수(15%)":     "_s_nm",
    "매출성장점수(20%)":   "_s_rg",
    "PEG점수(30%)":        "_s_peg",
    "종합점수":            "_total",
    "평균점수":            "_avg",
}


# ---------------------------------------------------------------------------
# 스코어링 함수
# ---------------------------------------------------------------------------

def score_market_cap(mc_usd: float, krw_rate: float) -> float:
    if not mc_usd or mc_usd <= 0:
        return 0.0
    mc_krw = mc_usd * krw_rate
    log_cap = math.log10(mc_krw)
    log_min = math.log10(5e11)
    log_max = math.log10(28e12)
    sigma = 0.15
    if log_min <= log_cap <= log_max:
        return 100.0
    elif log_cap < log_min:
        return 100.0 * math.exp(-((log_cap - log_min) ** 2) / (2 * sigma ** 2))
    else:
        return 100.0 * math.exp(-((log_cap - log_max) ** 2) / (2 * sigma ** 2))


def score_peg(peg: float) -> float:
    if peg <= 0:
        return 0.0
    if peg <= 0.5:
        return 100.0
    return max(0.0, 100.0 - 50.0 * (peg - 0.5))


def score_rev_growth(growth: float) -> float:
    return float(stats.norm.cdf((growth * 100 - 20) / 15) * 100)


def score_net_margin(margin: float) -> float:
    return float(stats.norm.cdf((margin * 100 - 5) / 10) * 100)


def score_de_ratio(de: float) -> float:
    return float((1 - stats.norm.cdf((de - 100) / 40)) * 100)


def score_gross_margin(gm: float) -> float:
    return float(stats.norm.cdf((gm * 100 - 10) / 15) * 100)


def compute_total(s: dict) -> float:
    """가중합 종합점수. None → 0 처리."""
    w = {
        "_s_mc":  0.20,
        "_s_peg": 0.30,
        "_s_rg":  0.20,
        "_s_nm":  0.15,
        "_s_de":  0.10,
        "_s_gm":  0.05,
    }
    return round(sum((s.get(k) or 0.0) * v for k, v in w.items()), 2)


def compute_avg(s: dict) -> float | None:
    """단순 산술평균. 유효 점수만 포함, 전부 None이면 None."""
    vals = [s[k] for k in ("_s_mc", "_s_peg", "_s_rg", "_s_nm", "_s_de", "_s_gm") if s.get(k) is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 2)


# ---------------------------------------------------------------------------
# 포맷 헬퍼
# ---------------------------------------------------------------------------

def _safe_float(v) -> float | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def fmt_usd(amount) -> str:
    v = _safe_float(amount)
    if v is None:
        return "N/A"
    eok = int(v // 100_000_000)
    man = int((v % 100_000_000) // 10_000)
    if eok > 0:
        return f"{eok}억 {man:,}만달러" if man > 0 else f"{eok}억달러"
    return f"{man:,}만달러"


def fmt_pct(value) -> str:
    v = _safe_float(value)
    return "N/A" if v is None else f"{v * 100:.1f}%"


def fmt_peg(value) -> str:
    v = _safe_float(value)
    return "N/A" if v is None else f"{v:.2f}"


def fmt_ratio(numerator, denominator) -> str:
    n, d = _safe_float(numerator), _safe_float(denominator)
    if n is None or d is None or d == 0:
        return "N/A"
    return f"{n / d * 100:.1f}%"


def fmt_score(value) -> str:
    v = _safe_float(value)
    return "N/A" if v is None else f"{v:.1f}"


def fmt_float(value) -> str:
    v = _safe_float(value)
    return "N/A" if v is None else f"{v:.2f}"


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
# 종목별 수집 + 점수 계산
# ---------------------------------------------------------------------------

def fetch_and_score(ticker: str, krw_rate: float) -> dict | None:
    try:
        info = yf.Ticker(ticker).info

        cr = _safe_float(info.get("currentRatio"))
        if cr is None or cr < CURRENT_RATIO_MIN:
            return None

        mc      = _safe_float(info.get("marketCap"))
        cash    = _safe_float(info.get("totalCash"))
        debt    = _safe_float(info.get("totalDebt"))
        de      = _safe_float(info.get("debtToEquity"))
        gm      = _safe_float(info.get("grossMargins"))
        om      = _safe_float(info.get("operatingMargins"))
        nm      = _safe_float(info.get("profitMargins"))
        rg      = _safe_float(info.get("revenueGrowth"))
        peg_raw = _safe_float(info.get("trailingPegRatio"))

        s_mc  = score_market_cap(mc, krw_rate) if mc else None
        s_peg = score_peg(peg_raw) if peg_raw is not None else None
        s_rg  = score_rev_growth(rg) if rg is not None else None
        s_nm  = score_net_margin(nm) if nm is not None else None
        s_de  = score_de_ratio(de) if de is not None else None
        s_gm  = score_gross_margin(gm) if gm is not None else None

        raw_scores = {
            "_s_mc":  s_mc,
            "_s_peg": s_peg,
            "_s_rg":  s_rg,
            "_s_nm":  s_nm,
            "_s_de":  s_de,
            "_s_gm":  s_gm,
        }
        total = compute_total(raw_scores)
        avg   = compute_avg(raw_scores)

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
            "_mc_raw":                 mc,
            "시가총액":                fmt_usd(mc),
            "시총점수(20%)":           fmt_score(s_mc),
            "현금비율":                fmt_ratio(cash, mc),
            "현금표시":                fmt_usd(cash),
            "유동비율":                fmt_float(cr),
            "부채비율":                fmt_ratio(debt, mc),
            "부채점수(10%)":           fmt_score(s_de),
            "부채표시":                fmt_usd(debt),
            "매출총이익률":            fmt_pct(gm),
            "매출총이익점수(5%)":      fmt_score(s_gm),
            "영업이익률":              fmt_pct(om),
            "순이익률":                fmt_pct(nm),
            "순이익점수(15%)":         fmt_score(s_nm),
            "매출성장률":              fmt_pct(rg),
            "매출성장점수(20%)":       fmt_score(s_rg),
            "PEG":                     fmt_peg(peg_raw),
            "PEG점수(30%)":            fmt_score(s_peg),
            "종합점수":                fmt_score(total),
            "평균점수":                fmt_score(avg),
            "인비저블 썸띵(텐버거)":   "",
            "인비저블 썸띵(리스크)":   "",
            "제미나이 점수":           "",
            "클로드 점수":             "",
            "그록 점수":               "",
            "점수 합":                 "",
            # 내부 숫자값 (정렬/평균행 계산용)
            "_s_mc":  s_mc,
            "_s_peg": s_peg,
            "_s_rg":  s_rg,
            "_s_nm":  s_nm,
            "_s_de":  s_de,
            "_s_gm":  s_gm,
            "_total": total,
            "_avg":   avg,
        }
    except Exception as e:
        logger.debug(f"{ticker} 수집 실패: {e}")
        return None


# ---------------------------------------------------------------------------
# DataFrame 빌드
# ---------------------------------------------------------------------------

INTERNAL_COLS = ["_mc_raw", "_s_mc", "_s_peg", "_s_rg", "_s_nm", "_s_de", "_s_gm", "_total", "_avg"]


def build_dataframe(tickers: list[str], krw_rate: float) -> pd.DataFrame:
    rows = []
    for ticker in tqdm(tickers, desc="수집 + Z-Score 점수화"):
        result = fetch_and_score(ticker, krw_rate)
        if result:
            rows.append(result)
        time.sleep(DELAY)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("_total", ascending=False, na_position="last").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 평균 행 생성
# ---------------------------------------------------------------------------

def make_avg_row(df: pd.DataFrame) -> pd.Series:
    """점수 칼럼 전 종목 평균을 담은 [평균] 행."""
    row = {c: "" for c in df.columns}
    row["회사명"] = "[평균]"
    row["티커"]   = ""
    row["CEO"]    = ""

    for display_col, int_col in SCORE_COL_TO_KEY.items():
        if int_col in df.columns:
            vals = pd.to_numeric(df[int_col], errors="coerce").dropna()
            row[display_col] = f"{vals.mean():.1f}" if len(vals) else "N/A"

    return pd.Series(row)


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
    n_total  = len(df)

    # 내부 칼럼 제거 → 표시용 칼럼만
    clean = df.drop(columns=[c for c in INTERNAL_COLS if c in df.columns])
    cols  = [c for c in COLS_ORDER if c in clean.columns]
    df_display = clean[cols].copy()

    # 평균 행을 내부 숫자 칼럼이 있는 df 기준으로 계산 후 display 칼럼에 맞게 변환
    avg_row = make_avg_row(df)
    avg_series = pd.Series({c: avg_row.get(c, "") for c in df_display.columns})
    df_out = pd.concat([df_display, avg_series.to_frame().T], ignore_index=True)

    base = OUTPUT_DIR / f"us_zscore_scored_{date_str}"

    _to_excel(df_out, base.with_suffix(".xlsx"))

    df_out.to_json(base.with_suffix(".json"), orient="records", force_ascii=False, indent=2)
    logger.info(f"JSON  → {base.with_suffix('.json')}")

    with open(base.with_suffix(".txt"), "w", encoding="utf-8") as f:
        f.write("US Z-Score 스크리닝 결과 [점수 포함]\n")
        f.write(f"수집일시  : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"환율      : 1 USD = {krw_rate:,.0f} KRW\n")
        f.write(f"유동비율  : ≥ {CURRENT_RATIO_MIN} 통과 기준\n")
        f.write(f"종목수    : {n_total:,}개\n")
        f.write(f"종합점수  : 가중합 (시총20%+PEG30%+매출성장20%+순이익15%+부채10%+매출총이익5%)\n")
        f.write(f"평균점수  : 6개 점수 단순평균 (None 제외)\n")
        f.write("=" * 150 + "\n")
        f.write(df_out.to_string(index=False))
    logger.info(f"TXT   → {base.with_suffix('.txt')}")


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------

def run(sample: int | None = None):
    logger.info("=== US Z-Score 점수 스크리너 시작 ===")

    tickers = load_universe()
    if sample:
        random.seed(42)
        tickers = random.sample(tickers, min(sample, len(tickers)))
        logger.info(f"샘플 {sample}개 랜덤 선택")
    logger.info(f"처리 대상: {len(tickers):,}개")

    krw_rate = get_krw_rate()
    df = build_dataframe(tickers, krw_rate)

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
