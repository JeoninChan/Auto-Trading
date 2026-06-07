import sys
import argparse
import time
from pathlib import Path

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from loguru import logger
from tqdm import tqdm

DELAY = 0.3

OLD_COLS = [
    "기술점수", "기술신호",
    "매매점수", "매매신호",
    "장세", "추세점수", "추세신호",
    "수급점수", "수급신호",
    "모멘텀점수", "모멘텀신호",
    "추천매수가", "추천목표가", "추천손절가",
]


# ---------------------------------------------------------------------------
# 장세 판단
# ---------------------------------------------------------------------------

def get_regime(high: pd.Series, low: pd.Series, close: pd.Series) -> str:
    adx_df = ta.adx(high, low, close, length=14)
    if adx_df is None or adx_df.empty:
        return "횡보장"
    cols = [c for c in adx_df.columns if c.startswith("ADX_")]
    if not cols:
        return "횡보장"
    adx = adx_df[cols[0]].iloc[-1]
    return "추세장" if (not np.isnan(adx) and adx > 25) else "횡보장"


# ---------------------------------------------------------------------------
# 추세장 — 돌파매매
# ---------------------------------------------------------------------------

def trend_breakout(close: pd.Series):
    sma5  = close.rolling(5).mean().iloc[-1]
    sma20 = close.rolling(20).mean().iloc[-1]
    if np.isnan(sma5) or np.isnan(sma20):
        return 0, "없음"
    if sma5 > sma20:
        return 30, "정배열↑"
    return 0, "역배열↓"


def supply_score(close: pd.Series, open_: pd.Series, volume: pd.Series):
    avg = volume.iloc[-6:-1].mean()
    if avg == 0 or np.isnan(avg):
        return 0, "없음"
    ratio = volume.iloc[-1] / avg
    up = close.iloc[-1] >= open_.iloc[-1]
    if not up:
        return 0, "없음"
    if ratio >= 2.0:
        return 40, "폭발매수"
    if ratio >= 1.5:
        return 25, "강세↑"
    return 0, "없음"


def momentum_breakout(close: pd.Series):
    rsi_s = ta.rsi(close, length=14)
    if rsi_s is None or rsi_s.empty:
        return 0, "없음"
    rsi = rsi_s.iloc[-1]
    if np.isnan(rsi):
        return 0, "없음"
    if rsi >= 70:
        return 30, "과매수↑↑"
    if rsi >= 60:
        return 20, "강세"
    if rsi >= 50:
        return 10, "중립↑"
    return 0, "없음"


# ---------------------------------------------------------------------------
# 횡보장 — 눌림목
# ---------------------------------------------------------------------------

def trend_pullback(close: pd.Series):
    bb_df = ta.bbands(close, length=20)
    lower = None
    if bb_df is not None and not bb_df.empty:
        lc = [c for c in bb_df.columns if "BBL" in c]
        if lc:
            lower = bb_df[lc[0]].iloc[-1]

    sma20 = close.rolling(20).mean().iloc[-1]
    sma60 = close.rolling(60).mean().iloc[-1]
    c = close.iloc[-1]

    if lower is not None and not np.isnan(lower) and c <= lower:
        return 30, "강지지↑↑"
    if not np.isnan(sma20) and c <= sma20:
        return 15, "지지↑"
    if not np.isnan(sma60) and abs(c - sma60) / sma60 <= 0.03:
        return 10, "근접"
    return 0, "없음"


def momentum_pullback(close: pd.Series):
    rsi_s = ta.rsi(close, length=14)
    if rsi_s is None or rsi_s.empty:
        return 0, "없음"
    rsi = rsi_s.iloc[-1]
    if np.isnan(rsi):
        return 0, "없음"
    if rsi <= 30:
        return 30, "과매도↑↑"
    if rsi <= 40:
        return 20, "매도탈출↑"
    if rsi <= 50:
        return 10, "중립↓"
    return 0, "없음"


# ---------------------------------------------------------------------------
# ATR 타점
# ---------------------------------------------------------------------------

def compute_prices(close: pd.Series, high: pd.Series, low: pd.Series):
    atr_s = ta.atr(high, low, close, length=14)
    if atr_s is None or atr_s.empty:
        return None, None, None
    atr = atr_s.iloc[-1]
    if np.isnan(atr) or atr <= 0:
        return None, None, None

    bb_df = ta.bbands(close, length=20)
    bb_upper = None
    if bb_df is not None and not bb_df.empty:
        uc = [c for c in bb_df.columns if "BBU" in c]
        if uc:
            bb_upper = bb_df[uc[0]].iloc[-1]

    entry = round(close.iloc[-1] - 0.5 * atr, 4)
    stop  = round(entry - 1.5 * atr, 4)
    t_atr = entry + 2.0 * atr
    target = round(min(bb_upper, t_atr) if (bb_upper and not np.isnan(bb_upper)) else t_atr, 4)

    return entry, target, stop


# ---------------------------------------------------------------------------
# 종목별 분석
# ---------------------------------------------------------------------------

def analyze_ticker(ticker: str):
    try:
        df = yf.Ticker(ticker).history(period="120d")
        if df.empty or len(df) < 61:
            return None
        df = df.sort_index()

        close  = df["Close"]
        open_  = df["Open"]
        high   = df["High"]
        low    = df["Low"]
        volume = df["Volume"]

        regime = get_regime(high, low, close)

        if regime == "추세장":
            ts, tl = trend_breakout(close)
            ms, ml = momentum_breakout(close)
        else:
            ts, tl = trend_pullback(close)
            ms, ml = momentum_pullback(close)

        ss, sl = supply_score(close, open_, volume)
        entry, target, stop = compute_prices(close, high, low)

        return regime, ts, tl, ss, sl, ms, ml, entry, target, stop

    except Exception as e:
        logger.debug(f"{ticker} 실패: {e}")
        return None


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def run(excel_path: Path, filter_val: str | None = None):
    logger.info(f"로드: {excel_path}")
    df = pd.read_excel(excel_path)

    if "티커" not in df.columns:
        logger.error("'티커' 칼럼을 찾을 수 없습니다.")
        sys.exit(1)

    drop_targets = [c for c in OLD_COLS if c in df.columns]
    if drop_targets:
        df = df.drop(columns=drop_targets)
        logger.info(f"구버전 컬럼 제거: {drop_targets}")

    mask = df["티커"].astype(str).str.strip().ne("")
    if "회사명" in df.columns:
        mask &= df["회사명"].astype(str).str.strip().ne("[평균]")

    if filter_val:
        fv = filter_val.strip()
        t_hit = df["티커"].astype(str).str.upper().eq(fv.upper())
        n_hit = (
            df["회사명"].astype(str).str.lower().str.contains(fv.lower(), na=False)
            if "회사명" in df.columns else pd.Series(False, index=df.index)
        )
        mask &= t_hit | n_hit

    idxs = df.index[mask].tolist()
    logger.info(f"처리 대상: {len(idxs)}개 종목")

    results: dict[int, tuple] = {}
    for i in tqdm(idxs, desc="매매 평가"):
        res = analyze_ticker(str(df.at[i, "티커"]))
        results[i] = res if res else (None,) * 10
        time.sleep(DELAY)

    df["장세"]      = pd.Series({i: results[i][0] for i in idxs})
    df["추세점수"]  = pd.Series({i: results[i][1] for i in idxs})
    df["추세신호"]  = pd.Series({i: results[i][2] for i in idxs})
    df["수급점수"]  = pd.Series({i: results[i][3] for i in idxs})
    df["수급신호"]  = pd.Series({i: results[i][4] for i in idxs})
    df["모멘텀점수"] = pd.Series({i: results[i][5] for i in idxs})
    df["모멘텀신호"] = pd.Series({i: results[i][6] for i in idxs})
    df["추천매수가"] = pd.Series({i: results[i][7] for i in idxs})
    df["추천목표가"] = pd.Series({i: results[i][8] for i in idxs})
    df["추천손절가"] = pd.Series({i: results[i][9] for i in idxs})

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="스크리닝결과")
        ws = writer.sheets["스크리닝결과"]
        for cell in ws[1]:
            cell.font = cell.font.copy(bold=True)
        for col in ws.columns:
            width = max(len(str(c.value or "")) for c in col) + 4
            ws.column_dimensions[col[0].column_letter].width = min(width, 45)

    logger.success(f"저장 완료 → {excel_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADX 장세 판단 + 3요소 독립 평가")
    parser.add_argument("excel", help="scored 엑셀 파일 경로")
    parser.add_argument("--ticker", "-t", default=None, help="특정 티커")
    parser.add_argument("--name",   "-n", default=None, help="특정 회사명 (부분 일치)")
    args = parser.parse_args()

    run(Path(args.excel), filter_val=args.ticker or args.name)
