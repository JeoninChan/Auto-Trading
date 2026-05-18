"""
US 소중형주 스크리너 — 시총 $100M ~ $2B

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

MARKET_CAP_MIN = 100_000_000     # $100M
MARKET_CAP_MAX = 2_000_000_000   # $2B
DELAY          = 0.3
OUTPUT_DIR     = Path(__file__).parent / "output"
UNIVERSE_PATH  = Path(__file__).parents[1] / "datasets" / "us" / "universe.parquet"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fmt_krw(amount) -> str:
    if not amount or pd.isna(amount):
        return "N/A"
    jo  = int(amount // 1_000_000_000_000)
    eok = int((amount % 1_000_000_000_000) // 100_000_000)
    if jo > 0:
        return f"{jo}조 {eok}억" if eok > 0 else f"{jo}조"
    return f"{eok}억"


def fmt_usd(amount) -> str:
    if not amount or pd.isna(amount):
        return "N/A"
    eok = int(amount // 100_000_000)
    man = int((amount % 100_000_000) // 10_000)
    if eok > 0:
        return f"{eok}억 {man:,}만달러" if man > 0 else f"{eok}억달러"
    return f"{man:,}만달러"


def get_krw_rate() -> float:
    try:
        rate = yf.Ticker("KRW=X").fast_info["lastPrice"]
        logger.info(f"USD/KRW 환율: {rate:,.0f}")
        return float(rate)
    except Exception:
        logger.warning("환율 조회 실패 → 기본값 1,380 사용")
        return 1380.0


def load_universe() -> list[str]:
    df = pd.read_parquet(UNIVERSE_PATH)
    return df["ticker"].tolist()


def filter_by_market_cap(tickers: list[str]) -> list[tuple[str, float]]:
    passed = []
    for ticker in tqdm(tickers, desc="1단계 시총 필터"):
        try:
            mc = yf.Ticker(ticker).fast_info["market_cap"]
            if mc and MARKET_CAP_MIN <= mc <= MARKET_CAP_MAX:
                passed.append((ticker, float(mc)))
        except Exception:
            pass
        time.sleep(DELAY)
    return passed


def fetch_detail(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info

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
            "회사이름": info.get("longName") or info.get("shortName", "N/A"),
            "CEO":     ceo,
            "현금":    info.get("totalCash"),
            "부채":    info.get("totalDebt"),
        }
    except Exception:
        return {"회사이름": "N/A", "CEO": "N/A", "현금": None, "부채": None}


def build_dataframe(passed: list[tuple[str, float]], krw_rate: float) -> pd.DataFrame:
    rows = []
    for ticker, mc in tqdm(passed, desc="2단계 상세 수집"):
        detail = fetch_detail(ticker)
        mc_krw = round(mc * krw_rate, 2)
        rows.append({
            "회사이름":      detail["회사이름"],
            "티커":         ticker,
            "CEO":          detail["CEO"],
            "시총(USD)":    round(mc, 2),
            "시총표시(USD)": fmt_usd(mc),
            "시총(KRW)":    mc_krw,
            "시총표시(KRW)": fmt_krw(mc_krw),
            "현금":         detail["현금"],
            "부채":         detail["부채"],
        })
        time.sleep(DELAY)

    df = pd.DataFrame(rows)
    return df.sort_values("시총(USD)", ascending=False).reset_index(drop=True)


def save_all(df: pd.DataFrame, krw_rate: float):
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    base = OUTPUT_DIR / f"us_smallmid_{date_str}"

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
        f.write(f"US 소중형주 스크리닝 결과\n")
        f.write(f"수집일시 : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"환율     : 1 USD = {krw_rate:,.0f} KRW\n")
        f.write(f"필터     : 시총 ${MARKET_CAP_MIN/1e6:.0f}M ~ ${MARKET_CAP_MAX/1e9:.0f}B\n")
        f.write(f"종목수   : {len(df):,}개\n")
        f.write("=" * 110 + "\n")
        f.write(df.to_string(index=False))
    logger.info(f"TXT   → {base.with_suffix('.txt')}")


def run(sample: int | None = None):
    logger.info("=== US 소중형주 스크리너 시작 ===")

    tickers = load_universe()
    if sample:
        random.seed(42)
        tickers = random.sample(tickers, min(sample, len(tickers)))
        logger.info(f"샘플 {sample}개 랜덤 선택")
    logger.info(f"처리 대상: {len(tickers):,}개")

    krw_rate = get_krw_rate()

    passed = filter_by_market_cap(tickers)
    logger.info(f"시총 필터 통과: {len(passed):,}개")

    if not passed:
        logger.warning("조건에 맞는 종목 없음.")
        return

    df = build_dataframe(passed, krw_rate)
    save_all(df, krw_rate)
    logger.success(f"완료. {len(df):,}개 종목 저장 → {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    run(sample=args.sample)
