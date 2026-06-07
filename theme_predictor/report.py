import json
import sys
from pathlib import Path
from datetime import datetime
from loguru import logger

sys.path.insert(0, str(Path(__file__).parents[1]))

from theme_predictor.collectors.form13f import run as run_13f
from theme_predictor.collectors.political_flow import run as run_political
from theme_predictor.collectors.sector_buzz import run as run_buzz
from theme_predictor.collectors.etf_flow import run as run_etf
from theme_predictor.collectors.smart_money import run as run_smart

SIGNALS_DIR = Path(__file__).parent / "signals"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_signals() -> dict:
    """저장된 signals/*.json 로드. 없으면 실시간 수집."""
    def _load(fname: str, runner):
        p = SIGNALS_DIR / fname
        if p.exists():
            logger.info(f"캐시 로드: {fname}")
            return json.loads(p.read_text())
        logger.info(f"수집 시작: {fname}")
        return runner()

    return {
        "etf":      _load("etf_inflow.json",     run_etf),
        "buzz":     _load("buzz_trend.json",      run_buzz),
        "f13":      _load("13f_sector.json",      run_13f),
        "political":_load("political_flow.json",  run_political),
        "smart":    _load("smart_money.json",     run_smart),
    }


def build_sector_table(signals: dict) -> list[dict]:
    """신호 5개를 섹터별로 통합."""
    all_sectors: set[str] = set()

    etf_flows = signals["etf"].get("sector_etf_flows", {})
    buzz      = signals["buzz"].get("sector_mentions", {})
    f13_flow  = signals["f13"].get("sector_flow_usd", {})
    pol_buys  = signals["political"].get("sector_buy_count", {})
    smart     = signals["smart"].get("smart_money_signals", {})

    for d in [etf_flows, buzz, f13_flow, pol_buys, smart]:
        all_sectors.update(d.keys())

    rows = []
    for sector in all_sectors:
        etf_surge = etf_flows.get(sector, {}).get("avg_vol_surge", 1.0) if isinstance(etf_flows.get(sector), dict) else 1.0
        buzz_total = buzz.get(sector, {}).get("total", 0) if isinstance(buzz.get(sector), dict) else 0
        f13_usd    = f13_flow.get(sector, 0)
        pol_count  = pol_buys.get(sector, 0)
        smart_total= smart.get(sector, {}).get("total", 0) if isinstance(smart.get(sector), dict) else 0

        signals_on = []
        if etf_surge >= 1.5:
            signals_on.append(f"ETF 거래량 {etf_surge:.1f}x")
        if buzz_total >= 5:
            signals_on.append(f"소셜/뉴스 언급 {buzz_total}건")
        if f13_usd > 0:
            signals_on.append(f"13F ${f13_usd/1e9:.1f}B")
        if pol_count >= 2:
            signals_on.append(f"의원 매수 {pol_count}건")
        if smart_total >= 3:
            signals_on.append(f"옵션/내부자 {smart_total}건")

        rows.append({
            "섹터": sector,
            "ETF거래량배수": etf_surge,
            "소셜뉴스언급": buzz_total,
            "13F자금(B$)": round(f13_usd / 1e9, 2) if f13_usd else 0,
            "의원매수건": pol_count,
            "스마트머니": smart_total,
            "신호수": len(signals_on),
            "신호요약": " / ".join(signals_on) if signals_on else "신호없음",
        })

    rows.sort(key=lambda x: (x["신호수"], x["소셜뉴스언급"]), reverse=True)
    return rows


def print_report(rows: list[dict]):
    print("\n" + "=" * 80)
    print(f"  다음 테마 후보 — {datetime.now().strftime('%Y-%m-%d %H:%M')} 기준")
    print("=" * 80)

    for i, row in enumerate(rows[:10], 1):
        print(f"\n{i}위: {row['섹터']}")
        print(f"     {row['신호요약']}")

    print("\n" + "=" * 80)


def save_excel(rows: list[dict]):
    try:
        import pandas as pd
        from openpyxl import load_workbook

        df = pd.DataFrame(rows)
        date_str = datetime.now().strftime("%Y%m%d_%H%M")
        path = OUTPUT_DIR / f"theme_predictor_{date_str}.xlsx"

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="테마예측")
            ws = writer.sheets["테마예측"]
            for cell in ws[1]:
                cell.font = cell.font.copy(bold=True)
            for col in ws.columns:
                width = max(len(str(c.value or "")) for c in col) + 4
                ws.column_dimensions[col[0].column_letter].width = min(width, 40)

        logger.success(f"엑셀 저장 → {path}")
    except ImportError:
        logger.warning("pandas/openpyxl 없음 → 엑셀 저장 스킵")


def run(refresh: bool = False):
    if refresh:
        logger.info("전체 신호 재수집 (--refresh)")
        signals = {
            "etf":       run_etf(),
            "buzz":      run_buzz(),
            "f13":       run_13f(),
            "political": run_political(),
            "smart":     run_smart(),
        }
    else:
        signals = load_signals()

    rows = build_sector_table(signals)
    print_report(rows)
    save_excel(rows)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="다음 테마 후보 섹터 예측")
    parser.add_argument("--refresh", action="store_true", help="캐시 무시하고 전체 재수집")
    args = parser.parse_args()
    run(refresh=args.refresh)
