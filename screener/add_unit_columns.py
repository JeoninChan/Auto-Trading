"""
기존 xlsx 파일에 시총 한국식 단위 칼럼 추가

사용법:
    python screener/add_unit_columns.py
    → 파일 경로 입력 (예: screener/output/us_smallmid_20260516_0200.xlsx)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd


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


def add_unit_columns(path: Path):
    df = pd.read_excel(path)

    if "시총(USD)" not in df.columns or "시총(KRW)" not in df.columns:
        print(f"오류: '시총(USD)' 또는 '시총(KRW)' 칼럼이 없습니다.")
        print(f"현재 칼럼: {list(df.columns)}")
        return

    df["시총표시(USD)"] = df["시총(USD)"].apply(fmt_usd)
    df["시총표시(KRW)"] = df["시총(KRW)"].apply(fmt_krw)

    # 칼럼 순서 재배치 — 표시 칼럼을 원본 바로 뒤에 삽입
    cols = list(df.columns)
    for display_col, after_col in [("시총표시(USD)", "시총(USD)"), ("시총표시(KRW)", "시총(KRW)")]:
        if display_col in cols:
            cols.remove(display_col)
        idx = cols.index(after_col) + 1
        cols.insert(idx, display_col)
    df = df[cols]

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="스크리닝결과")
        ws = writer.sheets["스크리닝결과"]
        for cell in ws[1]:
            cell.font = cell.font.copy(bold=True)
        for col in ws.columns:
            width = max(len(str(c.value or "")) for c in col) + 4
            ws.column_dimensions[col[0].column_letter].width = min(width, 45)

    print(f"완료: {path}")
    print(f"총 {len(df):,}개 종목, 칼럼: {list(df.columns)}")


if __name__ == "__main__":
    raw = input("파일 경로 입력: ").strip().strip('"')
    target = Path(raw)
    if not target.exists():
        print(f"파일 없음: {target}")
        sys.exit(1)
    add_unit_columns(target)
