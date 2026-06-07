import requests
import pandas as pd
from pathlib import Path
from loguru import logger

URL       = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
HEADERS   = {"User-Agent": "stock-research-bot hideinthecodes@gmail.com"}
SAVE_PATH = Path(__file__).parents[3] / "datasets" / "us" / "universe.parquet"


def fetch() -> pd.DataFrame:
    logger.info("NASDAQ Trader 종목 목록 다운로드...")
    r = requests.get(URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(r.text), sep="|")
    df = df[:-1]  # 마지막 행 파일 생성일 메타 제거
    logger.info(f"{len(df):,}개 수신")
    return df


def get_universe(save: bool = True) -> pd.DataFrame:
    df = fetch()

    # ETF·테스트 종목 제외, 실제 주식만
    df = df[(df["ETF"] == "N") & (df["Test Issue"] == "N")]

    # 점·하이픈·특수문자 포함 티커 제외
    df = df[~df["Symbol"].str.contains(r"[.\-+^$]", regex=True, na=True)]

    # Rights / Units / Warrants / Depositary / SPAC 제외
    _exclude = (
        "- Rights|- Units|- Warrant|Depositary Share|"
        "Acquisition Corp|Acquisition Inc|Blank Check|"
        "- Class A Ordinary|- Class B Ordinary"
    )
    df = df[~df["Security Name"].str.contains(_exclude, case=False, na=False, regex=True)]

    df = df[["Symbol", "Security Name"]].rename(
        columns={"Symbol": "ticker", "Security Name": "company"}
    ).reset_index(drop=True)

    logger.info(f"실제 상장 회사: {len(df):,}개")

    if save:
        SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(SAVE_PATH, index=False)
        logger.info(f"저장 완료: {SAVE_PATH}")

    return df


def load_universe() -> pd.DataFrame:
    if SAVE_PATH.exists():
        return pd.read_parquet(SAVE_PATH)
    return get_universe()


if __name__ == "__main__":
    df = get_universe()
    print(df.head(10))
    print(f"\n총 {len(df):,}개 종목")
