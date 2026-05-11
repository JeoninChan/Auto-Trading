import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[3]))

"""
한국 주식 뉴스 수집 (네이버 금융 크롤링)
종목별 최근 뉴스 제목 + 요약
"""
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from data.collectors.kr.universe import load_universe

SAVE_DIR = Path(__file__).parents[3] / "datasets" / "kr" / "news"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}
BASE_URL = "https://finance.naver.com/item/news_news.naver?code={ticker}&page=1"


def fetch_news(ticker: str, max_pages: int = 3) -> list[dict]:
    entries = []
    for page in range(1, max_pages + 1):
        url = f"https://finance.naver.com/item/news_news.naver?code={ticker}&page={page}"
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")

            rows = soup.select("table.type5 tr")
            for row in rows:
                title_tag = row.select_one("td.title a")
                date_tag  = row.select_one("td.date")
                if not title_tag:
                    continue
                entries.append({
                    "ticker":       ticker,
                    "title":        title_tag.get_text(strip=True),
                    "link":         "https://finance.naver.com" + title_tag.get("href", ""),
                    "published":    date_tag.get_text(strip=True) if date_tag else "",
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                })
        except Exception as e:
            logger.warning(f"{ticker} 뉴스 p{page} 실패: {e}")
            break
        time.sleep(0.5)

    return entries


def collect_ticker(ticker: str) -> pd.DataFrame:
    entries = fetch_news(ticker)
    if not entries:
        return pd.DataFrame()

    df = pd.DataFrame(entries)
    df.to_parquet(SAVE_DIR / f"{ticker}.parquet", index=False)
    logger.info(f"{ticker}: {len(df)}개 뉴스 수집")
    return df


def collect_all(tickers: list[str] | None = None) -> None:
    if tickers is None:
        universe = load_universe()
        tickers = universe["ticker"].tolist()

    logger.info(f"KR 뉴스 수집: {len(tickers):,}개 종목")
    for ticker in tqdm(tickers, desc="KR 뉴스 수집"):
        collect_ticker(ticker)
        time.sleep(0.5)


def load_news(ticker: str) -> pd.DataFrame | None:
    path = SAVE_DIR / f"{ticker}.parquet"
    return pd.read_parquet(path) if path.exists() else None


if __name__ == "__main__":
    test = ["005930", "000660", "035720"]
    for t in test:
        df = collect_ticker(t)
        if not df.empty:
            print(f"\n{t}: {df['title'].head(3).tolist()}")
