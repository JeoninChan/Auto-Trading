"""
미국 주식 뉴스 수집 (RSS 피드)
Yahoo Finance RSS, SEC RSS, Seeking Alpha RSS
"""
import time
import feedparser
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger

SAVE_DIR = Path(__file__).parents[3] / "datasets" / "us" / "news"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

RSS_SOURCES = {
    "yahoo":     "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
    "sec_edgar": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={ticker}&type=8-K&dateb=&owner=include&count=10&search_text=&output=atom",
}


def fetch_rss(ticker: str) -> list[dict]:
    entries = []
    for source, url_tpl in RSS_SOURCES.items():
        url = url_tpl.format(ticker=ticker)
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:20]:
                entries.append({
                    "ticker":    ticker,
                    "source":    source,
                    "title":     entry.get("title", ""),
                    "summary":   entry.get("summary", "")[:500],
                    "link":      entry.get("link", ""),
                    "published": entry.get("published", ""),
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                })
        except Exception as e:
            logger.warning(f"{ticker} RSS({source}) 실패: {e}")
        time.sleep(0.3)

    return entries


def collect_ticker(ticker: str) -> pd.DataFrame:
    entries = fetch_rss(ticker)
    if not entries:
        return pd.DataFrame()

    df = pd.DataFrame(entries)
    df.to_parquet(SAVE_DIR / f"{ticker}.parquet", index=False)
    logger.info(f"{ticker}: {len(df)}개 뉴스 수집")
    return df


def collect_all(tickers: list[str]) -> None:
    logger.info(f"US 뉴스 수집: {len(tickers)}개 종목")
    for ticker in tickers:
        collect_ticker(ticker)
        time.sleep(0.5)


def load_news(ticker: str) -> pd.DataFrame:
    path = SAVE_DIR / f"{ticker}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


if __name__ == "__main__":
    test = ["NVDA", "IONQ", "AAPL", "SHLS"]
    for t in test:
        df = collect_ticker(t)
        if not df.empty:
            print(f"\n{t}: {df['title'].head(3).tolist()}")
