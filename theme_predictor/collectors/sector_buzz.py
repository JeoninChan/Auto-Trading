import json
import time
import requests
import feedparser
from pathlib import Path
from datetime import datetime
from loguru import logger

import sys; sys.path.insert(0, str(Path(__file__).parents[2]))
from theme_predictor.config import SECTOR_KEYWORDS, REDDIT_SUBS, DELAY

OUT = Path(__file__).parents[1] / "signals" / "buzz_trend.json"
HEADERS = {"User-Agent": "stock-research-bot hideinthecodes@gmail.com"}

NEWS_RSS = [
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
    "https://www.reutersagency.com/feed/?best-topics=business-finance&post_type=best",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://seekingalpha.com/feed.xml",
]


def fetch_google_trends(keywords: list[str], timeframe: str = "today 3-m") -> dict[str, int]:
    """pytrends로 Google 검색량 수집. 미설치 시 빈 dict 반환."""
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl="en-US", tz=360)
        results = {}
        for i in range(0, len(keywords), 5):
            batch = keywords[i:i+5]
            try:
                pytrends.build_payload(batch, timeframe=timeframe)
                df = pytrends.interest_over_time()
                if df.empty:
                    continue
                for kw in batch:
                    if kw in df.columns:
                        results[kw] = int(df[kw].iloc[-4:].mean())
                time.sleep(1.5)
            except Exception as e:
                logger.debug(f"Trends 배치 실패: {e}")
        return results
    except ImportError:
        logger.warning("pytrends 미설치 → pip install pytrends")
        return {}


def fetch_reddit_mentions(subreddits: list[str], keywords: dict[str, list[str]]) -> dict[str, int]:
    """Reddit JSON API (인증 불필요)로 섹터 키워드 언급 수 집계."""
    sector_count: dict[str, int] = {}

    for sub in subreddits:
        url = f"https://www.reddit.com/r/{sub}/new.json?limit=100"
        try:
            r = requests.get(url, headers={"User-Agent": "stock-research-bot/1.0"}, timeout=15)
            if r.status_code != 200:
                continue
            posts = r.json().get("data", {}).get("children", [])
            for post in posts:
                text = (post["data"].get("title", "") + " " + post["data"].get("selftext", "")).lower()
                for sector, kws in keywords.items():
                    if any(k.lower() in text for k in kws):
                        sector_count[sector] = sector_count.get(sector, 0) + 1
            time.sleep(DELAY)
        except Exception as e:
            logger.debug(f"Reddit r/{sub} 실패: {e}")

    return dict(sorted(sector_count.items(), key=lambda x: x[1], reverse=True))


def fetch_news_rss_mentions(keywords: dict[str, list[str]]) -> dict[str, int]:
    """뉴스 RSS 피드에서 섹터 키워드 언급 집계."""
    sector_count: dict[str, int] = {}

    for rss_url in NEWS_RSS:
        try:
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:50]:
                text = (entry.get("title", "") + " " + entry.get("summary", "")).lower()
                for sector, kws in keywords.items():
                    if any(k.lower() in text for k in kws):
                        sector_count[sector] = sector_count.get(sector, 0) + 1
            time.sleep(0.3)
        except Exception as e:
            logger.debug(f"RSS 실패 ({rss_url}): {e}")

    return dict(sorted(sector_count.items(), key=lambda x: x[1], reverse=True))


def fetch_finviz_news_mentions(keywords: dict[str, list[str]]) -> dict[str, int]:
    """Finviz 뉴스 페이지 파싱."""
    sector_count: dict[str, int] = {}
    try:
        from bs4 import BeautifulSoup
        r = requests.get("https://finviz.com/news.ashx", headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        headlines = [a.get_text() for a in soup.select("a.nn-tab-link, a[href*='/news/']")]
        for headline in headlines:
            text = headline.lower()
            for sector, kws in keywords.items():
                if any(k.lower() in text for k in kws):
                    sector_count[sector] = sector_count.get(sector, 0) + 1
    except Exception as e:
        logger.debug(f"Finviz 뉴스 실패: {e}")
    return sector_count


def run() -> dict:
    logger.info("=== 섹터 버즈/언급량 수집 ===")

    logger.info("Reddit 언급량 수집...")
    reddit = fetch_reddit_mentions(REDDIT_SUBS, SECTOR_KEYWORDS)

    logger.info("뉴스 RSS 언급량 수집...")
    rss = fetch_news_rss_mentions(SECTOR_KEYWORDS)

    logger.info("Finviz 뉴스 언급량...")
    finviz = fetch_finviz_news_mentions(SECTOR_KEYWORDS)

    logger.info("Google Trends 수집...")
    top_kws = [kws[0] for kws in SECTOR_KEYWORDS.values()]
    trends = fetch_google_trends(top_kws)

    # 섹터별 통합 합산
    all_sectors = set(list(reddit.keys()) + list(rss.keys()) + list(finviz.keys()))
    combined: dict[str, dict] = {}
    for s in all_sectors:
        combined[s] = {
            "reddit": reddit.get(s, 0),
            "rss_news": rss.get(s, 0),
            "finviz": finviz.get(s, 0),
            "total": reddit.get(s, 0) + rss.get(s, 0) + finviz.get(s, 0),
        }

    sorted_combined = dict(sorted(combined.items(), key=lambda x: x[1]["total"], reverse=True))

    output = {
        "updated": datetime.now().isoformat(),
        "google_trends": trends,
        "sector_mentions": sorted_combined,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    logger.success(f"저장 → {OUT}")
    return output


if __name__ == "__main__":
    run()
