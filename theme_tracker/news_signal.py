import time
import requests
import yfinance as yf
import feedparser
from datetime import datetime, timedelta
from loguru import logger
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from theme_tracker.config import DELAY

SEC_HEADERS = {"User-Agent": "stock-research-bot hideinthecodes@gmail.com"}
SAM_RSS = "https://sam.gov/api/prod/opportunities/v2/search?limit=25&offset=0&postedFrom={from_date}&postedTo={to_date}&ptype=o"

CONFERENCE_CALENDAR = {
    "2026-07-14": "National Space Symposium",
    "2026-08-11": "Hot Chips (반도체 컨퍼런스)",
    "2026-09-08": "Citi Global Technology Conference",
    "2026-09-15": "Deutsche Bank Tech Conference",
    "2026-10-05": "Space Symposium Europe",
    "2026-11-18": "Supercomputing SC26",
}


def check_earnings_imminent(ticker: str, days: int = 14) -> dict:
    """실적 발표일이 N일 이내인지 확인."""
    try:
        cal = yf.Ticker(ticker).calendar
        if cal is None or cal.empty:
            return {"imminent": False, "date": None, "days_left": None}
        earn_date = cal.iloc[0, 0] if hasattr(cal.iloc[0, 0], "date") else None
        if earn_date is None:
            return {"imminent": False, "date": None, "days_left": None}
        days_left = (earn_date.date() - datetime.today().date()).days
        return {
            "imminent": 0 <= days_left <= days,
            "date": str(earn_date.date()),
            "days_left": days_left,
        }
    except Exception:
        return {"imminent": False, "date": None, "days_left": None}


def check_8k_surge(ticker: str) -> dict:
    """SEC 8-K 최근 30일 제출 빈도 vs 90일 평균 비교."""
    try:
        search = (
            f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22"
            f"&forms=8-K&dateRange=custom"
        )
        now = datetime.today()

        def count_filings(days_back: int) -> int:
            from_dt = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")
            r = requests.get(
                f"{search}&startdt={from_dt}&enddt={now.strftime('%Y-%m-%d')}",
                headers=SEC_HEADERS, timeout=10,
            )
            if r.status_code != 200:
                return 0
            return r.json().get("hits", {}).get("total", {}).get("value", 0)

        count_30 = count_filings(30)
        count_90 = count_filings(90)
        avg_monthly = round(count_90 / 3, 1)
        surge = count_30 >= max(avg_monthly * 1.5, avg_monthly + 2)
        return {
            "count_30d": count_30,
            "avg_monthly": avg_monthly,
            "surge": surge,
        }
    except Exception as e:
        logger.debug(f"{ticker} 8-K 조회 실패: {e}")
        return {"count_30d": 0, "avg_monthly": 0, "surge": False}


def check_sam_gov(ticker: str) -> dict:
    """SAM.gov에서 회사명 관련 정부 계약 공고 확인."""
    try:
        info = yf.Ticker(ticker).info
        company_name = info.get("shortName") or info.get("longName") or ticker
        company_clean = company_name.split()[0]

        now = datetime.today()
        from_dt = (now - timedelta(days=30)).strftime("%m/%d/%Y")
        to_dt = now.strftime("%m/%d/%Y")

        url = (
            f"https://sam.gov/api/prod/opportunities/v2/search"
            f"?limit=10&offset=0&postedFrom={from_dt}&postedTo={to_dt}"
            f"&ptype=o&keyword={company_clean}"
        )
        r = requests.get(url, headers={"User-Agent": "stock-research-bot"}, timeout=15)
        if r.status_code != 200:
            return {"count": 0, "titles": []}
        data = r.json()
        opps = data.get("opportunitiesData", []) or []
        titles = [o.get("title", "") for o in opps[:5]]
        return {"count": len(opps), "titles": titles}
    except Exception as e:
        logger.debug(f"{ticker} SAM.gov 실패: {e}")
        return {"count": 0, "titles": []}


def check_news_rss(ticker: str) -> dict:
    """Yahoo Finance RSS에서 최근 7일 뉴스 건수 확인."""
    try:
        url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
        feed = feedparser.parse(url)
        cutoff = datetime.now() - timedelta(days=7)
        recent = []
        for e in feed.entries[:20]:
            try:
                pub = datetime(*e.published_parsed[:6])
                if pub >= cutoff:
                    recent.append(e.get("title", ""))
            except Exception:
                pass
        return {"count_7d": len(recent), "headlines": recent[:3]}
    except Exception:
        return {"count_7d": 0, "headlines": []}


def check_conference(ticker: str) -> dict | None:
    """config에 등록된 컨퍼런스 일정 중 30일 이내 항목."""
    today = datetime.today().date()
    upcoming = []
    for date_str, name in CONFERENCE_CALENDAR.items():
        event_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        days_left = (event_date - today).days
        if 0 <= days_left <= 30:
            upcoming.append({"event": name, "date": date_str, "days_left": days_left})
    return upcoming if upcoming else []


def assess(ticker: str) -> dict:
    """3+1가지 신호 종합 → 뉴스 임박도 판단."""
    earnings = check_earnings_imminent(ticker)
    sec8k    = check_8k_surge(ticker)
    sam      = check_sam_gov(ticker)
    rss      = check_news_rss(ticker)
    confs    = check_conference(ticker)

    signals = []
    if earnings["imminent"]:
        signals.append(f"실적 D-{earnings['days_left']} ({earnings['date']})")
    if sec8k["surge"]:
        signals.append(f"8-K 급증 ({sec8k['count_30d']}건/30일, 평균 {sec8k['avg_monthly']}건)")
    if sam["count"] > 0:
        signals.append(f"정부계약 공고 {sam['count']}건")
    if rss["count_7d"] >= 3:
        signals.append(f"뉴스 {rss['count_7d']}건/7일")
    if confs:
        signals.append(f"컨퍼런스 D-{confs[0]['days_left']} ({confs[0]['event']})")

    if len(signals) >= 3:
        level = "높음"
    elif len(signals) >= 1:
        level = "중간"
    else:
        level = "낮음"

    return {
        "뉴스임박": level,
        "뉴스임박근거": " | ".join(signals) if signals else "—",
        "_detail": {
            "earnings": earnings,
            "8k_surge": sec8k,
            "sam_gov": sam,
            "rss_7d": rss,
            "conferences": confs,
        },
    }


def batch_assess(tickers: list[str]) -> dict[str, dict]:
    results = {}
    for ticker in tickers:
        results[ticker] = assess(ticker)
        time.sleep(DELAY)
    return results
