import time
import yfinance as yf
from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))
from theme_tracker.config import THEME_KEYWORDS, DELAY


def classify(ticker: str) -> tuple[str, str]:
    """
    Returns (theme, sub_theme).
    theme = "AI인프라" / "우주" / ... / "none"
    sub_theme = "power_supply" / "launch" / ... / ""
    """
    try:
        info = yf.Ticker(ticker).info
        desc = " ".join([
            info.get("longBusinessSummary", ""),
            info.get("longName", ""),
            info.get("industry", ""),
            info.get("sector", ""),
        ]).lower()
    except Exception as e:
        logger.debug(f"{ticker} info 실패: {e}")
        return "none", ""

    if not desc.strip():
        return "none", ""

    for theme, subs in THEME_KEYWORDS.items():
        for sub, keywords in subs.items():
            if any(k.lower() in desc for k in keywords):
                return theme, sub

    # 키워드 불일치 → Groq LLM 분류 시도
    return _classify_with_llm(ticker, desc)


def _classify_with_llm(ticker: str, desc: str) -> tuple[str, str]:
    """Groq (무료) LLM으로 테마 분류."""
    try:
        import os, json, requests
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "none", ""

        themes_list = list(THEME_KEYWORDS.keys())
        prompt = (
            f"다음 회사의 사업 설명을 보고 가장 맞는 테마 하나를 선택하세요.\n"
            f"선택지: {themes_list + ['none']}\n\n"
            f"회사({ticker}) 설명: {desc[:500]}\n\n"
            f'JSON으로만 응답: {{"theme": "테마명", "sub": "세부카테고리 한 단어"}}'
        )
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": "llama3-8b-8192", "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 60, "temperature": 0},
            timeout=10,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        parsed = json.loads(text[text.find("{"):text.rfind("}")+1])
        theme = parsed.get("theme", "none")
        sub   = parsed.get("sub", "")
        if theme not in THEME_KEYWORDS and theme != "none":
            return "none", ""
        return theme, sub
    except Exception as e:
        logger.debug(f"Groq 분류 실패 ({ticker}): {e}")
        return "none", ""


def batch_classify(tickers: list[str]) -> list[dict]:
    results = []
    for ticker in tickers:
        theme, sub = classify(ticker)
        results.append({"ticker": ticker, "테마": theme, "서브테마": sub})
        time.sleep(DELAY)
    return results
