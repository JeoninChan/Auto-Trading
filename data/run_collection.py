"""
데이터 수집 통합 실행 스크립트
사용법:
    python data/run_collection.py --market all       # 전체
    python data/run_collection.py --market us        # 미국만
    python data/run_collection.py --market kr        # 한국만
    python data/run_collection.py --market us --tickers NVDA AAPL IONQ
    python data/run_collection.py --market kr --tickers 005930 000660
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from loguru import logger


def run_us(tickers: list[str] | None = None):
    from data.collectors.us.universe  import get_universe
    from data.collectors.us.price     import download_all as price
    from data.collectors.us.financial import download_all as financial
    from data.collectors.us.sec       import collect_all  as sec
    from data.collectors.us.news      import collect_all  as news

    logger.info("===== 미국 데이터 수집 시작 =====")
    if tickers is None:
        logger.info("유니버스 수집 중...")
        get_universe()

    logger.info("[1/4] 가격 수집")
    price(tickers)

    logger.info("[2/4] 재무제표 수집")
    financial(tickers)

    logger.info("[3/4] SEC 공시 수집")
    sec(tickers or [])

    logger.info("[4/4] 뉴스 수집")
    news(tickers or [])

    logger.info("===== 미국 완료 =====")


def run_kr(tickers: list[str] | None = None):
    from data.collectors.kr.universe   import get_universe
    from data.collectors.kr.price      import download_all as price
    from data.collectors.kr.financial  import download_all as financial
    from data.collectors.kr.disclosure import download_all as disclosure
    from data.collectors.kr.news       import collect_all  as news

    logger.info("===== 한국 데이터 수집 시작 =====")
    if tickers is None:
        logger.info("유니버스 수집 중...")
        get_universe()

    logger.info("[1/4] 가격 수집")
    price(tickers)

    logger.info("[2/4] 재무제표 수집")
    financial(tickers)

    logger.info("[3/4] 공시 수집")
    disclosure(tickers)

    logger.info("[4/4] 뉴스 수집")
    news(tickers)

    logger.info("===== 한국 완료 =====")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--market",  choices=["us", "kr", "all"], default="all")
    parser.add_argument("--tickers", nargs="*", default=None)
    args = parser.parse_args()

    if args.market in ("us", "all"):
        run_us(args.tickers)

    if args.market in ("kr", "all"):
        run_kr(args.tickers)
