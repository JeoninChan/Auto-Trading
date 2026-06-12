"""
Day Trading — 전 유니버스 실시간 신호 생성기

사용:
  python3 day_trading/live_signal.py              # 매 시간 :02분 자동 스캔
  python3 day_trading/live_signal.py --once       # 1회만 즉시 실행
  python3 day_trading/live_signal.py --top 50     # 상위 50개 출력
  python3 day_trading/live_signal.py --case smallcap
"""

import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yfinance as yf
from apscheduler.schedulers.blocking import BlockingScheduler
from loguru import logger

warnings.filterwarnings('ignore')

BASE_DIR    = Path(__file__).parent
MODEL_DIR   = BASE_DIR / 'models'
SIGNALS_LOG = BASE_DIR / 'signals_log.json'

THRESHOLD   = 0.35
BATCH_SIZE  = 50
BATCH_DELAY = 1.0
INTERVAL    = '1h'
SPY_LAG_WARN_SEC = 300  # SPY 타임스탬프 허용 최대 랙 (초)

UPDATE_PERIOD_MAP = {
    '1m': '3d', '2m': '7d', '5m': '7d',
    '15m': '14d', '30m': '14d',
    '1h': '5d', '1d': '30d',
}

PRICE_DIR = BASE_DIR / 'data' / f'price_{INTERVAL}'


# ─── SPY 실시간 fetch ─────────────────────────────────────────────────────────
def fetch_spy_live(interval: str = '1h') -> pd.DataFrame:
    period = UPDATE_PERIOD_MAP.get(interval, '5d')
    try:
        spy = yf.download('SPY', period=period, interval=interval,
                          auto_adjust=True, progress=False)
        if not spy.empty:
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.get_level_values(0)
            spy.columns = [c.lower() for c in spy.columns]
            return spy
    except Exception as e:
        logger.warning(f'SPY 실시간 fetch 실패: {e}')
    # 캐시 폴백
    spy_path = PRICE_DIR / 'SPY.parquet'
    if spy_path.exists():
        df = pd.read_parquet(spy_path)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        return df
    return pd.DataFrame()


# ─── 피처 계산 (local_trainer.py의 make_features와 동일 로직) ────────────────
def make_features(df: pd.DataFrame, spy_df: pd.DataFrame | None = None) -> pd.DataFrame:
    from local_trainer import make_features as _make_features
    return _make_features(df, spy_df=spy_df)


# ─── 최신 봉 업데이트 ─────────────────────────────────────────────────────────
def update_prices(tickers: list[str]):
    period = UPDATE_PERIOD_MAP.get(INTERVAL, '5d')
    logger.info(f'최신 봉 업데이트 중... ({len(tickers):,}개, {INTERVAL}/{period})')
    updated = 0

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        try:
            raw = yf.download(batch, period=period, interval=INTERVAL,
                              auto_adjust=True, progress=False, threads=True)
            if raw.empty:
                time.sleep(BATCH_DELAY)
                continue

            if isinstance(raw.columns, pd.MultiIndex):
                for ticker in batch:
                    try:
                        df_new = raw.xs(ticker, axis=1, level=1).dropna(how='all')
                        if df_new.empty:
                            continue
                        _append_parquet(ticker, df_new)
                        updated += 1
                    except Exception:
                        pass
            else:
                ticker = batch[0]
                if not raw.empty:
                    _append_parquet(ticker, raw)
                    updated += 1

        except Exception as e:
            logger.debug(f'배치 {i} 실패: {e}')

        time.sleep(BATCH_DELAY)

    logger.info(f'업데이트 완료: {updated:,}개')


def _append_parquet(ticker: str, df_new: pd.DataFrame):
    path = PRICE_DIR / f'{ticker}.parquet'
    if path.exists():
        df_old = pd.read_parquet(path)
        df_combined = pd.concat([df_old, df_new])
        df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
        df_combined = df_combined.sort_index()
    else:
        df_combined = df_new.sort_index()
    df_combined.to_parquet(path)


# ─── 전 종목 스캔 ─────────────────────────────────────────────────────────────
def scan_all(
    model: lgb.Booster,
    top_feats: list[str],
    spy_df: pd.DataFrame,
    calibrator=None,
) -> list[dict]:
    parquet_files = sorted(PRICE_DIR.glob('*.parquet'))
    results = []

    for path in parquet_files:
        ticker = path.stem
        if ticker == 'SPY':
            continue
        try:
            df = pd.read_parquet(path).tail(500)
            if len(df) < 60:
                continue

            # SPY 타임스탬프 씽크 체크
            spy_safe: pd.DataFrame | None = spy_df
            if not spy_df.empty:
                last_tick = df.index[-1]
                last_spy  = spy_df.index[-1]
                lag_sec   = abs((last_tick - last_spy).total_seconds())
                if lag_sec > SPY_LAG_WARN_SEC:
                    logger.warning(
                        f'{ticker}: SPY 타임스탬프 랙 {lag_sec:.0f}초 '
                        f'(종목={last_tick}, SPY={last_spy}) — RS 피처 비활성화'
                    )
                    spy_safe = None

            df = make_features(df, spy_df=spy_safe).dropna()
            if df.empty:
                continue
            missing = [f for f in top_feats if f not in df.columns]
            if missing:
                continue
            row   = df[top_feats].iloc[[-1]].astype(float)
            raw   = float(model.predict(row)[0])
            proba = float(calibrator.predict([raw])[0]) if calibrator else raw

            results.append({
                'ticker':   ticker,
                'proba':    round(proba, 4),
                'signal':   'BUY' if proba > THRESHOLD else 'HOLD',
                'bar_time': str(df.index[-1]),
            })
        except Exception:
            pass

    results.sort(key=lambda x: x['proba'], reverse=True)
    return results


# ─── 출력 ─────────────────────────────────────────────────────────────────────
def print_results(results: list[dict], top_n: int):
    buy_count = sum(1 for r in results if r['signal'] == 'BUY')
    total     = len(results)
    now       = datetime.now().strftime('%Y-%m-%d %H:%M')

    print(f'\n[{now}] 스캔 완료 — {total:,}개 종목')
    print('─' * 58)
    print(f'{"#":>4}  {"티커":<8}  {"확률":>6}  신호')
    print('─' * 58)
    for i, r in enumerate(results[:top_n], 1):
        label = '▶ BUY ◀' if r['signal'] == 'BUY' else '── HOLD'
        print(f'{i:>4}  {r["ticker"]:<8}  {r["proba"]:>6.3f}  {label}')
    print('─' * 58)
    print(f'BUY 신호: {buy_count:,}개 / {total:,}개 ({buy_count/max(total,1)*100:.1f}%)')
    print(f'기준 threshold: {THRESHOLD}\n')


def save_signals(results: list[dict]):
    buy_signals = [r for r in results if r['signal'] == 'BUY']
    if not buy_signals:
        return
    record = {
        'timestamp': datetime.now().isoformat(),
        'count':     len(buy_signals),
        'signals':   buy_signals,
    }
    existing = json.loads(SIGNALS_LOG.read_text()) if SIGNALS_LOG.exists() else []
    existing.append(record)
    SIGNALS_LOG.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
    logger.info(f'신호 저장: {len(buy_signals)}개 → {SIGNALS_LOG.name}')


# ─── 메인 스캔 작업 ───────────────────────────────────────────────────────────
def run_scan(
    model: lgb.Booster,
    top_feats: list[str],
    top_n: int,
    calibrator=None,
):
    t0      = time.time()
    tickers = [p.stem for p in sorted(PRICE_DIR.glob('*.parquet')) if p.stem != 'SPY']

    if not tickers:
        logger.error('parquet 파일 없음. local_trainer.py 먼저 실행하세요.')
        return

    update_prices(tickers)
    spy_df = fetch_spy_live(INTERVAL)
    if spy_df.empty:
        logger.warning('SPY 데이터 없음 — RS 피처 없이 스캔')

    results = scan_all(model, top_feats, spy_df, calibrator=calibrator)
    print_results(results, top_n)
    save_signals(results)

    elapsed = time.time() - t0
    logger.info(f'총 소요: {elapsed/60:.1f}분')


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    global INTERVAL, PRICE_DIR, THRESHOLD
    parser = argparse.ArgumentParser(description='전 유니버스 실시간 신호 생성기')
    parser.add_argument('--case', default='all',
                        choices=['smallcap', 'small_mid', 'mid_large', 'all', 'leverage_mid_large'])
    parser.add_argument('--interval',  default=INTERVAL,
                        choices=['1m', '2m', '5m', '15m', '30m', '1h', '1d'])
    parser.add_argument('--threshold', type=float, default=THRESHOLD)
    parser.add_argument('--top',  type=int, default=20)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()

    if args.interval != INTERVAL:
        INTERVAL  = args.interval
        PRICE_DIR = BASE_DIR / 'data' / f'price_{INTERVAL}'
        logger.info(f'인터벌 변경: {INTERVAL}')

    if args.threshold != THRESHOLD:
        THRESHOLD = args.threshold
        logger.info(f'threshold 변경: {THRESHOLD}')

    lgb_path  = MODEL_DIR / f'model_{args.case}_lgb.txt'
    feat_path = MODEL_DIR / f'features_{args.case}.json'
    cal_path  = MODEL_DIR / f'calibrator_{args.case}_lgb.pkl'

    if not lgb_path.exists():
        logger.error(f'모델 없음: {lgb_path}')
        logger.error('먼저 실행: python3 day_trading/local_trainer.py --skip-download')
        return

    model     = lgb.Booster(model_file=str(lgb_path))
    top_feats = json.loads(feat_path.read_text())
    cal       = joblib.load(cal_path) if cal_path.exists() else None
    if cal:
        logger.info(f'캘리브레이터 로드: {cal_path.name}')
    else:
        logger.warning(f'캘리브레이터 없음 — raw proba 사용 (재학습 권장)')
    logger.info(f'모델: {lgb_path.name}  피처: {len(top_feats)}개  threshold: {THRESHOLD}')

    if args.once:
        run_scan(model, top_feats, args.top, calibrator=cal)
        return

    run_scan(model, top_feats, args.top, calibrator=cal)

    scheduler = BlockingScheduler(timezone='Asia/Seoul')
    scheduler.add_job(
        run_scan, 'cron', minute=2,
        args=[model, top_feats, args.top, cal],
    )
    logger.info('스케줄러 시작 — 매 시간 :02분 자동 스캔 (Ctrl+C로 종료)')
    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info('종료')


if __name__ == '__main__':
    main()
