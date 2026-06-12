"""
Day Trading — 로컬 데이터 수집 + LightGBM/XGBoost 모델 학습 + 백테스트 (M4 Pro 최적화)

레이블: Triple Barrier Method (López de Prado, 2018)
피처 : RS(상대강도), ORB(오프닝 레인지 돌파), VWAP 거리, 매물대 — RSI/MACD 제거
캘리브레이션: IsotonicRegression (학습 후 자동 적용)

사용:
  python day_trading/local_trainer.py                                                        # 전체 학습 + 백테스트 (기본)
  python day_trading/local_trainer.py --skip-download                                       # 학습 + 백테스트 (다운로드 스킵)
  python day_trading/local_trainer.py --skip-download --no-backtest                         # 학습만 (백테스트 생략)
  python day_trading/local_trainer.py --backtest-only --cases all                           # 백테스트만
  python day_trading/local_trainer.py --skip-download --replay --replay-start 2024-01-01   # replay 시뮬
  python day_trading/local_trainer.py --interval 5m                                         # 5분봉 학습+백테스트
"""

import argparse
import json
import random
import time
import warnings
from datetime import datetime
from functools import partial
from io import StringIO
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_ta as ta  # noqa: F401
import psutil
import requests
import xgboost as xgb
import yfinance as yf
from loguru import logger
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import classification_report

warnings.filterwarnings('ignore')

# ─── 경로 ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / 'data'
PRICE_DIR = DATA_DIR / 'price_1h'
MODEL_DIR = BASE_DIR / 'models'
MC_CACHE        = DATA_DIR / 'market_caps.json'
FINANCIALS_DIR  = DATA_DIR / 'financials'
RESULTS_PATH    = MODEL_DIR / 'backtest_results.json'

DATA_DIR.mkdir(parents=True, exist_ok=True)
PRICE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FINANCIALS_DIR.mkdir(parents=True, exist_ok=True)

# ─── CONFIG ──────────────────────────────────────────────────────────────────
LARGE_CAP_USD  = 17_000_000_000
MID_CAP_USD    =  1_400_000_000
INTERVAL         = '1h'

# 인터벌별 기본값 — 수동 지정 없을 때 자동 적용
TAKE_PROFIT_MAP = {'1m': 0.005, '2m': 0.007, '5m': 0.010, '15m': 0.015, '30m': 0.020, '1h': 0.080, '1d': 0.050}
STOP_LOSS_MAP   = {'1m': -0.003,'2m': -0.005,'5m': -0.007,'15m': -0.010,'30m': -0.013,'1h': -0.030,'1d': -0.030}
THRESHOLD_MAP   = {'1m': 0.55,  '2m': 0.50,  '5m': 0.45,  '15m': 0.40,  '30m': 0.35,  '1h': 0.35,  '1d': 0.35}

THRESHOLD        = THRESHOLD_MAP.get(INTERVAL, 0.35)
TRIPLE_BARRIER   = True     # Triple Barrier 레이블 사용 (기본값)
TOP_FEATURES   = 30
HV_WINDOW      = 240        # 매물대 탐색 윈도우 (봉 수)
BATCH_MC       = 100
BATCH_PRICE    = 50
DELAY_SEC      = 2.0
MC_DELAY       = 0.3

PERIOD_MAP = {
    '1m': '7d', '2m': '60d', '5m': '60d',
    '15m': '60d', '30m': '60d',
    '1h': '2y', '1d': 'max',
}
PERIOD = PERIOD_MAP.get(INTERVAL, '2y')

RETURN_THRESH_MAP = {
    '1m': 0.003, '2m': 0.003, '5m': 0.005,
    '15m': 0.006, '30m': 0.007,
    '1h': 0.008, '1d': 0.020,
}
RETURN_THRESH = RETURN_THRESH_MAP.get(INTERVAL, 0.008)

# 인터벌별 케이스별 포워드 바 수
# smallcap(잡주) / small_mid(잡주+중형) / mid_large(중형+우량) / all / leverage_mid_large
FORWARD_BARS_MAP_BY_INTERVAL = {
    '1m':  {'smallcap': 30, 'small_mid': 60,  'mid_large':  90, 'all': 30, 'leverage_mid_large': 180},
    '2m':  {'smallcap': 15, 'small_mid': 30,  'mid_large':  45, 'all': 15, 'leverage_mid_large':  90},
    '5m':  {'smallcap': 12, 'small_mid': 18,  'mid_large':  24, 'all': 12, 'leverage_mid_large':  48},
    '15m': {'smallcap':  6, 'small_mid':  8,  'mid_large':  12, 'all':  6, 'leverage_mid_large':  24},
    '30m': {'smallcap':  6, 'small_mid':  8,  'mid_large':  12, 'all':  6, 'leverage_mid_large':  16},
    '1h':  {'smallcap':  3, 'small_mid':  5,  'mid_large':   7, 'all':  3, 'leverage_mid_large':   8},
    '1d':  {'smallcap':  5, 'small_mid':  7,  'mid_large':  10,            'leverage_mid_large':  14},
}
FORWARD_BARS_MAP = FORWARD_BARS_MAP_BY_INTERVAL.get(INTERVAL, FORWARD_BARS_MAP_BY_INTERVAL['1h'])

BACKTEST_MIN_SHARPE    = 1.5
BACKTEST_MAX_DD        = -0.20
BACKTEST_MIN_WINRATE   = 0.50
BACKTEST_MIN_PF        = 1.50


# ─── [1] 유니버스 수집 ────────────────────────────────────────────────────────
def fetch_universe() -> list[str]:
    logger.info('유니버스 수집 중 (NASDAQ Trader)...')
    url = 'https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt'
    r = requests.get(url, headers={'User-Agent': 'stock-research-bot hideinthecodes@gmail.com'}, timeout=30)
    df = pd.read_csv(StringIO(r.text), sep='|')[:-1]

    df = df[(df['ETF'] == 'N') & (df['Test Issue'] == 'N')]
    df = df[~df['Symbol'].str.contains(r'[.\-+^$]', regex=True, na=True)]

    excl = (
        '- Rights|- Units|- Warrant|Depositary Share|'
        'Acquisition Corp|Acquisition Inc|Blank Check|'
        '- Class A Ordinary|- Class B Ordinary'
    )
    df = df[~df['Security Name'].str.contains(excl, case=False, na=False, regex=True)]

    tickers = df['Symbol'].tolist()
    logger.info(f'유니버스: {len(tickers):,}개 종목')
    return tickers


# ─── [2] 시가총액 수집 ────────────────────────────────────────────────────────
def _fetch_mc(ticker: str) -> tuple[str, float]:
    try:
        mc = yf.Ticker(ticker).info.get('marketCap', None)
        return ticker, float(mc) if mc else 0.0
    except Exception:
        return ticker, 0.0


def fetch_market_caps(all_tickers: list[str]) -> dict[str, float]:
    market_caps: dict[str, float] = {}
    if MC_CACHE.exists():
        market_caps = json.loads(MC_CACHE.read_text())
        logger.info(f'시총 캐시 로드: {len(market_caps):,}개')

    todo = [t for t in all_tickers if t not in market_caps or market_caps[t] == 0.0]
    logger.info(f'시총 수집 필요: {len(todo):,}개 (캐시 0값 포함)')

    for i, ticker in enumerate(todo):
        t, mc = _fetch_mc(ticker)
        market_caps[t] = mc
        time.sleep(MC_DELAY)

        if (i + 1) % BATCH_MC == 0 or i == len(todo) - 1:
            MC_CACHE.write_text(json.dumps(market_caps))
            logger.info(f'  시총 {i+1}/{len(todo)} ({(i+1)/len(todo)*100:.1f}%)')

    logger.success(f'시총 수집 완료: {len(market_caps):,}개')
    return market_caps


# ─── [2-B] 분기 재무 수집 ────────────────────────────────────────────────────
def fetch_all_financials(tickers: list[str]):
    need = [t for t in tickers if not (FINANCIALS_DIR / f'{t}.parquet').exists()]
    logger.info(f'재무 다운로드 필요: {len(need):,}개  캐시: {len(tickers)-len(need):,}개')

    BS_COLS  = ['Ordinary Shares Number', 'Cash And Cash Equivalents', 'Total Debt']
    INC_COLS = ['Total Revenue', 'Operating Income', 'Net Income']

    for i, ticker in enumerate(need):
        try:
            t   = yf.Ticker(ticker)
            bs  = t.quarterly_balance_sheet.T
            inc = t.quarterly_income_stmt.T

            bc = [c for c in BS_COLS  if c in bs.columns]
            ic = [c for c in INC_COLS if c in inc.columns]
            df = bs[bc].join(inc[ic], how='outer').sort_index()
            df.index.name = 'date'

            if not df.empty:
                df.to_parquet(FINANCIALS_DIR / f'{ticker}.parquet')
        except Exception:
            pass

        time.sleep(0.3)
        if (i + 1) % 500 == 0 or i == len(need) - 1:
            logger.info(f'  재무 {i+1}/{len(need)} ({(i+1)/len(need)*100:.1f}%)')

    saved = list(FINANCIALS_DIR.glob('*.parquet'))
    logger.success(f'재무 수집 완료: {len(saved):,}개  →  {FINANCIALS_DIR}')


# ─── [2-C] SPY 데이터 수집 (상대강도 계산용) ─────────────────────────────────
def fetch_spy_data() -> pd.DataFrame:
    """SPY 가격 다운로드 및 캐시. 상대강도(RS) 피처 계산에 사용."""
    spy_path = PRICE_DIR / 'SPY.parquet'
    try:
        spy = yf.download('SPY', period=PERIOD, interval=INTERVAL,
                          auto_adjust=True, progress=False)
        if not spy.empty:
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.get_level_values(0)
            spy.columns = [c.lower() for c in spy.columns]
            spy.to_parquet(spy_path)
            logger.info(f'SPY 데이터: {len(spy):,}봉  → {spy_path.name}')
            return spy
    except Exception as e:
        logger.warning(f'SPY 다운로드 실패: {e}')

    if spy_path.exists():
        df = pd.read_parquet(spy_path)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        logger.info(f'SPY 캐시 로드: {len(df):,}봉')
        return df

    logger.warning('SPY 데이터 없음 — RS 피처 비활성화')
    return pd.DataFrame()


# ─── [2-D] VIX 데이터 수집 ────────────────────────────────────────────────────
def fetch_vix_data() -> pd.DataFrame:
    """VIX 가격 다운로드 및 캐시. make_features의 vix_regime 피처에 사용."""
    vix_path = PRICE_DIR / 'VIX.parquet'
    try:
        raw = yf.download('^VIX', period=PERIOD, interval=INTERVAL,
                          auto_adjust=True, progress=False)
        if not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw.columns = [c.lower() for c in raw.columns]
            raw.to_parquet(vix_path)
            logger.info(f'VIX 데이터: {len(raw):,}봉  → {vix_path.name}')
            return raw
    except Exception as e:
        logger.warning(f'VIX 다운로드 실패: {e}')

    if vix_path.exists():
        df = pd.read_parquet(vix_path)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        logger.info(f'VIX 캐시 로드: {len(df):,}봉')
        return df

    logger.warning('VIX 데이터 없음 — vix 피처 비활성화')
    return pd.DataFrame()


# ─── [3] 케이스별 티커 분류 ──────────────────────────────────────────────────
def classify_cases(market_caps: dict[str, float]) -> dict[str, list[str]]:
    large     = [t for t, mc in market_caps.items() if mc and mc >= LARGE_CAP_USD]
    mid       = [t for t, mc in market_caps.items() if mc and MID_CAP_USD <= mc < LARGE_CAP_USD]
    small     = [t for t, mc in market_caps.items() if mc and 0 < mc < MID_CAP_USD]
    all_valid = [t for t, mc in market_caps.items() if mc and mc > 0]

    cases = {
        'smallcap':           small,
        'small_mid':          small + mid,
        'mid_large':          mid + large,
        'all':                all_valid,
        'leverage_mid_large': mid + large,
    }
    for name, tickers in cases.items():
        logger.info(f'  {name:20s}: {len(tickers):,}개')
    return cases


# ─── [4] OHLCV 수집 ──────────────────────────────────────────────────────────
def download_prices(all_valid: list[str], skip: bool = False):
    if skip:
        downloaded = list(PRICE_DIR.glob('*.parquet'))
        logger.info(f'다운로드 스킵 — 기존 파일: {len(downloaded):,}개')
        return

    need = [t for t in all_valid if not (PRICE_DIR / f'{t}.parquet').exists()]
    logger.info(f'다운로드 필요: {len(need):,}개')

    for i in range(0, len(need), BATCH_PRICE):
        batch = need[i:i + BATCH_PRICE]
        try:
            raw = yf.download(
                batch,
                period=PERIOD,
                interval=INTERVAL,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if raw.empty:
                continue

            if isinstance(raw.columns, pd.MultiIndex):
                for ticker in batch:
                    try:
                        df_t = raw.xs(ticker, axis=1, level=1).dropna(how='all')
                        if len(df_t) > 100:
                            df_t.to_parquet(PRICE_DIR / f'{ticker}.parquet')
                    except Exception:
                        pass
            else:
                ticker = batch[0]
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                if len(raw) > 100:
                    raw.to_parquet(PRICE_DIR / f'{ticker}.parquet')

        except Exception as e:
            logger.warning(f'배치 {i} 실패: {e}')

        done = min(i + BATCH_PRICE, len(need))
        if (i // BATCH_PRICE) % 10 == 0:
            logger.info(f'  가격 {done}/{len(need)} ({done/len(need)*100:.1f}%)')
        time.sleep(DELAY_SEC)

    downloaded = list(PRICE_DIR.glob('*.parquet'))
    logger.success(f'가격 수집 완료: {len(downloaded):,}개')


# ─── [5] 피처 엔지니어링 ─────────────────────────────────────────────────────
def make_features(
    df: pd.DataFrame,
    spy_df: pd.DataFrame | None = None,
    vix_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    알파 피처 계산.
    제거: RSI, MACD, Stochastic (후행 지표, 공유된 edge 없음)
    추가: RS vs SPY, ORB, VWAP 거리, 매물대 노드, VIX 공포지수
    """
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]

    # ── 기본 가격 피처 ────────────────────────────────────────────────────────
    df['ret_1']  = df['close'].pct_change(1)
    df['ret_3']  = df['close'].pct_change(3)
    df['ret_6']  = df['close'].pct_change(6)
    df['ret_20'] = df['close'].pct_change(20)
    df['gap']       = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-9)
    df['hl_ratio']  = (df['high'] - df['low']) / (df['close'] + 1e-9)
    df['close_pos'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)

    # ── 거래량 ───────────────────────────────────────────────────────────────
    vol_ema = df['volume'].ewm(span=20).mean()
    df['vol_ratio']   = df['volume'] / (vol_ema + 1e-9)
    df['vol_ratio_5'] = df['volume'] / (df['volume'].rolling(5).mean() + 1e-9)

    # ── 기술적 지표 (ATR, BB, ADX, EMA, OBV 유지) ───────────────────────────
    df.ta.atr(length=14, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.ema(length=9,  append=True)
    df.ta.ema(length=21, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.obv(append=True)

    ema9_cols  = [c for c in df.columns if 'ema' in c.lower() and '_9'  in c.lower()]
    ema21_cols = [c for c in df.columns if 'ema' in c.lower() and '21'  in c]
    ema50_cols = [c for c in df.columns if 'ema' in c.lower() and '50'  in c]
    atr_cols   = [c for c in df.columns if c.startswith('atr') and c != 'atr_pct']

    if ema21_cols:
        df['price_to_ema21'] = df['close'] / (df[ema21_cols[0]] + 1e-9)
    if ema9_cols and ema50_cols:
        df['ema_cross'] = (df[ema9_cols[0]] > df[ema50_cols[0]]).astype(int)
    if atr_cols:
        df['atr_pct'] = df[atr_cols[0]] / (df['close'] + 1e-9)

    # ── VWAP (일중 누적, 매일 리셋) ──────────────────────────────────────────
    _dates = (
        df.index.tz_convert('America/New_York').date
        if df.index.tz is not None else df.index.date
    )
    df['_date'] = _dates
    df['_tp']  = (df['high'] + df['low'] + df['close']) / 3
    df['_tpv'] = df['_tp'] * df['volume']
    df['_cum_tpv'] = df.groupby('_date')['_tpv'].cumsum()
    df['_cum_vol'] = df.groupby('_date')['volume'].cumsum()
    df['vwap']      = df['_cum_tpv'] / (df['_cum_vol'] + 1e-9)
    df['vwap_dist'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-9)
    df.drop(columns=['_tp', '_tpv', '_cum_tpv', '_cum_vol'], inplace=True)

    # ── Opening Range Breakout (ORB) ─────────────────────────────────────────
    # 장별 첫봉(09:30봉) 고저 → 당일 2번째 봉부터 ORB 피처 활성화
    # bar_of_day == 0 이 첫봉이므로 NaN 처리 (Data Leakage 방지)
    orb = df.groupby('_date').agg(
        orb_high=('high', 'first'),
        orb_low =('low',  'first'),
    )
    df = df.join(orb, on='_date')
    bar_of_day = df.groupby('_date').cumcount()
    df.loc[bar_of_day == 0, 'orb_high'] = np.nan
    df.loc[bar_of_day == 0, 'orb_low']  = np.nan

    df['orb_break_up']   = (df['close'] > df['orb_high']).astype(float)
    df['orb_break_down'] = (df['close'] < df['orb_low']).astype(float)
    day_first_vol        = df.groupby('_date')['volume'].transform('first')
    df['orb_vol_ratio']  = df['volume'] / (day_first_vol + 1e-9)
    df.drop(columns=['_date'], inplace=True)

    # ── Volatility Breakout (Larry Williams: target = open + k*(전일고-전일저)) ──
    df['vb_target'] = df['open'] + 0.5 * (df['high'].shift(1) - df['low'].shift(1))
    df['vb_break']  = (df['close'] > df['vb_target']).astype(float)
    df['vb_dist']   = (df['close'] - df['vb_target']) / (df['close'] + 1e-9)

    # ── 매물대 (High-Volume Price Node, numpy 벡터화) ─────────────────────────
    close_arr = df['close'].values
    vol_arr   = df['volume'].values
    hv_prices = np.full(len(df), np.nan)
    for i in range(HV_WINDOW, len(df)):
        hv_prices[i] = close_arr[i - HV_WINDOW:i][np.argmax(vol_arr[i - HV_WINDOW:i])]
    df['hv_node_price'] = hv_prices
    df['hv_node_above'] = (df['hv_node_price'] > df['close']).astype(float)
    df['hv_node_dist']  = (df['close'] - df['hv_node_price']) / (df['hv_node_price'] + 1e-9)

    # ── SPY 상대 강도 ─────────────────────────────────────────────────────────
    if spy_df is not None and not spy_df.empty:
        spy = spy_df.copy()
        spy.columns = [c.lower() for c in spy.columns]

        spy_ret1   = spy['close'].pct_change(1).rename('spy_ret_1')
        spy_ret20  = spy['close'].pct_change(20).rename('spy_ret_20')
        spy_ema20  = spy['close'].ewm(span=20).mean()
        spy_ema60  = spy['close'].ewm(span=60).mean()
        spy_regime = (spy_ema20 > spy_ema60).astype(int).rename('spy_regime')

        df = df.join(spy_ret1,   how='left')
        df = df.join(spy_ret20,  how='left')
        df = df.join(spy_regime, how='left')

        df['rs_vs_spy_1']  = df['ret_1']  - df['spy_ret_1']
        df['rs_vs_spy_20'] = df['ret_20'] - df['spy_ret_20']

        spy_cols = ['spy_ret_1', 'spy_ret_20', 'spy_regime', 'rs_vs_spy_1', 'rs_vs_spy_20']
        df[spy_cols] = df[spy_cols].ffill()
    else:
        df['spy_ret_1']    = np.nan
        df['spy_ret_20']   = np.nan
        df['spy_regime']   = np.nan
        df['rs_vs_spy_1']  = np.nan
        df['rs_vs_spy_20'] = np.nan

    # ── VIX 공포지수 피처 ─────────────────────────────────────────────────────
    if vix_df is not None and not vix_df.empty:
        _vix = vix_df.copy()
        _vix.columns = [c.lower() for c in _vix.columns]
        # timezone 정규화: join 전 df와 tz 맞춤
        if _vix.index.tz is not None and df.index.tz is None:
            _vix.index = _vix.index.tz_localize(None)
        elif _vix.index.tz is None and df.index.tz is not None:
            _vix.index = _vix.index.tz_localize(df.index.tz)
        elif _vix.index.tz is not None and df.index.tz is not None:
            _vix.index = _vix.index.tz_convert(df.index.tz)
        _vix_close  = _vix['close'].rename('vix_level')
        _vix_ma     = _vix['close'].rolling(30).mean()
        _vix_rel    = (_vix['close'] / (_vix_ma + 1e-9) - 1).rename('vix_vs_ma30')
        # 0=안정(<20) / 1=공포(20~30, 숏 선호) / 2=극공포(>30, 레버리지 위험)
        _vix_reg    = pd.cut(_vix['close'], bins=[0, 20, 30, 9999],
                             labels=[0, 1, 2]).astype(float).rename('vix_regime')
        df = df.join(_vix_close, how='left')
        df = df.join(_vix_rel,   how='left')
        df = df.join(_vix_reg,   how='left')
        df[['vix_level', 'vix_vs_ma30', 'vix_regime']] = \
            df[['vix_level', 'vix_vs_ma30', 'vix_regime']].ffill()
    else:
        df['vix_level']   = np.nan
        df['vix_vs_ma30'] = np.nan
        df['vix_regime']  = np.nan

    return df


# ─── [5-B] 레이블 함수 ────────────────────────────────────────────────────────
def make_label(df: pd.DataFrame, forward_bars: int) -> tuple[pd.Series, pd.Series]:
    """원래 N봉 종가 기반 레이블 (참고용)"""
    fwd_ret = df['close'].shift(-forward_bars) / df['close'] - 1
    label = (fwd_ret > RETURN_THRESH).astype(int)
    return label, fwd_ret


def make_triple_barrier_label(
    df: pd.DataFrame,
    forward_bars: int,
    stop_loss: float = -0.03,
    take_profit: float = 0.08,
) -> tuple[pd.Series, pd.Series]:
    """
    Triple Barrier Method (López de Prado, 2018)
    +1: 상단 배리어 먼저 터치 (익절) → 이진 레이블 = 1
    -1: 하단 배리어 먼저 터치 (손절) → 이진 레이블 = 0
     0: 시간 배리어 소진 (타임아웃)  → 이진 레이블 = 0
    """
    close = df['close'].values
    high  = df['high'].values
    low   = df['low'].values
    n     = len(close)
    labels = np.zeros(n, dtype=np.int8)

    for i in range(n - forward_bars - 1):
        entry = close[i]
        upper = entry * (1 + take_profit)
        lower = entry * (1 + stop_loss)
        tb_label = 0
        for j in range(1, forward_bars + 1):
            k = i + j
            if high[k] >= upper:
                tb_label = 1;  break
            if low[k]  <= lower:
                tb_label = -1; break
        labels[i] = int(tb_label == 1)

    fwd_ret = df['close'].shift(-forward_bars) / df['close'] - 1
    return pd.Series(labels, index=df.index), fwd_ret


# ─── [6] 데이터셋 빌드 ───────────────────────────────────────────────────────
def build_dataset(
    tickers: list[str],
    forward_bars: int,
    spy_df: pd.DataFrame | None = None,
    vix_df: pd.DataFrame | None = None,
    stop_loss: float = -0.03,
    take_profit: float = 0.08,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    frames = []
    ok = fail = 0

    for ticker in tickers:
        path = PRICE_DIR / f'{ticker}.parquet'
        if not path.exists():
            fail += 1
            continue
        try:
            df = pd.read_parquet(path)
            df = make_features(df, spy_df=spy_df, vix_df=vix_df)
            if TRIPLE_BARRIER:
                label, fwd_ret = make_triple_barrier_label(
                    df, forward_bars, stop_loss=stop_loss, take_profit=take_profit
                )
            else:
                label, fwd_ret = make_label(df, forward_bars)
            df['label']   = label
            df['fwd_ret'] = fwd_ret
            # SPY/VIX 컬럼은 없을 때 전부 NaN — LightGBM이 NaN 자체 처리하므로 dropna 대상 제외
            _spy_cols = {'spy_ret_1', 'spy_ret_20', 'spy_regime', 'rs_vs_spy_1', 'rs_vs_spy_20',
                         'vix_level', 'vix_vs_ma30', 'vix_regime'}
            _drop_check = [c for c in df.columns if c not in _spy_cols]
            df = df.dropna(subset=_drop_check)
            if len(df) < 200:
                fail += 1
                continue
            drop_cols = ['open', 'high', 'low', 'close', 'volume', 'label', 'fwd_ret',
                         'vwap', 'hv_node_price', 'orb_high', 'orb_low', 'vb_target']
            feat_cols = [c for c in df.columns if c not in drop_cols]
            df['_ticker'] = ticker
            frames.append(df[feat_cols + ['label', 'fwd_ret', '_ticker']])
            ok += 1
        except Exception as e:
            if fail == 0:
                logger.warning(f'  [{ticker}] 피처 계산 실패 (첫 오류): {e}')
            fail += 1

    logger.info(f'  로드 성공: {ok}, 실패/스킵: {fail}')
    if not frames:
        raise ValueError('데이터 없음')

    combined = pd.concat(frames).sort_index()
    feat_cols = [c for c in combined.columns if c not in ['label', 'fwd_ret', '_ticker']]
    X       = combined[feat_cols].astype(float)
    y       = combined['label']
    fwd_ret = combined['fwd_ret']
    return X, y, fwd_ret


# ─── [7] 백테스트 ────────────────────────────────────────────────────────────
def backtest_signals(
    fwd_ret_test: pd.Series,
    proba: np.ndarray,
    threshold: float,
    case_name: str,
    model_name: str,
):
    # 저장된 val-set optimal threshold 우선 사용
    thresh_path = MODEL_DIR / f'threshold_{case_name}.json'
    thresh_source = 'global'
    if thresh_path.exists():
        saved = json.loads(thresh_path.read_text())
        if model_name in saved:
            threshold = saved[model_name]
            thresh_source = 'val-optimal'

    signals = proba > threshold
    n_total  = len(proba)
    n_signal = int(signals.sum())

    if n_signal == 0:
        # 신호 없으면 상위 5% 분위로 자동 재시도
        auto_thresh = float(np.percentile(proba, 95))
        logger.warning(
            f'  [{model_name}] 신호 없음 (threshold={threshold:.4f}, 출처={thresh_source}) '
            f'→ 상위 5% 자동 재시도 (threshold={auto_thresh:.4f})'
        )
        threshold    = auto_thresh
        thresh_source = 'p95-auto'
        signals  = proba > threshold
        n_signal = int(signals.sum())

    if n_signal == 0:
        logger.warning(f'  [{model_name}] 재시도 후도 신호 없음')
        return

    sig_rets  = fwd_ret_test.values[signals]
    sig_dates = fwd_ret_test.index[signals]
    wins      = sig_rets[sig_rets > 0]
    losses    = sig_rets[sig_rets <= 0]

    win_rate      = len(wins) / n_signal
    profit_factor = wins.sum() / (abs(losses.sum()) + 1e-9)
    avg_ret       = sig_rets.mean()

    bars_per_year = 6.5 * 252
    sharpe = avg_ret / (sig_rets.std() + 1e-9) * (bars_per_year ** 0.5)

    # 복리 재투자 (매 거래마다 현재 잔고의 1% 베팅)
    START              = 10_000
    position_fraction  = 0.01
    sig_rets_c         = np.clip(sig_rets, -0.50, 2.0)
    equity_vals        = [START]
    for ret in sig_rets_c:
        bet = equity_vals[-1] * position_fraction
        equity_vals.append(equity_vals[-1] + bet * ret)
    equity     = pd.Series(equity_vals[1:])
    final_eq   = equity_vals[-1]
    total_ret  = final_eq / START - 1
    max_dd     = float((equity / equity.cummax() - 1).min())
    max_dd_usd = (equity / equity.cummax() - 1).min() * equity.cummax()[(equity / equity.cummax() - 1).idxmin()]

    # 테스트 기간
    test_start = sig_dates[0]
    test_end   = sig_dates[-1]
    try:
        duration_days = (test_end - test_start).days
    except Exception:
        duration_days = 0

    passed = (
        sharpe   >= BACKTEST_MIN_SHARPE  and
        max_dd   >= BACKTEST_MAX_DD      and
        win_rate >= BACKTEST_MIN_WINRATE and
        profit_factor >= BACKTEST_MIN_PF
    )
    status = '✓ PASS' if passed else '✗ FAIL'

    logger.info(f'  ── 백테스트 [{case_name} / {model_name}] {status} ──')
    logger.info(f'     threshold: {threshold:.4f}  ({thresh_source})')
    logger.info(f'     테스트 기간: {str(test_start)[:10]} ~ {str(test_end)[:10]}  ({duration_days}일)')
    logger.info(f'     시드: $10,000 → ${final_eq:,.0f}  ({total_ret:+.1%})')
    logger.info(f'     신호: {n_signal:,} / {n_total:,}봉  ({n_signal/n_total:.1%})')
    logger.info(f'     승률: {win_rate:.1%}  (기준 ≥ {BACKTEST_MIN_WINRATE:.0%})')
    logger.info(f'     평균수익: {avg_ret:.3%}')
    logger.info(f'     Profit Factor: {profit_factor:.2f}  (기준 ≥ {BACKTEST_MIN_PF})')
    logger.info(f'     Sharpe: {sharpe:.2f}  (기준 ≥ {BACKTEST_MIN_SHARPE})')
    logger.info(f'     Max Drawdown: ${max_dd_usd:,.0f}  ({max_dd:.1%})  (기준 ≥ {BACKTEST_MAX_DD:.0%})')

    fig, ax = plt.subplots(figsize=(10, 4))
    equity.plot(ax=ax, color='steelblue')
    ax.axhline(START, color='gray', linestyle='--', linewidth=0.8)
    ax.set_title(
        f'Equity Curve — {case_name} / {model_name}  {status}\n'
        f'${START:,} → ${final_eq:,.0f}  ({total_ret:+.1%})  |  {str(test_start)[:10]} ~ {str(test_end)[:10]}'
    )
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Portfolio Value ($)')
    plt.tight_layout()
    out = MODEL_DIR / f'equity_{case_name}_{model_name}.png'
    plt.savefig(out, dpi=100)
    plt.close(fig)
    logger.info(f'     에퀴티 커브 → {out.name}')

    record = {
        'timestamp':    datetime.now().isoformat(),
        'case':         case_name,
        'model':        model_name,
        'test_start':   str(test_start)[:10],
        'test_end':     str(test_end)[:10],
        'duration_days':duration_days,
        'final_equity': round(final_eq, 2),
        'total_return': round(total_ret, 4),
        'n_signals':    n_signal,
        'signal_rate':  round(n_signal / n_total, 4),
        'win_rate':     round(win_rate, 4),
        'avg_ret':      round(float(avg_ret), 6),
        'profit_factor':round(profit_factor, 4),
        'sharpe':       round(sharpe, 4),
        'max_drawdown': round(max_dd, 4),
        'passed':       bool(passed),
    }
    existing = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else []
    existing.append(record)
    RESULTS_PATH.write_text(json.dumps(existing, indent=2))


# ─── [8] 모델 학습 ───────────────────────────────────────────────────────────
def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    fwd_ret: pd.Series,
    case_name: str,
    run_backtest: bool = False,
    train_cutoff: str | None = None,
) -> dict:
    # train-cutoff: 리플레이 시작일 이후 데이터 학습/캘리브 제외 (look-ahead 방지)
    if train_cutoff:
        cutoff_ts = pd.Timestamp(train_cutoff, tz='UTC')
        mask = X.index < cutoff_ts
        X       = X[mask]
        y       = y[mask]
        fwd_ret = fwd_ret[mask]
        logger.info(f'  train-cutoff: {train_cutoff} 이전 데이터만 사용 ({len(X):,}행)')

    # 시계열 분할 (랜덤 split 금지)
    unique_dates = X.index.unique().sort_values()
    n = len(unique_dates)
    cut_train = unique_dates[int(n * 0.70)]
    cut_val   = unique_dates[int(n * 0.85)]

    X_train = X[X.index <  cut_train];  y_train = y[y.index <  cut_train]
    X_val   = X[(X.index >= cut_train) & (X.index < cut_val)]
    y_val   = y[(y.index >= cut_train) & (y.index < cut_val)]
    X_test  = X[X.index >= cut_val];   y_test  = y[y.index >= cut_val]
    fwd_ret_test = fwd_ret[fwd_ret.index >= cut_val]

    logger.info(f'  train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}')
    logger.info(f'  분할: ~{cut_train.date()} / ~{cut_val.date()}  pos={y_train.mean():.3f}')

    # ── LightGBM ─────────────────────────────────────────────────────────────
    lgb_params = {
        'objective':         'binary',
        'metric':            'binary_logloss',
        'boosting_type':     'gbdt',
        'num_leaves':        63,
        'learning_rate':     0.05,
        'feature_fraction':  0.8,
        'bagging_fraction':  0.8,
        'bagging_freq':      5,
        'is_unbalance':      True,
        'min_child_samples': 20,
        'verbose':           -1,
        'n_jobs':            -1,
    }

    dtrain = lgb.Dataset(X_train, label=y_train)
    dval   = lgb.Dataset(X_val,   label=y_val, reference=dtrain)

    lgb_model0 = lgb.train(
        lgb_params, dtrain,
        num_boost_round=1000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    fi = pd.Series(lgb_model0.feature_importance('gain'), index=X_train.columns)
    top_feats = fi.nlargest(TOP_FEATURES).index.tolist()

    dtrain2 = lgb.Dataset(X_train[top_feats], label=y_train)
    dval2   = lgb.Dataset(X_val[top_feats],   label=y_val, reference=dtrain2)
    lgb_model = lgb.train(
        lgb_params, dtrain2,
        num_boost_round=1000,
        valid_sets=[dval2],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    # LGB 캘리브레이션
    lgb_val_proba     = lgb_model.predict(X_val[top_feats])
    cal_lgb           = IsotonicRegression(out_of_bounds='clip').fit(lgb_val_proba, y_val)
    lgb_val_proba_cal = cal_lgb.predict(lgb_val_proba)
    proba_lgb         = cal_lgb.predict(lgb_model.predict(X_test[top_feats]))

    logger.info('  ── LightGBM (캘리브레이션 적용) ──')
    logger.info(f'     proba 범위: [{proba_lgb.min():.3f}, {proba_lgb.max():.3f}]  평균: {proba_lgb.mean():.3f}')
    print(classification_report(y_test, (proba_lgb > THRESHOLD).astype(int), digits=3))

    # ── XGBoost ──────────────────────────────────────────────────────────────
    scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum() + 1e-9)

    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        early_stopping_rounds=50,
        n_jobs=-1,
        verbosity=0,
        random_state=42,
    )
    xgb_model.fit(
        X_train[top_feats], y_train,
        eval_set=[(X_val[top_feats], y_val)],
        verbose=False,
    )

    # XGB 캘리브레이션
    xgb_val_proba     = xgb_model.predict_proba(X_val[top_feats])[:, 1]
    cal_xgb           = IsotonicRegression(out_of_bounds='clip').fit(xgb_val_proba, y_val)
    xgb_val_proba_cal = cal_xgb.predict(xgb_val_proba)
    proba_xgb         = cal_xgb.predict(xgb_model.predict_proba(X_test[top_feats])[:, 1])

    logger.info('  ── XGBoost (캘리브레이션 적용) ──')
    logger.info(f'     proba 범위: [{proba_xgb.min():.3f}, {proba_xgb.max():.3f}]  평균: {proba_xgb.mean():.3f}')
    print(classification_report(y_test, (proba_xgb > THRESHOLD).astype(int), digits=3))

    proba_ens = (proba_lgb + proba_xgb) / 2
    logger.info('  ── Ensemble (LGB+XGB 평균, 캘리브레이션 적용) ──')
    print(classification_report(y_test, (proba_ens > THRESHOLD).astype(int), digits=3))

    # ── Val set 기반 optimal threshold 계산 ─────────────────────────────────
    from sklearn.metrics import precision_recall_curve as _prc

    def _opt_thresh(val_proba_cal: np.ndarray, val_labels: np.ndarray,
                    min_precision: float = 0.60, min_recall: float = 0.05) -> float:
        prec, rec, threshs = _prc(val_labels, val_proba_cal)
        mask = (prec[:-1] >= min_precision) & (rec[:-1] >= min_recall)
        if mask.any():
            return float(threshs[mask][np.argmax(rec[:-1][mask])])
        return float(np.percentile(val_proba_cal, 90))

    opt_lgb = _opt_thresh(lgb_val_proba_cal, y_val.values)
    opt_xgb = _opt_thresh(xgb_val_proba_cal, y_val.values)
    opt_ens = _opt_thresh((lgb_val_proba_cal + xgb_val_proba_cal) / 2, y_val.values)
    thresh_dict = {'lgb': opt_lgb, 'xgb': opt_xgb, 'ensemble': opt_ens}
    thresh_path = MODEL_DIR / f'threshold_{case_name}.json'
    thresh_path.write_text(json.dumps(thresh_dict, indent=2))
    logger.info(f'  최적 threshold → lgb={opt_lgb:.4f}  xgb={opt_xgb:.4f}  ens={opt_ens:.4f}')

    # ── 저장 ─────────────────────────────────────────────────────────────────
    lgb_path      = MODEL_DIR / f'model_{case_name}_lgb.txt'
    xgb_path      = MODEL_DIR / f'model_{case_name}_xgb.pkl'
    feat_path     = MODEL_DIR / f'features_{case_name}.json'
    cal_lgb_path  = MODEL_DIR / f'calibrator_{case_name}_lgb.pkl'
    cal_xgb_path  = MODEL_DIR / f'calibrator_{case_name}_xgb.pkl'

    lgb_model.save_model(str(lgb_path))
    joblib.dump(xgb_model,  xgb_path)
    joblib.dump(cal_lgb,    cal_lgb_path)
    joblib.dump(cal_xgb,    cal_xgb_path)
    feat_path.write_text(json.dumps(top_feats))
    logger.success(
        f'  저장: {lgb_path.name}, {xgb_path.name}, '
        f'{cal_lgb_path.name}, {cal_xgb_path.name}'
    )

    _save_importance_plot(lgb_model, case_name)

    if run_backtest:
        backtest_signals(fwd_ret_test, proba_lgb,  THRESHOLD, case_name, 'lgb')
        backtest_signals(fwd_ret_test, proba_xgb,  THRESHOLD, case_name, 'xgb')
        backtest_signals(fwd_ret_test, proba_ens,  THRESHOLD, case_name, 'ensemble')

    return {'lgb': lgb_model, 'xgb': xgb_model, 'top_feats': top_feats,
            'cal_lgb': cal_lgb, 'cal_xgb': cal_xgb}


def _save_importance_plot(model: lgb.Booster, case_name: str):
    fi = pd.Series(
        model.feature_importance('gain'),
        index=model.feature_name(),
    ).nlargest(20)

    fig, ax = plt.subplots(figsize=(8, 5))
    fi[::-1].plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title(f'Feature Importance — {case_name}')
    ax.set_xlabel('Gain')
    plt.tight_layout()
    out = MODEL_DIR / f'feature_importance_{case_name}.png'
    plt.savefig(out, dpi=100)
    plt.close(fig)
    logger.info(f'  피처 중요도 → {out.name}')


# ─── [9] 케이스 실행 ─────────────────────────────────────────────────────────
def run_cases(
    cases: dict[str, list[str]],
    run_list: list[str],
    run_backtest: bool,
    spy_df: pd.DataFrame | None = None,
    vix_df: pd.DataFrame | None = None,
    stop_loss: float = -0.03,
    take_profit: float = 0.08,
    train_cutoff: str | None = None,
):
    trained = {}
    for case_name in run_list:
        forward_bars = FORWARD_BARS_MAP[case_name]
        tickers = cases.get(case_name, [])
        tickers = [t for t in tickers if (PRICE_DIR / f'{t}.parquet').exists()]
        logger.info(f'\n=== [{case_name}] {len(tickers):,}개 종목  (forward={forward_bars}봉) ===')

        if not tickers:
            logger.warning('  데이터 없음, 스킵')
            continue

        mem_gb = psutil.virtual_memory().available / 1e9
        safe_limit = int(mem_gb * 80)
        if len(tickers) > safe_limit:
            logger.warning(f'  메모리 부족 위험 → {safe_limit}개로 샘플링 (가용 {mem_gb:.1f}GB)')
            random.seed(42)
            tickers = random.sample(tickers, safe_limit)

        X, y, fwd_ret = build_dataset(
            tickers, forward_bars, spy_df=spy_df, vix_df=vix_df,
            stop_loss=stop_loss, take_profit=take_profit,
        )
        models = train_model(X, y, fwd_ret, case_name, run_backtest=run_backtest,
                             train_cutoff=train_cutoff)
        trained[case_name] = models
        del X, y, fwd_ret

    return trained


# ─── [10] 백테스트 전용 실행 (학습 없이 저장된 모델로) ──────────────────────
def run_backtest_only(
    cases: dict[str, list[str]],
    run_list: list[str],
    spy_df: pd.DataFrame | None = None,
    vix_df: pd.DataFrame | None = None,
    stop_loss: float = -0.03,
    take_profit: float = 0.08,
):
    for case_name in run_list:
        lgb_path  = MODEL_DIR / f'model_{case_name}_lgb.txt'
        xgb_path  = MODEL_DIR / f'model_{case_name}_xgb.pkl'
        feat_path = MODEL_DIR / f'features_{case_name}.json'
        cal_lgb_path = MODEL_DIR / f'calibrator_{case_name}_lgb.pkl'
        cal_xgb_path = MODEL_DIR / f'calibrator_{case_name}_xgb.pkl'

        if not lgb_path.exists() or not feat_path.exists():
            logger.warning(f'[{case_name}] 저장된 모델 없음, 스킵')
            continue

        forward_bars = FORWARD_BARS_MAP[case_name]
        lgb_model  = lgb.Booster(model_file=str(lgb_path))
        xgb_model  = joblib.load(xgb_path) if xgb_path.exists() else None
        top_feats  = json.loads(feat_path.read_text())
        cal_lgb    = joblib.load(cal_lgb_path) if cal_lgb_path.exists() else None
        cal_xgb    = joblib.load(cal_xgb_path) if cal_xgb_path.exists() else None

        tickers = [t for t in cases.get(case_name, []) if (PRICE_DIR / f'{t}.parquet').exists()]
        if len(tickers) > 200:
            random.seed(42)
            tickers = random.sample(tickers, 200)

        logger.info(f'\n=== 백테스트 [{case_name}] {len(tickers):,}개 (forward={forward_bars}봉) ===')
        X, y, fwd_ret = build_dataset(
            tickers, forward_bars, spy_df=spy_df, vix_df=vix_df,
            stop_loss=stop_loss, take_profit=take_profit,
        )

        unique_dates = X.index.unique().sort_values()
        cut_val = unique_dates[int(len(unique_dates) * 0.85)]
        X_test       = X[X.index >= cut_val]
        fwd_ret_test = fwd_ret[fwd_ret.index >= cut_val]

        raw_lgb   = lgb_model.predict(X_test[top_feats])
        proba_lgb = cal_lgb.predict(raw_lgb) if cal_lgb else raw_lgb
        backtest_signals(fwd_ret_test, proba_lgb, THRESHOLD, case_name, 'lgb')

        if xgb_model is not None:
            raw_xgb   = xgb_model.predict_proba(X_test[top_feats])[:, 1]
            proba_xgb = cal_xgb.predict(raw_xgb) if cal_xgb else raw_xgb
            backtest_signals(fwd_ret_test, proba_xgb, THRESHOLD, case_name, 'xgb')
            proba_ens = (proba_lgb + proba_xgb) / 2
            backtest_signals(fwd_ret_test, proba_ens, THRESHOLD, case_name, 'ensemble')

        del X, y, fwd_ret


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    global INTERVAL, PERIOD, RETURN_THRESH, FORWARD_BARS_MAP, PRICE_DIR, THRESHOLD, TRIPLE_BARRIER
    parser = argparse.ArgumentParser(description='Day Trading 로컬 학습')
    parser.add_argument(
        '--cases', nargs='+',
        choices=['smallcap', 'small_mid', 'mid_large', 'all', 'leverage_mid_large'],
        default=None,
    )
    parser.add_argument('--skip-download',        action='store_true')
    parser.add_argument('--download-only',        action='store_true')
    parser.add_argument('--download-financials',  action='store_true')
    parser.add_argument('--no-backtest',     action='store_true', help='백테스트 생략')
    parser.add_argument('--backtest-only',   action='store_true')
    parser.add_argument('--replay',            action='store_true')
    parser.add_argument('--portfolio-replay', action='store_true',
                        help='시간순 포트폴리오 시뮬 (자본 공유, 복리)')
    parser.add_argument('--replay-start',     default=None)
    parser.add_argument('--replay-end',       default=None)
    parser.add_argument('--replay-tickers',   nargs='+', default=None)
    parser.add_argument('--max-positions',    type=int, default=5,
                        help='포트폴리오 최대 동시 보유 종목 수 (기본 5)')
    parser.add_argument('--min-mktcap',       type=float, default=0.0)
    parser.add_argument('--max-tickers',     type=int,   default=0)
    parser.add_argument('--stop-loss',       type=float, default=None,
                        help='손절 비율 (미지정 시 인터벌 기본값 적용)')
    parser.add_argument('--take-profit',     type=float, default=None,
                        help='익절 비율 (미지정 시 인터벌 기본값 적용)')
    parser.add_argument('--commission',      type=float, default=0.0025,
                        help='편도 수수료 (기본: 0.0025 = 0.25%%, 한국투자증권 기준)')
    parser.add_argument('--position-fraction', type=float, default=0.01)
    parser.add_argument('--no-triple-barrier', action='store_true',
                        help='트리플 배리어 비활성화 → 원래 N봉 종가 레이블 사용')
    parser.add_argument('--exit-mode',       default='fixed', choices=['fixed', 'signal'])
    parser.add_argument('--exit-threshold',  type=float, default=0.5)
    parser.add_argument('--leverage-long-threshold',  type=float, default=0.75)
    parser.add_argument('--leverage-short-threshold', type=float, default=0.25)
    parser.add_argument('--leverage-replay', action='store_true',
                        help='레버리지 ETF 방향 시뮬 (전 페어 멀티포지션)')
    parser.add_argument('--download-leverage-etfs', action='store_true',
                        help='레버리지 ETF 전종목 1h 데이터 다운로드 (기존 주식과 동일 방식)')
    parser.add_argument('--train-cutoff', default=None,
                        help='이 날짜 이후 데이터 학습/캘리브 제외 (예: 2025-06-01). 리플레이 시작일 이전으로 설정')
    parser.add_argument('--circuit-breaker', type=float, default=0.15,
                        help='고점 대비 낙폭 임계값, 초과 시 신규 진입 중단 (기본 0.15=15%%, 0=비활성화)')
    parser.add_argument('--interval',   default=INTERVAL,
                        choices=['1m', '2m', '5m', '15m', '30m', '1h', '1d'])
    parser.add_argument('--threshold',  type=float, default=None,
                        help='신호 threshold (미지정 시 인터벌 기본값 적용)')
    args = parser.parse_args()

    if args.cases is None:
        args.cases = ['leverage_mid_large'] if args.leverage_replay else ['smallcap', 'small_mid', 'mid_large', 'all']

    if args.interval != INTERVAL:
        INTERVAL      = args.interval
        PERIOD        = PERIOD_MAP.get(INTERVAL, '2y')
        RETURN_THRESH = RETURN_THRESH_MAP.get(INTERVAL, 0.008)
        FORWARD_BARS_MAP = FORWARD_BARS_MAP_BY_INTERVAL.get(INTERVAL, FORWARD_BARS_MAP_BY_INTERVAL['1h'])
        PRICE_DIR = DATA_DIR / f'price_{INTERVAL}'
        PRICE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f'봉 단위: {INTERVAL}  기간: {PERIOD}  수익기준: {RETURN_THRESH*100:.1f}%')

    # 인터벌 기본값 적용 (사용자 미지정 시)
    THRESHOLD   = args.threshold   if args.threshold   is not None else THRESHOLD_MAP.get(INTERVAL, 0.35)
    stop_loss   = args.stop_loss   if args.stop_loss   is not None else STOP_LOSS_MAP.get(INTERVAL, -0.03)
    take_profit = args.take_profit if args.take_profit is not None else TAKE_PROFIT_MAP.get(INTERVAL, 0.08)
    logger.info(f'파라미터: stop={stop_loss:.3f}  target={take_profit:.3f}  threshold={THRESHOLD}')

    if args.no_triple_barrier:
        TRIPLE_BARRIER = False
        logger.info('레이블: N봉 종가 기반 (트리플 배리어 비활성화)')
    else:
        logger.info(f'레이블: Triple Barrier  (stop={stop_loss}, target={take_profit})')

    logger.info('=== Day Trading 로컬 학습 시작 ===')
    logger.info(f'저장 경로: {BASE_DIR}')
    mem_total = psutil.virtual_memory().total / 1e9
    logger.info(f'시스템 RAM: {mem_total:.1f} GB')

    if args.download_leverage_etfs:
        from backtest.leverage_replay import DEFAULT_ETF_PAIRS, ETF_PAIRS_FILE, save_default_etf_pairs
        etf_tickers = list(dict.fromkeys(e for pair in DEFAULT_ETF_PAIRS for e in pair))
        logger.info(f'레버리지 ETF {len(etf_tickers)}개 다운로드 시작...')
        download_prices(etf_tickers, skip=False)
        if not ETF_PAIRS_FILE.exists():
            save_default_etf_pairs()
        return

    all_tickers = fetch_universe()
    replay_mode = args.leverage_replay or args.portfolio_replay or args.replay
    if replay_mode and MC_CACHE.exists():
        market_caps = json.loads(MC_CACHE.read_text())
        logger.info(f'시총 캐시 로드 (리플레이 모드, 네트워크 스킵): {len(market_caps):,}개')
    else:
        market_caps = fetch_market_caps(all_tickers)
    cases       = classify_cases(market_caps)
    all_valid   = cases['all']

    if args.download_financials:
        fetch_all_financials(all_valid)
        return

    if args.portfolio_replay:
        from backtest.portfolio_replay import run_portfolio_replay

        if not args.replay_start:
            logger.error('--replay-start 날짜를 지정하세요. 예: --replay-start 2025-06-11')
            return

        spy_df           = fetch_spy_data()
        make_features_fn = partial(make_features, spy_df=spy_df if not spy_df.empty else None)

        for case_name in args.cases:
            tickers = cases.get(case_name, [])
            tickers = [t for t in tickers if (PRICE_DIR / f'{t}.parquet').exists()]
            if not tickers:
                logger.warning(f'[{case_name}] parquet 파일 없음')
                continue
            logger.info(f'\n=== 포트폴리오 Replay [{case_name}] {len(tickers):,}개 ===')
            run_portfolio_replay(
                tickers            = tickers,
                price_dir          = PRICE_DIR,
                case_name          = case_name,
                make_features_fn   = make_features_fn,
                start_date         = args.replay_start,
                end_date           = args.replay_end,
                start_capital      = 10_000,
                max_positions      = args.max_positions,
                stop_loss          = stop_loss,
                take_profit        = take_profit,
                forward_bars       = FORWARD_BARS_MAP[case_name],
                threshold          = args.threshold,
                commission         = args.commission,
                spy_regime_filter  = True,
                adx_filter         = 0,
                circuit_breaker    = args.circuit_breaker,
            )
        return

    if args.leverage_replay:
        from backtest.leverage_replay import run_leverage_replay

        if not args.replay_start:
            logger.error('--replay-start 날짜를 지정하세요. 예: --replay-start 2025-06-11')
            return

        spy_df           = fetch_spy_data()
        vix_df           = fetch_vix_data()
        make_features_fn = partial(
            make_features,
            spy_df=spy_df if not spy_df.empty else None,
            vix_df=vix_df if not vix_df.empty else None,
        )

        for case_name in args.cases:
            tickers = cases.get(case_name, [])
            tickers = [t for t in tickers if (PRICE_DIR / f'{t}.parquet').exists()]
            if not tickers:
                logger.warning(f'[{case_name}] parquet 파일 없음')
                continue
            logger.info(f'\n=== 레버리지 ETF Replay [{case_name}] {len(tickers):,}개 ===')
            run_leverage_replay(
                tickers          = tickers,
                price_dir        = PRICE_DIR,
                case_name        = case_name,
                make_features_fn = make_features_fn,
                start_date       = args.replay_start,
                end_date         = args.replay_end,
                start_capital    = 10_000,
                stop_loss        = stop_loss,
                take_profit      = take_profit,
                forward_bars     = FORWARD_BARS_MAP[case_name],
                commission       = args.commission,
                spy_regime_filter= True,
                circuit_breaker  = args.circuit_breaker,
            )
        return

    if args.replay:
        from backtest.replay import run_replay

        spy_df = fetch_spy_data()
        make_features_fn = partial(make_features, spy_df=spy_df if not spy_df.empty else None)

        min_usd = args.min_mktcap * 1e8 / 1_400 if args.min_mktcap > 0 else 0.0
        if min_usd > 0:
            logger.info(f'시총 필터: ≥ {args.min_mktcap:,.0f}억원 (≈ ${min_usd/1e6:.0f}M USD)')

        for case_name in args.cases:
            tickers = cases.get(case_name, [])
            if args.replay_tickers:
                tickers = [t for t in args.replay_tickers if (PRICE_DIR / f'{t}.parquet').exists()]
            if min_usd > 0:
                before = len(tickers)
                tickers = [t for t in tickers if market_caps.get(t, 0) >= min_usd]
                logger.info(f'  [{case_name}] 시총 필터: {before:,}개 → {len(tickers):,}개')
            if not args.replay_tickers and args.max_tickers > 0 and len(tickers) > args.max_tickers:
                random.seed(42)
                tickers = random.sample(tickers, args.max_tickers)
                logger.info(f'  [{case_name}] max-tickers 제한: {args.max_tickers}개 랜덤 샘플')
            run_replay(
                tickers                  = tickers,
                price_dir                = PRICE_DIR,
                case_name                = case_name,
                make_features_fn         = make_features_fn,
                threshold                = THRESHOLD,
                forward_bars             = FORWARD_BARS_MAP[case_name],
                stop_loss                = stop_loss,
                take_profit              = take_profit,
                start_date               = args.replay_start,
                end_date                 = args.replay_end,
                position_fraction        = args.position_fraction,
                commission               = args.commission,
                exit_mode                = args.exit_mode,
                exit_threshold           = args.exit_threshold,
                leverage_long_threshold  = args.leverage_long_threshold,
                leverage_short_threshold = args.leverage_short_threshold,
            )
        return

    if args.backtest_only:
        spy_df = fetch_spy_data()
        vix_df = fetch_vix_data()
        run_backtest_only(
            cases, args.cases,
            spy_df=spy_df if not spy_df.empty else None,
            vix_df=vix_df if not vix_df.empty else None,
            stop_loss=stop_loss, take_profit=take_profit,
        )
        return

    download_prices(all_valid, skip=args.skip_download)
    spy_df = fetch_spy_data()
    vix_df = fetch_vix_data()

    if args.download_only:
        logger.success(f'[{INTERVAL}] 다운로드 완료 → {PRICE_DIR}')
        return

    run_cases(
        cases, args.cases, run_backtest=not args.no_backtest,
        spy_df=spy_df if not spy_df.empty else None,
        vix_df=vix_df if not vix_df.empty else None,
        stop_loss=stop_loss, take_profit=take_profit,
        train_cutoff=args.train_cutoff,
    )

    logger.success('\n=== 완료 ===')
    for f in sorted(MODEL_DIR.glob('*.txt')) + sorted(MODEL_DIR.glob('*.pkl')):
        size_kb = f.stat().st_size / 1024
        logger.info(f'  {f.name:45s}  {size_kb:.1f} KB')

    cases_str = ' '.join(f'--cases {c}' for c in args.cases) if len(args.cases) == 1 else ''
    print('\n' + '─' * 60)
    print('학습 완료. 다음 명령어로 백테스트/신호 확인:')
    print()
    print('  백테스트 재실행 (학습 없이):')
    print(f'    python3 day_trading/local_trainer.py --skip-download --backtest-only {cases_str}')
    print()
    print('  실시간 신호 스캔 (1회):')
    for c in args.cases:
        print(f'    python3 day_trading/live_signal.py --once --case {c}')
    print()
    print('  과거 신호 재현 (replay):')
    for c in args.cases:
        print(f'    python3 day_trading/local_trainer.py --skip-download --replay --cases {c}')
    print('─' * 60)


if __name__ == '__main__':
    main()
