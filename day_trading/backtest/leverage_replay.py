"""
Leverage ETF Replay — 종목별 개별 proba 기반 레버리지 ETF 멀티포지션 시뮬레이터

각 ETF 페어의 기초주 개별 proba로 방향 판단 (NVDL→NVDA proba, TSLL→TSLA proba 등).
기초주 없는 페어는 mid+large 전체 median proba 사용 (fallback).
VIX 구간별 허용 방향: <20=롱숏모두 / 20~30=숏전용 / >30=전차단

사용:
  python3 day_trading/local_trainer.py --skip-download --leverage-replay \\
      --cases leverage_mid_large --replay-start 2025-06-11
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

BASE_DIR       = Path(__file__).parent.parent
MODEL_DIR      = BASE_DIR / 'models'
OUT_DIR        = Path(__file__).parent / 'reports'
ETF_PAIRS_FILE = BASE_DIR / 'data' / 'etf_pairs.json'
OUT_DIR.mkdir(parents=True, exist_ok=True)

PERIOD_1H       = '2y'
DELIST_GAP_DAYS = 30

# 전수 조사 완료 (2026-06-11 기준 전부 ACTIVE)
DEFAULT_ETF_PAIRS: list[tuple[str, str]] = [
    # ── MEGA CAP TECH (확인됨) ──────────────────────────────────────────────
    ('NVDL', 'NVDS'),   # NVDA  2x / -1x  Direxion
    ('TSLL', 'TSLS'),   # TSLA  2x / -1x  ProShares
    ('AAPU', 'AAPD'),   # AAPL  2x / -1x  Direxion
    ('AMZU', 'AMZD'),   # AMZN  2x / -1x  Direxion
    ('GGLL', 'GGLS'),   # GOOGL 2x / -1x  Direxion
    ('METU', 'METD'),   # META  2x / -1x  Direxion
    ('MSFO', 'MSFD'),   # MSFT  2x / -1x  Direxion
    # ── CRYPTO / HIGH BETA (확인됨) ────────────────────────────────────────
    ('CONL', 'CONS'),   # COIN  2x / -1x  Direxion
    ('MSTU', 'MSTD'),   # MSTR  2x / -1x  Direxion
    # ── SEMI / TECH (데이터 없으면 _check_etf_coverage 자동 스킵) ──────────
    ('AMDU', 'AMDD'),   # AMD   2x / -1x  Direxion
    ('NFXL', 'NFXS'),   # NFLX  2x / -1x
    ('PLTU', 'PLTD'),   # PLTR  2x / -1x
    ('SMCU', 'SMCD'),   # SMCI  2x / -1x
    ('RDTL', 'RDTS'),   # RDDT  2x / -1x
    ('ARML', 'ARMS'),   # ARM   2x / -1x
    ('UONH', 'UOND'),   # UNH   2x / -1x
]

UNDERLYING_MAP: dict[str, str] = {
    'NVDL': 'NVDA', 'NVDS': 'NVDA',
    'TSLL': 'TSLA', 'TSLS': 'TSLA',
    'AAPU': 'AAPL', 'AAPD': 'AAPL',
    'AMZU': 'AMZN', 'AMZD': 'AMZN',
    'GGLL': 'GOOGL', 'GGLS': 'GOOGL',
    'METU': 'META',  'METD': 'META',
    'MSFO': 'MSFT',  'MSFD': 'MSFT',
    'CONL': 'COIN',  'CONS': 'COIN',
    'MSTU': 'MSTR',  'MSTD': 'MSTR',
    'AMDU': 'AMD',   'AMDD': 'AMD',
    'NFXL': 'NFLX',  'NFXS': 'NFLX',
    'PLTU': 'PLTR',  'PLTD': 'PLTR',
    'SMCU': 'SMCI',  'SMCD': 'SMCI',
    'RDTL': 'RDDT',  'RDTS': 'RDDT',
    'ARML': 'ARM',   'ARMS': 'ARM',
    'UONH': 'UNH',   'UOND': 'UNH',
}


def save_default_etf_pairs() -> None:
    """DEFAULT_ETF_PAIRS를 etf_pairs.json으로 저장 (최초 1회)."""
    data = []
    for l, s in DEFAULT_ETF_PAIRS:
        und = UNDERLYING_MAP.get(l, '')
        data.append({
            'underlying':    und,
            'long_lev':      l,    'long_lev_mult':  2,
            'long_1x':       None,
            'short_lev':     None, 'short_lev_mult': None,
            'short_1x':      s,
        })
    ETF_PAIRS_FILE.parent.mkdir(parents=True, exist_ok=True)
    ETF_PAIRS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    logger.info(f'ETF 페어 파일 생성: {ETF_PAIRS_FILE}  ({len(data)}개 종목)')


def load_etf_registry() -> list[dict]:
    """etf_pairs.json 로드. 없으면 빈 리스트 반환."""
    if ETF_PAIRS_FILE.exists():
        return json.loads(ETF_PAIRS_FILE.read_text())
    return []


def build_pairs_from_registry(
    registry: list[dict],
    etf_data: dict,
) -> tuple[list[tuple[str, str]], list[str], list[str], dict[str, str]]:
    """
    registry → (full_pairs, long_only, short_only, und_map)
    - full_pairs : 롱+숏 ETF 둘 다 데이터 있는 페어
    - long_only  : 롱 ETF만 있음 (상승장 전용)
    - short_only : 숏 ETF만 있음 (하락장 전용)
    - und_map    : {etf_ticker: underlying}
    롱 진입: long_lev 우선, 없으면 long_1x / 숏 진입: short_lev 우선, 없으면 short_1x
    """
    full_pairs, long_only, short_only = [], [], []
    und_map: dict[str, str] = {}

    for r in registry:
        und = r.get('underlying', '')
        l   = r.get('long_lev')  or r.get('long_1x')
        s   = r.get('short_lev') or r.get('short_1x')

        if l: und_map[l] = und
        if s: und_map[s] = und

        l_ok = l in etf_data if l else False
        s_ok = s in etf_data if s else False

        if l_ok and s_ok:
            full_pairs.append((l, s))
        elif l_ok:
            long_only.append(l)
        elif s_ok:
            short_only.append(s)

    return full_pairs, long_only, short_only, und_map


def _check_etf_coverage(
    name: str,
    df: pd.DataFrame,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    sim_ts_list: list,
) -> dict:
    """ETF 상장일·상폐 여부·커버리지 비율 검증. 결과 dict 반환."""
    first = df.index[0]
    last  = df.index[-1]

    if hasattr(first, 'tz') and first.tz is None and start_ts.tz is not None:
        first = first.tz_localize(start_ts.tz)
        last  = last.tz_localize(start_ts.tz)

    if first > start_ts:
        gap = (first - start_ts).days
        logger.warning(
            f'  ⚠  {name} 상장일: {str(first)[:10]}  →  시뮬 시작보다 {gap}일 늦음 → 그 이전 진입 불가'
        )
    else:
        logger.info(f'  ✓  {name} 상장: {str(first)[:10]}')

    gap_end = (end_ts - last).days
    if gap_end > DELIST_GAP_DAYS:
        logger.warning(
            f'  ⚠  {name} 마지막 데이터: {str(last)[:10]}  ({gap_end}일 갭) → 상폐 또는 누락 가능'
        )

    etf_ts_set = set(df.index)
    covered = sum(1 for t in sim_ts_list if t in etf_ts_set)
    pct     = covered / max(len(sim_ts_list), 1) * 100
    logger.info(f'     커버리지: {covered:,}/{len(sim_ts_list):,}봉 ({pct:.1f}%)')

    return {'first': first, 'last': last, 'covered': covered, 'pct': pct}


def _download_etf(ticker: str, price_dir: Path, period: str = PERIOD_1H) -> pd.DataFrame:
    path = price_dir / f'{ticker}.parquet'
    if path.exists():
        df = pd.read_parquet(path)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        return df

    logger.info(f'{ticker} 다운로드 중...')
    try:
        raw = yf.download(ticker, period=period, interval='1h',
                          auto_adjust=True, progress=False)
        if raw.empty:
            logger.warning(f'{ticker} 데이터 없음 — 스킵')
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.columns = [c.lower() for c in raw.columns]
        raw.to_parquet(path)
        logger.info(f'{ticker} 저장: {len(raw):,}봉')
        return raw
    except Exception as e:
        logger.warning(f'{ticker} 다운로드 실패: {e} — 스킵')
        return pd.DataFrame()


def run_leverage_replay(
    tickers: list[str],
    price_dir: Path,
    case_name: str,
    make_features_fn,
    start_date: str,
    end_date: str | None = None,
    start_capital: float = 10_000,
    position_fraction: float = 0.08,   # 포지션당 자본 8% (5포지션 최대 40%)
    max_positions: int = 5,            # 최대 동시 포지션 수
    etf_pairs: list[tuple[str, str]] | None = None,  # None → DEFAULT_ETF_PAIRS
    long_thresh: float = 0.60,         # rolling P90으로 덮어씀 (fallback 용)
    short_thresh: float = 0.40,
    circuit_breaker: float = 0.15,     # 고점 대비 -15% 시 신규 진입 중단 (0=비활성화)
    stop_loss: float = -0.03,
    take_profit: float = 0.08,
    forward_bars: int = 8,
    commission: float = 0.0025,
    slippage: float = 0.0005,
    spy_regime_filter: bool = True,
) -> None:

    # etf_pairs.json 있으면 파일 기준, 없으면 DEFAULT_ETF_PAIRS 기준
    _registry = load_etf_registry()
    _use_registry = bool(_registry) and etf_pairs is None

    # ── 모델 로드 ──────────────────────────────────────────────────────────────
    lgb_path     = MODEL_DIR / f'model_{case_name}_lgb.txt'
    xgb_path     = MODEL_DIR / f'model_{case_name}_xgb.pkl'
    feat_path    = MODEL_DIR / f'features_{case_name}.json'
    cal_lgb_path = MODEL_DIR / f'calibrator_{case_name}_lgb.pkl'
    cal_xgb_path = MODEL_DIR / f'calibrator_{case_name}_xgb.pkl'
    thresh_path  = MODEL_DIR / f'threshold_{case_name}.json'

    if not lgb_path.exists() or not feat_path.exists():
        logger.error(f'모델 없음: {lgb_path}  먼저 학습하세요.')
        return

    model_lgb = lgb.Booster(model_file=str(lgb_path))
    model_xgb = joblib.load(xgb_path) if xgb_path.exists() else None
    cal_lgb   = joblib.load(cal_lgb_path) if cal_lgb_path.exists() else None
    cal_xgb   = joblib.load(cal_xgb_path) if cal_xgb_path.exists() else None
    top_feats = json.loads(feat_path.read_text())

    base_threshold = 0.35
    if thresh_path.exists():
        saved = json.loads(thresh_path.read_text())
        base_threshold = saved.get('ensemble', saved.get('lgb', 0.35))
        logger.info(f'base threshold: {base_threshold:.4f}')

    # rolling P90/P10 없이 학습된 threshold 고정 사용 (모델 판단 직접 반영)
    long_thresh  = max(base_threshold, 0.60)
    short_thresh = min(1.0 - base_threshold, 0.40)
    logger.info(f'진입 threshold — 롱: {long_thresh:.4f}  숏: {short_thresh:.4f}')

    # ── ETF 전종목 로드 (parquet 캐시 우선, 없으면 다운로드) ────────────────────
    if _use_registry:
        _all_etf_tickers = []
        for r in _registry:
            for k in ('long_lev', 'long_1x', 'short_lev', 'short_1x'):
                if r.get(k):
                    _all_etf_tickers.append(r[k])
        _all_etf_tickers = list(dict.fromkeys(_all_etf_tickers))
    else:
        _input_pairs    = etf_pairs or DEFAULT_ETF_PAIRS
        _all_etf_tickers = list(dict.fromkeys(e for pair in _input_pairs for e in pair))

    logger.info(f'ETF 로드: {len(_all_etf_tickers)}개 티커...')
    etf_data: dict[str, pd.DataFrame] = {}
    for name in _all_etf_tickers:
        df = _download_etf(name, price_dir)
        if not df.empty:
            etf_data[name] = df

    # 페어/롱전용/숏전용 분류
    if _use_registry:
        full_pairs, long_only, short_only, und_map = build_pairs_from_registry(_registry, etf_data)
    else:
        _input_pairs = etf_pairs or DEFAULT_ETF_PAIRS
        full_pairs, long_only, short_only = [], [], []
        und_map = dict(UNDERLYING_MAP)
        for l, s in _input_pairs:
            l_ok = l in etf_data
            s_ok = s in etf_data
            if l_ok and s_ok:
                full_pairs.append((l, s))
            elif l_ok:
                long_only.append(l)
            elif s_ok:
                short_only.append(s)

    etf_pairs = full_pairs
    if not etf_pairs and not long_only and not short_only:
        logger.error('사용 가능한 ETF 없음')
        return
    logger.info(
        f'ETF 로드 완료: {len(etf_data)}개  |  '
        f'완전페어 {len(full_pairs)}개 / 롱전용 {len(long_only)}개 / 숏전용 {len(short_only)}개'
    )

    # ETF별 ATR(14)/price 비율 사전계산 → 동적 포지션 크기 조정
    etf_atr_ratio: dict[str, float] = {}
    for _name, _etf_df in etf_data.items():
        if {'high', 'low', 'close'}.issubset(_etf_df.columns):
            _tr = pd.concat([
                _etf_df['high'] - _etf_df['low'],
                (_etf_df['high'] - _etf_df['close'].shift()).abs(),
                (_etf_df['low']  - _etf_df['close'].shift()).abs(),
            ], axis=1).max(axis=1)
            _atr   = _tr.ewm(span=14, adjust=False).mean().iloc[-1]
            _price = _etf_df['close'].iloc[-1]
            etf_atr_ratio[_name] = float(_atr / _price) if _price > 0 else 0.02

    # ── SPY 레짐 빌드 ──────────────────────────────────────────────────────────
    spy_regime_map: dict = {}
    if spy_regime_filter:
        spy_path = price_dir / 'SPY.parquet'
        if spy_path.exists():
            spy_df = pd.read_parquet(spy_path)
            if isinstance(spy_df.columns, pd.MultiIndex):
                spy_df.columns = spy_df.columns.get_level_values(0)
            spy_df.columns = [c.lower() for c in spy_df.columns]
            spy_ema20 = spy_df['close'].ewm(span=20).mean()
            spy_ema60 = spy_df['close'].ewm(span=60).mean()
            spy_regime_map = (spy_ema20 > spy_ema60).astype(int).to_dict()
            bull_pct = sum(spy_regime_map.values()) / max(len(spy_regime_map), 1) * 100
            logger.info(f'SPY 레짐: 상승장 {bull_pct:.1f}%')

    # ── SPY ADX 추세 필터 (leverage_replay 전용) ──────────────────────────────
    spy_adx_map: dict = {}
    spy_path = price_dir / 'SPY.parquet'
    if spy_path.exists():
        spy_df_adx = pd.read_parquet(spy_path)
        if isinstance(spy_df_adx.columns, pd.MultiIndex):
            spy_df_adx.columns = spy_df_adx.columns.get_level_values(0)
        spy_df_adx.columns = [c.lower() for c in spy_df_adx.columns]
        _h = spy_df_adx['high']; _l = spy_df_adx['low']; _c = spy_df_adx['close']
        _tr = pd.concat([_h - _l,
                         (_h - _c.shift()).abs(),
                         (_l - _c.shift()).abs()], axis=1).max(axis=1)
        _atr14    = _tr.ewm(span=14, adjust=False).mean()
        _plus_dm  = _h.diff().clip(lower=0).where(_h.diff() > (-_l.diff()), 0.0)
        _minus_dm = (-_l.diff()).clip(lower=0).where((-_l.diff()) > _h.diff(), 0.0)
        _plus_di  = 100 * _plus_dm.ewm(span=14, adjust=False).mean()  / (_atr14 + 1e-9)
        _minus_di = 100 * _minus_dm.ewm(span=14, adjust=False).mean() / (_atr14 + 1e-9)
        _dx       = 100 * (_plus_di - _minus_di).abs() / (_plus_di + _minus_di + 1e-9)
        _adx      = _dx.ewm(span=14, adjust=False).mean()
        spy_adx_map  = _adx.to_dict()
        trending_pct = (_adx > 35).mean() * 100
        logger.info(f'SPY ADX>35 (추세장): {trending_pct:.1f}%  (횡보장 = 진입 차단)')

    # ── VIX 구간별 방향 허용 맵 ───────────────────────────────────────────────
    # <20: 롱/숏 모두 허용 / 20~30: 숏만 (공포장=하락=숏이 맞음) / >30: 전차단 (레버리지 붕괴)
    vix_map: dict = {}
    vix_path = price_dir / 'VIX.parquet'
    if not vix_path.exists():
        try:
            _raw_vix = yf.download('^VIX', period=PERIOD_1H, interval='1h',
                                   auto_adjust=True, progress=False)
            if not _raw_vix.empty:
                if isinstance(_raw_vix.columns, pd.MultiIndex):
                    _raw_vix.columns = _raw_vix.columns.get_level_values(0)
                _raw_vix.columns = [c.lower() for c in _raw_vix.columns]
                _raw_vix.to_parquet(vix_path)
        except Exception as _e:
            logger.warning(f'VIX 다운로드 실패: {_e}')
    if vix_path.exists():
        _vix_df = pd.read_parquet(vix_path)
        if isinstance(_vix_df.columns, pd.MultiIndex):
            _vix_df.columns = _vix_df.columns.get_level_values(0)
        _vix_df.columns = [c.lower() for c in _vix_df.columns]
        vix_map = _vix_df['close'].to_dict()
        logger.info(
            f'VIX 구간 분포: '
            f'<20(롱OK) {(_vix_df["close"] < 20).mean()*100:.1f}%  '
            f'20~30(숏전용) {((_vix_df["close"] >= 20) & (_vix_df["close"] < 30)).mean()*100:.1f}%  '
            f'>30(전차단) {(_vix_df["close"] >= 30).mean()*100:.1f}%'
        )

    # ── 개별종목 피처 로드 ─────────────────────────────────────────────────────
    logger.info(f'방향신호 종목 피처 로드... ({len(tickers):,}개)')
    ticker_data: dict[str, pd.DataFrame] = {}
    fail = 0
    for i, ticker in enumerate(tickers):
        path = price_dir / f'{ticker}.parquet'
        if not path.exists():
            fail += 1
            continue
        try:
            raw = pd.read_parquet(path)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw.columns = [c.lower() for c in raw.columns]
            df = make_features_fn(raw)
            missing = [f for f in top_feats if f not in df.columns]
            if missing or df.empty:
                fail += 1
                continue
            keep = [c for c in list(dict.fromkeys(top_feats)) if c in df.columns]
            ticker_data[ticker] = df[keep].dropna(subset=top_feats)
        except Exception:
            fail += 1
        if (i + 1) % 500 == 0:
            logger.info(f'  {i+1:,}/{len(tickers):,} 로드...')

    logger.info(f'로드 완료: {len(ticker_data):,}개  실패: {fail}개')
    if not ticker_data:
        logger.error('종목 데이터 없음')
        return

    # ── 타임스탬프 통합 ────────────────────────────────────────────────────────
    all_ts_set: set = set()
    for df in ticker_data.values():
        all_ts_set.update(df.index.tolist())
    all_ts = sorted(all_ts_set)

    sample_ts = all_ts[0]
    tz = sample_ts.tz if hasattr(sample_ts, 'tz') else None

    def _to_ts(date_str: str) -> pd.Timestamp:
        t = pd.Timestamp(date_str)
        if tz is not None and t.tz is None:
            t = t.tz_localize(tz)
        return t

    start_ts = _to_ts(start_date)
    end_ts   = _to_ts(end_date) if end_date else None
    all_ts   = [t for t in all_ts if t >= start_ts and (end_ts is None or t <= end_ts)]

    if not all_ts:
        logger.error(f'해당 기간 데이터 없음: {start_date}')
        return

    ticker_ts_map = {t: {ts: i for i, ts in enumerate(df.index)}
                     for t, df in ticker_data.items()}

    # ETF timestamp 인덱스
    etf_ts_map: dict[str, dict] = {}
    for name, df in etf_data.items():
        if isinstance(df.index, pd.DatetimeIndex) and tz is not None:
            if df.index.tz is None:
                df.index = df.index.tz_localize(tz)
            elif df.index.tz != tz:
                df.index = df.index.tz_convert(tz)
        etf_ts_map[name] = {ts: i for i, ts in enumerate(df.index)}
        etf_data[name] = df

    # ── ETF 상장/상폐 커버리지 검증 ───────────────────────────────────────────
    effective_end_ts = end_ts if end_ts else all_ts[-1]
    logger.info('\nETF 커버리지 검증:')
    etf_coverage: dict[str, dict] = {}
    for etf_name, etf_df in etf_data.items():
        etf_coverage[etf_name] = _check_etf_coverage(
            etf_name, etf_df, start_ts, effective_end_ts, all_ts
        )

    # ── 배치 예측 → median proba ───────────────────────────────────────────────
    def _get_market_proba(ts) -> float | None:
        rows = []
        for t, df in ticker_data.items():
            ts_map = ticker_ts_map[t]
            if ts not in ts_map:
                continue
            row = df.iloc[ts_map[ts]][top_feats].values
            if np.isnan(row).any():
                continue
            rows.append(row)
        if len(rows) < 10:
            return None
        X = np.array(rows, dtype=float)
        raw_lgb = model_lgb.predict(X)
        p_lgb   = cal_lgb.predict(raw_lgb) if cal_lgb else raw_lgb
        if model_xgb is not None:
            raw_xgb = model_xgb.predict_proba(X)[:, 1]
            p_xgb   = cal_xgb.predict(raw_xgb) if cal_xgb else raw_xgb
        else:
            p_xgb = p_lgb
        return float(np.median((p_lgb + p_xgb) / 2))

    # ── ETF 봉 조회 헬퍼 ──────────────────────────────────────────────────────
    def _get_etf_bar(etf_name: str, ts):
        ts_map = etf_ts_map.get(etf_name, {})
        if ts not in ts_map:
            return None
        return etf_data[etf_name].iloc[ts_map[ts]]

    # 단일 종목 proba (UNDERLYING_MAP 기초주 전용)
    def _get_single_ticker_proba(ticker: str, ts) -> float | None:
        ts_map = ticker_ts_map.get(ticker, {})
        if ts not in ts_map:
            return None
        row = ticker_data[ticker].iloc[ts_map[ts]][top_feats].values
        if np.isnan(row).any():
            return None
        X = row.reshape(1, -1).astype(float)
        raw_lgb = model_lgb.predict(X)
        p_lgb   = cal_lgb.predict(raw_lgb) if cal_lgb else raw_lgb
        if model_xgb is not None:
            raw_xgb = model_xgb.predict_proba(X)[:, 1]
            p_xgb   = cal_xgb.predict(raw_xgb) if cal_xgb else raw_xgb
        else:
            p_xgb = p_lgb
        return float((p_lgb[0] + p_xgb[0]) / 2)

    # ── 사전 계산: 전 기간 market proba 캐싱 + auto-threshold ──────────────────
    logger.info(f'사전 계산: market proba 캐싱 중... ({len(all_ts):,}봉)')
    proba_cache: dict = {}
    for _ts in all_ts:
        _mp = _get_market_proba(_ts)
        if _mp is not None:
            proba_cache[_ts] = _mp

    # 기초주별 개별 proba 캐싱 (und_map 기초주 중 ticker_data에 있는 것만)
    _und_tickers = {v for v in und_map.values() if v in ticker_data}
    underlying_proba_cache: dict[str, dict] = {t: {} for t in _und_tickers}
    logger.info(f'종목별 proba 캐싱 중... ({len(underlying_proba_cache)}개 본주)')
    for _t in underlying_proba_cache:
        for _ts in all_ts:
            _p = _get_single_ticker_proba(_t, _ts)
            if _p is not None:
                underlying_proba_cache[_t][_ts] = _p

    cached_vals = list(proba_cache.values())
    if cached_vals:
        logger.info(
            f'proba 분포  '
            f'min={min(cached_vals):.4f}  median={np.median(cached_vals):.4f}  '
            f'max={max(cached_vals):.4f}  (rolling P90/P10 사용 — look-ahead 없음)'
        )
    else:
        logger.warning('market proba 캐싱 실패 — fallback threshold 사용')

    logger.info(
        f'\n레버리지 ETF 시뮬 시작\n'
        f'  기간       : {str(start_ts)[:10]} ~ {str(all_ts[-1])[:10]}\n'
        f'  타임스탬프 : {len(all_ts):,}개\n'
        f'  방향신호   : {len(ticker_data):,}개 종목 median proba\n'
        f'  유효 페어  : {len(etf_pairs)}개 (롱+숏 {len(etf_pairs)*2}종목)\n'
        f'  롱 임계값  : proba >= {long_thresh:.4f}\n'
        f'  숏 임계값  : proba <= {short_thresh:.4f}\n'
        f'  손절/익절  : {stop_loss*100:.1f}% / {take_profit*100:.1f}%\n'
        f'  포지션당   : {position_fraction*100:.0f}%  최대 {max_positions}개\n'
        f'  시드       : ${start_capital:,.0f}'
    )

    # ── 시뮬레이션 ─────────────────────────────────────────────────────────────
    equity                      = start_capital
    positions: dict[str, dict]  = {}   # ETF명 → 포지션 dict
    pendings:  list[dict]       = []   # 다음봉 진입 대기
    trades:    list[dict]       = []
    etf_missing_entry           = 0
    equity_peak                 = start_capital
    cb_logged                   = False

    log_every = max(1, len(all_ts) // 40)

    for step, ts in enumerate(all_ts):

        # ── 1. 대기 진입 실행 ─────────────────────────────────────────────────
        next_pendings = []
        for p in pendings:
            etf_name = p['etf']
            if etf_name in positions:
                continue
            bar = _get_etf_bar(etf_name, ts)
            if bar is not None:
                entry_price  = float(bar['open']) * (1 + slippage)
                _TARGET_VOL  = 0.02
                _etf_atr     = etf_atr_ratio.get(etf_name, _TARGET_VOL)
                _adj_frac    = position_fraction * min(1.0, _TARGET_VOL / max(_etf_atr, 0.001))
                positions[etf_name] = {
                    'etf':          etf_name,
                    'direction':    p['direction'],
                    'entry_price':  entry_price,
                    'entry_ts':     ts,
                    'bars_held':    0,
                    'size':         equity * _adj_frac,
                    'stop_p':       entry_price * (1 + stop_loss),
                    'target_p':     entry_price * (1 + take_profit),
                    'market_proba': p['proba'],
                }
                logger.info(
                    f'[BUY]  {etf_name:<8} @ ${entry_price:>8.2f} '
                    f'| proba={p["proba"]:.3f} '
                    f'| 잔고: ${equity:,.0f} ({(equity/start_capital-1)*100:+.1f}%)'
                )
            else:
                etf_missing_entry += 1
        pendings = next_pendings  # 1봉 대기만 허용 → 전부 소진

        # ── 2. 포지션 청산 체크 ───────────────────────────────────────────────
        to_close: list[str] = []
        for etf_name, pos in positions.items():
            bar = _get_etf_bar(etf_name, ts)
            if bar is None:
                continue
            high  = float(bar.get('high',  bar['close']))
            low   = float(bar.get('low',   bar['close']))
            close = float(bar['close'])
            pos['bars_held'] += 1

            exit_price = reason = None
            if low <= pos['stop_p']:
                exit_price, reason = pos['stop_p'] * (1 - slippage), 'stop'
            elif high >= pos['target_p']:
                exit_price, reason = pos['target_p'] * (1 - slippage), 'target'
            elif pos['bars_held'] >= forward_bars:
                exit_price, reason = close * (1 - slippage), 'timeout'

            if exit_price is not None:
                ret_pct = (exit_price - pos['entry_price']) / pos['entry_price'] * 100
                ret_pct -= commission * 2 * 100
                pnl      = pos['size'] * ret_pct / 100
                equity  += pnl
                reason_tag = {'stop': '손절', 'target': '익절', 'timeout': '타임아웃'}.get(reason, reason)
                win_mark   = '✓' if ret_pct > 0 else '✗'
                logger.info(
                    f'[SELL] {etf_name:<8} @ ${exit_price:>8.2f} '
                    f'| {win_mark} {ret_pct:>+6.2f}%  PnL: ${pnl:>+7.2f} '
                    f'| {reason_tag:<5} '
                    f'| 잔고: ${equity:,.0f} ({(equity/start_capital-1)*100:+.1f}%)'
                )
                trades.append({
                    'etf':          etf_name,
                    'direction':    pos['direction'],
                    'entry_time':   str(pos['entry_ts']),
                    'exit_time':    str(ts),
                    'entry':        round(pos['entry_price'], 4),
                    'exit':         round(exit_price, 4),
                    'ret_pct':      round(ret_pct, 3),
                    'pnl':          round(pnl, 2),
                    'reason':       reason,
                    'market_proba': round(pos['market_proba'], 4),
                    'win':          int(ret_pct > 0),
                    'equity_after': round(equity, 2),
                })
                to_close.append(etf_name)
        for etf_name in to_close:
            del positions[etf_name]

        # ── 3. 새 방향 신호 ───────────────────────────────────────────────────
        equity_peak = max(equity_peak, equity)
        cb_triggered = circuit_breaker > 0 and (equity / equity_peak - 1) < -circuit_breaker
        if cb_triggered and not cb_logged:
            logger.warning(
                f'[서킷브레이커] 고점 대비 {(equity/equity_peak-1)*100:.1f}% 낙폭 → 신규 진입 중단'
            )
            cb_logged = True
        elif not cb_triggered:
            cb_logged = False

        n_open = len(positions) + len(pendings)
        if n_open < max_positions and not cb_triggered:
            spy_bull    = spy_regime_map.get(ts, 1)
            # is_trending = spy_adx_map.get(ts, 0) > 25  # ADX>25 (이전)
            is_trending = spy_adx_map.get(ts, 0) > 30   # ADX>30 (중간)
            # is_trending = spy_adx_map.get(ts, 0) > 35  # ADX>35 (강화)
            market_p    = proba_cache.get(ts)

            # VIX 구간별 허용 방향 결정
            _vix_now     = vix_map.get(ts, 15.0)
            long_allowed  = _vix_now < 20            # VIX<20만 롱 허용
            short_allowed = _vix_now < 30            # VIX<30까지 숏 허용 (공포장 = 숏 ↑)

            if market_p is not None and is_trending:
                pending_etfs = {p['etf'] for p in pendings}

                # 완전페어 (롱+숏 둘 다)
                for long_e, short_e in etf_pairs:
                    if n_open >= max_positions:
                        break
                    _und = und_map.get(long_e) or und_map.get(short_e)
                    pair_p = underlying_proba_cache.get(_und, {}).get(ts, market_p) if _und else market_p
                    if pair_p is None:
                        continue
                    if pair_p >= long_thresh and spy_bull == 1 and long_allowed:
                        if long_e not in positions and long_e not in pending_etfs:
                            pendings.append({'etf': long_e, 'direction': 'LONG', 'proba': pair_p})
                            pending_etfs.add(long_e); n_open += 1
                    elif pair_p <= short_thresh and spy_bull == 0 and short_allowed:
                        if short_e not in positions and short_e not in pending_etfs:
                            pendings.append({'etf': short_e, 'direction': 'SHORT', 'proba': pair_p})
                            pending_etfs.add(short_e); n_open += 1

                # 롱전용 ETF — 상승장+VIX<20 시만 진입
                for long_e in long_only:
                    if n_open >= max_positions:
                        break
                    _und   = und_map.get(long_e)
                    pair_p = underlying_proba_cache.get(_und, {}).get(ts, market_p) if _und else market_p
                    if pair_p is None:
                        continue
                    if pair_p >= long_thresh and spy_bull == 1 and long_allowed:
                        if long_e not in positions and long_e not in pending_etfs:
                            pendings.append({'etf': long_e, 'direction': 'LONG', 'proba': pair_p})
                            pending_etfs.add(long_e); n_open += 1

                # 숏전용 ETF — 하락장+VIX<30 시만 진입
                for short_e in short_only:
                    if n_open >= max_positions:
                        break
                    _und   = und_map.get(short_e)
                    pair_p = underlying_proba_cache.get(_und, {}).get(ts, market_p) if _und else market_p
                    if pair_p is None:
                        continue
                    if pair_p <= short_thresh and spy_bull == 0 and short_allowed:
                        if short_e not in positions and short_e not in pending_etfs:
                            pendings.append({'etf': short_e, 'direction': 'SHORT', 'proba': pair_p})
                            pending_etfs.add(short_e); n_open += 1

        # ── 4. 진행 로그 ─────────────────────────────────────────────────────
        if (step + 1) % log_every == 0 or step == len(all_ts) - 1:
            pct = (step + 1) / len(all_ts) * 100
            chg = (equity / start_capital - 1) * 100
            pos_str = ', '.join(
                f'{k}({v["direction"][0]})' for k, v in positions.items()
            ) or '없음'
            logger.info(
                f'  [{pct:5.1f}%] {str(ts)[:16]}  '
                f'잔고: ${equity:,.0f} ({chg:+.1f}%)  '
                f'포지션[{len(positions)}]: {pos_str}  거래: {len(trades):,}건'
            )

    # ── 요약 ──────────────────────────────────────────────────────────────────
    if not trades:
        logger.warning('거래 없음 — threshold/ADX 조건 확인')
        return

    df_t  = pd.DataFrame(trades).sort_values('entry_time').reset_index(drop=True)
    total = len(df_t)
    wins  = int(df_t['win'].sum())
    avg_r = df_t['ret_pct'].mean()
    eq_s  = pd.Series([start_capital] + df_t['equity_after'].tolist())
    mdd   = float((eq_s / eq_s.cummax() - 1).min() * 100)
    final = equity
    by_r  = df_t['reason'].value_counts().to_dict()
    by_d  = df_t['direction'].value_counts().to_dict()
    dur   = (all_ts[-1] - start_ts).days

    logger.info(f'\n{"─"*65}')
    logger.info(f'  기간       : {str(start_ts)[:10]} ~ {str(all_ts[-1])[:10]}  ({dur}일)')
    logger.info(f'  시드       : ${start_capital:,.0f} → ${final:,.0f}  ({(final/start_capital-1)*100:+.1f}%)')
    logger.info(f'  수수료     : 편도 {commission*100:.2f}% × 왕복 = {commission*2*100:.2f}%')
    logger.info(f'  슬리피지   : 편도 {slippage*100:.3f}%')
    if etf_missing_entry:
        logger.warning(f'  ⚠ 진입 불가  : {etf_missing_entry:,}봉 (상장 전/상폐 후)')
    logger.info(f'  총 거래    : {total:,}건  (롱={by_d.get("LONG",0)}  숏={by_d.get("SHORT",0)})')
    logger.info(f'  승률       : {wins/total*100:.1f}%  ({wins}승 {total-wins}패)')
    logger.info(f'  평균수익   : {avg_r:+.3f}%  per trade')
    logger.info(f'  Max DD     : {mdd:.1f}%')
    reasons_str = '  '.join(f'{k}={v:,}' for k, v in by_r.items())
    logger.info(f'  청산사유   : {reasons_str}')
    logger.info(f'{"─"*65}')

    # ETF별 요약
    by_etf = df_t.groupby('etf').agg(
        trades=('win', 'count'),
        wins=('win', 'sum'),
        avg_ret=('ret_pct', 'mean'),
        total_pnl=('pnl', 'sum'),
    ).reset_index()
    logger.info('\n  ETF별 요약:')
    for _, row in by_etf.sort_values('total_pnl', ascending=False).iterrows():
        wr = row['wins'] / row['trades'] * 100 if row['trades'] > 0 else 0
        logger.info(
            f'    {row["etf"]:<6}  {row["trades"]:>4}건  '
            f'승률{wr:4.0f}%  평균{row["avg_ret"]:+.3f}%  PnL${row["total_pnl"]:+,.0f}'
        )

    # 월별 요약
    df_t['entry_dt'] = pd.to_datetime(df_t['entry_time'], utc=True)
    df_t['month']    = df_t['entry_dt'].dt.to_period('M')
    monthly = df_t.groupby('month').agg(
        trades=('win', 'count'),
        wins=('win', 'sum'),
        avg_ret=('ret_pct', 'mean'),
    ).reset_index()
    logger.info('\n  월별 요약:')
    for _, row in monthly.iterrows():
        wr = row['wins'] / row['trades'] * 100 if row['trades'] > 0 else 0
        logger.info(
            f'    {row["month"]}  거래{row["trades"]:>4}건  '
            f'승률{wr:4.0f}%  평균{row["avg_ret"]:+.3f}%'
        )

    # 저장
    ts_str  = datetime.now().strftime('%Y%m%d_%H%M')
    out_csv = OUT_DIR / f'leverage_{case_name}_{ts_str}.csv'
    df_t.to_csv(out_csv, index=False, encoding='utf-8-sig')
    logger.success(f'거래 로그 → {out_csv.name}')
