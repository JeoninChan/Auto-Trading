"""
Portfolio Replay — 시간순 포트폴리오 시뮬레이터

기존 replay.py와 달리 전 종목을 시간축으로 동시에 관리:
  - 매 봉: 오픈 포지션 청산 체크 → 빈 슬롯에 신호 진입
  - $10,000 공유 자본, 복리 재투자
  - 최대 N개 동시 보유 (기본 5개)

사용:
  python3 day_trading/local_trainer.py --skip-download --portfolio-replay \\
      --cases mid_large --replay-start 2025-06-11

  python3 day_trading/local_trainer.py --skip-download --portfolio-replay \\
      --cases mid_large --replay-start 2024-06-11 --max-positions 3
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger

BASE_DIR  = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / 'models'
OUT_DIR   = Path(__file__).parent / 'reports'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_portfolio_replay(
    tickers: list[str],
    price_dir: Path,
    case_name: str,
    make_features_fn,
    start_date: str,
    end_date: str | None = None,
    start_capital: float = 10_000,
    max_positions: int = 5,
    stop_loss: float = -0.03,
    take_profit: float = 0.08,
    forward_bars: int = 7,
    threshold: float | None = None,
    commission: float = 0.0025,        # 0.25% 편도 (한국투자증권 미국주식 기준)
    slippage:   float = 0.0005,        # 0.05% 편도 (S&P500 실측 4.5bps + 안전마진)
    min_dollar_volume: float = 100_000,    # 봉당 최소 $10만 거래대금
    spy_regime_filter: bool = True,    # True: SPY EMA20 < EMA60 구간 신규 진입 금지
    adx_filter: int = 0,               # 0=비활성화, >0이면 SPY ADX가 이 값 이하일 때 진입 차단
    circuit_breaker: float = 0.15,     # 고점 대비 -15% 낙폭 시 신규 진입 중단 (0=비활성화)
) -> None:

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

    if threshold is None and thresh_path.exists():
        saved     = json.loads(thresh_path.read_text())
        threshold = saved.get('ensemble', saved.get('lgb', 0.35))
        logger.info(f'threshold: {threshold:.4f}  (val-optimal, ensemble)')
    elif threshold is None:
        threshold = 0.35
        logger.info(f'threshold: {threshold:.4f}  (기본값)')

    # ── SPY 레짐 빌드 (EMA20 > EMA60 → 상승장=1, 하락장=0) ─────────────────────
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
            spy_regime_s = (spy_ema20 > spy_ema60).astype(int)
            spy_regime_map = spy_regime_s.to_dict()
            bull_pct = spy_regime_s.mean() * 100
            logger.info(f'SPY 레짐 로드: {len(spy_regime_map):,}봉  상승장 비율: {bull_pct:.1f}%')
        else:
            logger.warning('SPY.parquet 없음 — 레짐 필터 비활성화')

    # ── SPY ADX 추세 필터 빌드 ────────────────────────────────────────────────
    spy_adx_map: dict = {}
    if adx_filter > 0:
        spy_path = price_dir / 'SPY.parquet'
        if spy_path.exists():
            _spy = pd.read_parquet(spy_path)
            if isinstance(_spy.columns, pd.MultiIndex):
                _spy.columns = _spy.columns.get_level_values(0)
            _spy.columns = [c.lower() for c in _spy.columns]
            _h = _spy['high']; _l = _spy['low']; _c = _spy['close']
            _tr  = pd.concat([_h - _l, (_h - _c.shift()).abs(), (_l - _c.shift()).abs()], axis=1).max(axis=1)
            _atr = _tr.ewm(span=14, adjust=False).mean()
            _pdm = _h.diff().clip(lower=0).where(_h.diff() > (-_l.diff()), 0.0)
            _mdm = (-_l.diff()).clip(lower=0).where((-_l.diff()) > _h.diff(), 0.0)
            _pdi = 100 * _pdm.ewm(span=14, adjust=False).mean() / (_atr + 1e-9)
            _mdi = 100 * _mdm.ewm(span=14, adjust=False).mean() / (_atr + 1e-9)
            _dx  = 100 * (_pdi - _mdi).abs() / (_pdi + _mdi + 1e-9)
            _adx = _dx.ewm(span=14, adjust=False).mean()
            spy_adx_map  = _adx.to_dict()
            trending_pct = (_adx > adx_filter).mean() * 100
            logger.info(f'SPY ADX>{adx_filter} (추세장): {trending_pct:.1f}%  (횡보장 = 진입 차단)')

    # ── 전 종목 피처 로드 ──────────────────────────────────────────────────────
    logger.info(f'피처 로드 중... ({len(tickers):,}개 종목, 잠시 기다리세요)')
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
            # OHLCV + 피처 전부 보유 (open/high/low/close 청산가 계산용)
            keep_cols = list(dict.fromkeys(['open', 'high', 'low', 'close', 'volume'] + top_feats))
            keep_cols = [c for c in keep_cols if c in df.columns]
            ticker_data[ticker] = df[keep_cols].dropna(subset=top_feats)
        except Exception as e:
            fail += 1
            if fail <= 3:
                logger.debug(f'{ticker} 로드 실패: {e}')

        if (i + 1) % 500 == 0:
            logger.info(f'  {i+1:,}/{len(tickers):,} 로드...')

    logger.info(f'로드 완료: {len(ticker_data):,}개  실패/스킵: {fail}개')
    if not ticker_data:
        logger.error('사용 가능한 종목 없음')
        return

    # ── 타임스탬프 통합 정렬 ───────────────────────────────────────────────────
    all_ts_set: set = set()
    for df in ticker_data.values():
        all_ts_set.update(df.index.tolist())
    all_ts = sorted(all_ts_set)

    # 날짜 필터
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
        logger.error(f'해당 기간 데이터 없음: {start_date} ~ {end_date or "현재"}')
        return

    # 종목별 timestamp → iloc 인덱스 빠른 조회용
    ticker_ts_map: dict[str, dict] = {
        t: {ts: i for i, ts in enumerate(df.index)}
        for t, df in ticker_data.items()
    }

    logger.info(
        f'\n포트폴리오 시뮬 시작\n'
        f'  기간      : {str(start_ts)[:10]} ~ {str(all_ts[-1])[:10]}\n'
        f'  타임스탬프: {len(all_ts):,}개\n'
        f'  종목 수   : {len(ticker_data):,}개\n'
        f'  시드      : ${start_capital:,.0f}\n'
        f'  최대 포지션: {max_positions}개\n'
        f'  손절/익절 : {stop_loss*100:.1f}% / {take_profit*100:.1f}%\n'
        f'  threshold : {threshold:.4f}'
    )

    # ── 시뮬레이션 ─────────────────────────────────────────────────────────────
    equity      = start_capital
    equity_peak = start_capital
    cb_logged   = False
    # positions: ticker → {entry_price, entry_ts, bars_held, size, stop_p, target_p, proba}
    positions: dict[str, dict] = {}
    # pending: ticker → proba  (다음 봉 시가에 진입 대기)
    pending:   dict[str, float] = {}
    trades:    list[dict] = []

    # 배치 예측용 함수
    def _batch_predict(cand_tickers: list[str], ts) -> dict[str, float]:
        rows, valid = [], []
        for t in cand_tickers:
            ts_map = ticker_ts_map[t]
            if ts not in ts_map:
                continue
            idx = ts_map[ts]
            row = ticker_data[t].iloc[idx][top_feats].values
            rows.append(row)
            valid.append(t)
        if not rows:
            return {}
        X = np.array(rows, dtype=float)
        raw_lgb = model_lgb.predict(X)
        p_lgb   = cal_lgb.predict(raw_lgb) if cal_lgb else raw_lgb
        if model_xgb is not None:
            raw_xgb = model_xgb.predict_proba(X)[:, 1]
            p_xgb   = cal_xgb.predict(raw_xgb) if cal_xgb else raw_xgb
        else:
            p_xgb = p_lgb
        probas = (p_lgb + p_xgb) / 2
        return {t: float(probas[i]) for i, t in enumerate(valid)}

    log_every = max(1, len(all_ts) // 40)

    for step, ts in enumerate(all_ts):

        # ── 1. 대기 진입 실행 (전 봉 신호 → 이 봉 시가 매수) ─────────────────
        for ticker in list(pending.keys()):
            ts_map = ticker_ts_map.get(ticker, {})
            if ts not in ts_map:
                continue
            bar = ticker_data[ticker].iloc[ts_map[ts]]
            if 'open' not in ticker_data[ticker].columns:
                del pending[ticker]
                continue
            # 슬리피지: 시가보다 0.05% 더 비싸게 체결
            entry_price = float(bar['open']) * (1 + slippage)
            slot_size   = equity / max_positions
            positions[ticker] = {
                'entry_price':  entry_price,
                'entry_ts':     ts,
                'bars_held':    0,
                'size':         slot_size,
                'stop_p':       entry_price * (1 + stop_loss),
                'target_p':     entry_price * (1 + take_profit),
                'proba':        pending[ticker],
            }
            logger.info(
                f'[BUY]  {ticker:<8} @ ${entry_price:>8.2f} '
                f'| proba={pending[ticker]:.3f} '
                f'| 잔고: ${equity:,.0f} ({(equity/start_capital-1)*100:+.1f}%)'
            )
            del pending[ticker]

        # ── 2. 오픈 포지션 청산 체크 ─────────────────────────────────────────
        for ticker in list(positions.keys()):
            pos    = positions[ticker]
            ts_map = ticker_ts_map.get(ticker, {})
            if ts not in ts_map:
                pos['bars_held'] += 1
                continue

            bar   = ticker_data[ticker].iloc[ts_map[ts]]
            high  = float(bar.get('high',  bar['close']))
            low   = float(bar.get('low',   bar['close']))
            close = float(bar['close'])
            pos['bars_held'] += 1

            exit_price = reason = None
            if low <= pos['stop_p']:
                # 슬리피지: 손절가보다 더 불리하게 체결
                exit_price, reason = pos['stop_p'] * (1 - slippage), 'stop'
            elif high >= pos['target_p']:
                exit_price, reason = pos['target_p'] * (1 - slippage), 'target'
            elif pos['bars_held'] >= forward_bars:
                exit_price, reason = close * (1 - slippage), 'timeout'

            if exit_price is not None:
                ret_pct  = (exit_price - pos['entry_price']) / pos['entry_price'] * 100
                ret_pct -= commission * 2 * 100  # 왕복 수수료 (매수 + 매도)
                pnl      = pos['size'] * ret_pct / 100
                equity  += pnl
                reason_tag = {'stop': '손절', 'target': '익절', 'timeout': '타임아웃'}.get(reason, reason)
                win_mark   = '✓' if ret_pct > 0 else '✗'
                logger.info(
                    f'[SELL] {ticker:<8} @ ${exit_price:>8.2f} '
                    f'| {win_mark} {ret_pct:>+6.2f}%  PnL: ${pnl:>+7.2f} '
                    f'| {reason_tag:<5} '
                    f'| 잔고: ${equity:,.0f} ({(equity/start_capital-1)*100:+.1f}%)'
                )
                trades.append({
                    'ticker':       ticker,
                    'entry_time':   str(pos['entry_ts']),
                    'exit_time':    str(ts),
                    'entry':        round(pos['entry_price'], 4),
                    'exit':         round(exit_price, 4),
                    'ret_pct':      round(ret_pct, 3),
                    'pnl':          round(pnl, 2),
                    'reason':       reason,
                    'proba':        round(pos['proba'], 4),
                    'win':          int(ret_pct > 0),
                    'equity_after': round(equity, 2),
                })
                del positions[ticker]

        # ── 3. 새 신호 생성 (배치 예측) ─────────────────────────────────────
        equity_peak = max(equity_peak, equity)
        cb_triggered = circuit_breaker > 0 and (equity / equity_peak - 1) < -circuit_breaker
        if cb_triggered and not cb_logged:
            logger.warning(
                f'[서킷브레이커] 고점 대비 {(equity/equity_peak-1)*100:.1f}% 낙폭 → 신규 진입 중단'
            )
            cb_logged = True
        elif not cb_triggered:
            cb_logged = False

        free_slots = max_positions - len(positions) - len(pending)
        # SPY 레짐 체크: 하락장(EMA20 < EMA60)이면 신규 진입 전면 차단
        if spy_regime_filter and spy_regime_map:
            if spy_regime_map.get(ts, 1) == 0:
                free_slots = 0
        # ADX 추세 필터
        if adx_filter > 0 and spy_adx_map:
            if spy_adx_map.get(ts, 0) <= adx_filter:
                free_slots = 0
        # 서킷브레이커
        if cb_triggered:
            free_slots = 0
        if free_slots > 0:
            occupied = set(positions) | set(pending)
            # 유동성 필터: 봉당 최소 $10만 거래대금인 종목만 후보에 올림
            cands = []
            for t in ticker_data:
                if t in occupied:
                    continue
                ts_map_t = ticker_ts_map.get(t, {})
                if ts not in ts_map_t:
                    continue
                bar_t = ticker_data[t].iloc[ts_map_t[ts]]
                vol   = float(bar_t.get('volume', 0) if hasattr(bar_t, 'get') else bar_t['volume'] if 'volume' in bar_t.index else 0)
                price = float(bar_t['close'])
                if vol * price >= min_dollar_volume:
                    cands.append(t)

            proba_map = _batch_predict(cands, ts)
            above = [(t, p) for t, p in proba_map.items() if p >= threshold]
            above.sort(key=lambda x: x[1], reverse=True)
            for ticker, proba in above[:free_slots]:
                pending[ticker] = proba

        # ── 4. 진행 로그 ─────────────────────────────────────────────────────
        if (step + 1) % log_every == 0 or step == len(all_ts) - 1:
            pct = (step + 1) / len(all_ts) * 100
            chg = (equity / start_capital - 1) * 100
            logger.info(
                f'  [{pct:5.1f}%] {str(ts)[:16]}  '
                f'잔고: ${equity:,.0f} ({chg:+.1f}%)  '
                f'포지션: {len(positions)}개  거래: {len(trades):,}건'
            )

    # ── 요약 출력 ──────────────────────────────────────────────────────────────
    if not trades:
        logger.warning('거래 없음 — threshold 낮추거나 기간 확인')
        return

    df_t  = pd.DataFrame(trades).sort_values('entry_time').reset_index(drop=True)
    total = len(df_t)
    wins  = int(df_t['win'].sum())
    avg_r = df_t['ret_pct'].mean()
    eq_s  = pd.Series([start_capital] + df_t['equity_after'].tolist())
    mdd   = float((eq_s / eq_s.cummax() - 1).min() * 100)
    final = equity
    by_r  = df_t['reason'].value_counts().to_dict()

    dur   = (all_ts[-1] - start_ts).days

    logger.info(f'\n{"─"*60}')
    logger.info(f'  기간      : {str(start_ts)[:10]} ~ {str(all_ts[-1])[:10]}  ({dur}일)')
    logger.info(f'  시드      : ${start_capital:,.0f} → ${final:,.0f}  ({(final/start_capital-1)*100:+.1f}%)')
    logger.info(f'  수수료    : 편도 {commission*100:.2f}% × 왕복 = {commission*2*100:.2f}%  (한국투자증권 기준)')
    logger.info(f'  슬리피지  : 편도 {slippage*100:.3f}%')
    logger.info(f'  유동성 필터: 봉당 ${min_dollar_volume:,.0f} 이상')
    logger.info(f'  총 거래   : {total:,}건')
    logger.info(f'  승률      : {wins/total*100:.1f}%  ({wins}승 {total-wins}패)')
    logger.info(f'  평균수익  : {avg_r:+.3f}%  per trade')
    logger.info(f'  Max DD    : {mdd:.1f}%')
    reasons_str = '  '.join(f'{k}={v:,}' for k, v in by_r.items())
    logger.info(f'  청산사유  : {reasons_str}')
    logger.info(f'{"─"*60}')

    # ── 월별 수익 ─────────────────────────────────────────────────────────────
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

    # ── 저장 ──────────────────────────────────────────────────────────────────
    ts_str  = datetime.now().strftime('%Y%m%d_%H%M')
    out_csv = OUT_DIR / f'portfolio_{case_name}_{ts_str}.csv'
    df_t.to_csv(out_csv, index=False, encoding='utf-8-sig')
    logger.success(f'거래 로그 → {out_csv.name}')
    logger.success(f'경로: {out_csv}')
