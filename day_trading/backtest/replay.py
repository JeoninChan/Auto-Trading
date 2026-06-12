"""
Replay 백테스트 — 특정 날짜부터 시간여행 시뮬

진입: 신호 발생 봉의 다음 봉 시가
청산: 손절(-3%) / 익절(+8%) / N봉 타임스톱 중 먼저 (fixed 모드)
      손절(-3%) / 신호 약화(proba < exit_threshold) / N봉 타임스톱 (signal 모드)
출력: 거래 로그 CSV

사용:
  python3 day_trading/local_trainer.py --skip-download --replay \
      --replay-start 2024-01-01 --threshold 0.6 --cases all

  # 신호 기반 익절
  python3 day_trading/local_trainer.py --skip-download --replay \
      --replay-start 2024-01-01 --cases mid_large \
      --exit-mode signal --exit-threshold 0.5

  # 레버리지 방향 결정
  python3 day_trading/local_trainer.py --skip-download --replay \
      --replay-start 2024-01-01 --cases leverage_mid_large \
      --leverage-long-threshold 0.75 --leverage-short-threshold 0.25
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from loguru import logger

BASE_DIR       = Path(__file__).parent.parent
MODEL_DIR      = BASE_DIR / 'models'
FINANCIALS_DIR = BASE_DIR / 'data' / 'financials'
OUT_DIR        = Path(__file__).parent / 'reports'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 레버리지 ETF 매핑 (기초자산 → 롱/숏 ETF)
LEVERAGE_PAIRS: dict[str, dict[str, str]] = {
    'NVDA': {'long': 'NVDL', 'short': 'NVDS'},
    'QQQ':  {'long': 'TQQQ', 'short': 'SQQQ'},
    'SPY':  {'long': 'SPXL', 'short': 'SPXS'},
    'AAPL': {'long': 'AAPL', 'short': 'AAPL'},
}


def replay_ticker(
    ticker: str,
    df_ohlcv: pd.DataFrame,
    model_lgb: lgb.Booster,
    model_xgb,
    top_feats: list[str],
    make_features_fn,
    threshold: float = 0.6,
    forward_bars: int = 3,
    stop_loss: float = -0.03,
    take_profit: float = 0.08,
    start_date: str | None = None,
    end_date: str | None = None,
    commission: float = 0.0,
    exit_mode: str = 'fixed',
    exit_threshold: float = 0.5,
    leverage_long_threshold: float = 0.75,
    leverage_short_threshold: float = 0.25,
    is_leverage_case: bool = False,
    cal_lgb=None,
    cal_xgb=None,
) -> pd.DataFrame:
    df = make_features_fn(df_ohlcv).dropna()

    if start_date:
        df = df[df.index >= pd.Timestamp(start_date, tz=df.index.tz)]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date, tz=df.index.tz)]

    if len(df) < forward_bars + 2:
        return pd.DataFrame()

    trades = []
    for i in range(len(df) - forward_bars - 1):
        row = df[top_feats].iloc[[i]]
        try:
            raw_lgb = float(model_lgb.predict(row)[0])
            p_lgb   = float(cal_lgb.predict([raw_lgb])[0]) if cal_lgb else raw_lgb
            if model_xgb:
                raw_xgb = float(model_xgb.predict_proba(row)[:, 1][0])
                p_xgb   = float(cal_xgb.predict([raw_xgb])[0]) if cal_xgb else raw_xgb
            else:
                p_xgb = p_lgb
        except Exception:
            continue
        proba = (p_lgb + p_xgb) / 2

        # 레버리지 케이스: 이중 threshold로 롱/숏 결정
        if is_leverage_case:
            if proba >= leverage_long_threshold:
                direction = 'long'
            elif proba <= leverage_short_threshold:
                direction = 'short'
            else:
                continue  # 신호 약하면 HOLD
        else:
            if proba < threshold:
                continue
            direction = 'long'

        entry_bar   = df.iloc[i + 1]
        entry_price = float(entry_bar['open'])
        entry_time  = entry_bar.name

        exit_price = exit_time = reason = None
        for j in range(1, forward_bars + 1):
            bar = df.iloc[i + 1 + j]
            high  = float(bar['high'])
            low   = float(bar['low'])
            close = float(bar['close'])

            if direction == 'long':
                ret = (close - entry_price) / entry_price
                stop_hit   = low  <= entry_price * (1 + stop_loss)
                target_hit = high >= entry_price * (1 + take_profit)
            else:  # short: 하락 = 수익
                ret = (entry_price - close) / entry_price
                stop_hit   = high >= entry_price * (1 - stop_loss)
                target_hit = low  <= entry_price * (1 - take_profit)

            if stop_hit:
                exit_p = entry_price * (1 + stop_loss) if direction == 'long' else entry_price * (1 - stop_loss)
                exit_price, exit_time, reason = exit_p, bar.name, 'stop'
                break

            if exit_mode == 'fixed' and target_hit:
                exit_p = entry_price * (1 + take_profit) if direction == 'long' else entry_price * (1 - take_profit)
                exit_price, exit_time, reason = exit_p, bar.name, 'target'
                break

            if exit_mode == 'signal':
                # 보유 중 proba 재계산 — 신호 약화 시 다음 봉 시가에 청산
                try:
                    curr_row   = df[top_feats].iloc[[i + 1 + j]]
                    cr_lgb_raw = float(model_lgb.predict(curr_row)[0])
                    cp_lgb     = float(cal_lgb.predict([cr_lgb_raw])[0]) if cal_lgb else cr_lgb_raw
                    if model_xgb:
                        cr_xgb_raw = float(model_xgb.predict_proba(curr_row)[:, 1][0])
                        cp_xgb     = float(cal_xgb.predict([cr_xgb_raw])[0]) if cal_xgb else cr_xgb_raw
                    else:
                        cp_xgb = cp_lgb
                    curr_proba = (cp_lgb + cp_xgb) / 2
                    if curr_proba < exit_threshold:
                        # 다음 봉 시가에 청산 (없으면 현재 close)
                        next_idx = i + 1 + j + 1
                        exit_p   = float(df.iloc[next_idx]['open']) if next_idx < len(df) else close
                        exit_price, exit_time, reason = exit_p, bar.name, 'signal_exit'
                        break
                except Exception:
                    pass

            if j == forward_bars:
                exit_price, exit_time, reason = close, bar.name, 'timeout'

        if exit_price is None:
            continue

        if direction == 'long':
            ret_pct = (exit_price - entry_price) / entry_price * 100
        else:
            ret_pct = (entry_price - exit_price) / entry_price * 100
        ret_pct -= commission * 100

        # 레버리지 ETF 매핑 표시
        leverage_etf = ''
        if is_leverage_case and ticker in LEVERAGE_PAIRS:
            leverage_etf = LEVERAGE_PAIRS[ticker][direction]

        trades.append({
            'ticker':        ticker,
            'direction':     direction,
            'leverage_etf':  leverage_etf,
            'entry_time':    str(entry_time),
            'exit_time':     str(exit_time),
            'entry':         round(entry_price, 4),
            'exit':          round(exit_price,  4),
            'ret_pct':       round(ret_pct,     3),
            'reason':        reason,
            'proba':         round(proba,        4),
            'win':           int(ret_pct > 0),
        })

    return pd.DataFrame(trades)


def run_replay(
    tickers: list[str],
    price_dir: Path,
    case_name: str,
    make_features_fn,
    threshold: float = 0.6,
    forward_bars: int = 3,
    stop_loss: float = -0.03,
    take_profit: float = 0.08,
    start_date: str | None = None,
    end_date: str | None = None,
    position_fraction: float = 0.01,
    commission: float = 0.0,
    exit_mode: str = 'fixed',
    exit_threshold: float = 0.5,
    leverage_long_threshold: float = 0.75,
    leverage_short_threshold: float = 0.25,
) -> None:
    lgb_path      = MODEL_DIR / f'model_{case_name}_lgb.txt'
    xgb_path      = MODEL_DIR / f'model_{case_name}_xgb.pkl'
    feat_path     = MODEL_DIR / f'features_{case_name}.json'
    cal_lgb_path  = MODEL_DIR / f'calibrator_{case_name}_lgb.pkl'
    cal_xgb_path  = MODEL_DIR / f'calibrator_{case_name}_xgb.pkl'

    if not lgb_path.exists() or not feat_path.exists():
        logger.error(f'모델 없음: {lgb_path}')
        return

    model_lgb = lgb.Booster(model_file=str(lgb_path))
    model_xgb = joblib.load(xgb_path) if xgb_path.exists() else None
    cal_lgb   = joblib.load(cal_lgb_path) if cal_lgb_path.exists() else None
    cal_xgb   = joblib.load(cal_xgb_path) if cal_xgb_path.exists() else None
    top_feats = json.loads(feat_path.read_text())
    if cal_lgb:
        logger.info(f'  캘리브레이터 로드: {cal_lgb_path.name}')
    else:
        logger.warning(f'  캘리브레이터 없음 ({cal_lgb_path.name}) — raw proba 사용')

    is_leverage_case = (case_name == 'leverage_mid_large')

    logger.info(
        f'=== Replay [{case_name}] {len(tickers):,}개  '
        f'start={start_date}  end={end_date}  threshold={threshold}  '
        f'exit={exit_mode}' +
        (f'  long≥{leverage_long_threshold}/short≤{leverage_short_threshold}' if is_leverage_case else '') +
        ' ==='
    )

    # ─── 체크포인트 설정 ─────────────────────────────────────────────────────────
    ts_key    = (start_date or 'all').replace('-', '')
    ckpt_csv  = OUT_DIR / f'replay_{case_name}_{ts_key}_checkpoint.csv'
    ckpt_done = OUT_DIR / f'replay_{case_name}_{ts_key}_done.json'

    done_set: set[str] = set(json.loads(ckpt_done.read_text())) if ckpt_done.exists() else set()
    if done_set:
        remaining = [t for t in tickers if t not in done_set]
        logger.info(f'체크포인트 재개: {len(done_set)}개 완료, {len(remaining)}개 남음')
        tickers = remaining

    SAVE_EVERY = 50
    batch: list[pd.DataFrame] = []

    for i, ticker in enumerate(tickers):
        path = price_dir / f'{ticker}.parquet'
        if not path.exists():
            done_set.add(ticker)
            continue
        try:
            df = pd.read_parquet(path)
            result = replay_ticker(
                ticker, df, model_lgb, model_xgb, top_feats, make_features_fn,
                threshold=threshold, forward_bars=forward_bars,
                stop_loss=stop_loss, take_profit=take_profit,
                start_date=start_date, end_date=end_date,
                commission=commission,
                exit_mode=exit_mode, exit_threshold=exit_threshold,
                leverage_long_threshold=leverage_long_threshold,
                leverage_short_threshold=leverage_short_threshold,
                is_leverage_case=is_leverage_case,
                cal_lgb=cal_lgb, cal_xgb=cal_xgb,
            )
            if not result.empty:
                batch.append(result)
        except Exception as e:
            logger.debug(f'{ticker} 실패: {e}')

        done_set.add(ticker)

        # 50개마다 즉시 디스크 저장
        if (i + 1) % SAVE_EVERY == 0 or i == len(tickers) - 1:
            if batch:
                df_b = pd.concat(batch, ignore_index=True)
                df_b.to_csv(ckpt_csv, mode='a',
                            header=not ckpt_csv.exists(), index=False,
                            encoding='utf-8-sig')
                batch = []
            ckpt_done.write_text(json.dumps(sorted(done_set)))

            # 현재 계좌 잔액 표시
            cur_equity = 10_000.0
            if ckpt_csv.exists():
                try:
                    df_cur = pd.read_csv(ckpt_csv)
                    if not df_cur.empty:
                        cur_equity = float(
                            10_000 * (1 + df_cur['ret_pct'] / 100 * position_fraction).cumprod().iloc[-1]
                        )
                except Exception:
                    pass

            pct = (i + 1) / len(tickers) * 100 if tickers else 100
            chg = (cur_equity / 10_000 - 1) * 100
            logger.info(
                f'  체크포인트: {i+1}/{len(tickers)} ({pct:.0f}%) — '
                f'현재 계좌: ${cur_equity:,.0f} ({chg:+.1f}%)'
            )

    # ─── 최종 요약 — checkpoint CSV에서 읽기 ────────────────────────────────────
    if not ckpt_csv.exists():
        logger.warning('Replay 결과: 신호 없음')
        return

    df_all = pd.read_csv(ckpt_csv)
    df_all = df_all.sort_values('entry_time').reset_index(drop=True)

    # equity_after 컬럼 추가
    df_all['equity_after'] = (
        10_000 * (1 + df_all['ret_pct'] / 100 * position_fraction).cumprod()
    ).round(2)

    total     = len(df_all)
    wins      = int(df_all['win'].sum())
    avg_ret   = df_all['ret_pct'].mean()
    max_dd    = _calc_max_dd(df_all['ret_pct'], position_fraction)
    by_reason = df_all['reason'].value_counts().to_dict()

    logger.info(f'\n{"─"*55}')
    logger.info(f'  총 거래  : {total:,}건')
    logger.info(f'  승률     : {wins/total*100:.1f}%  ({wins}승 {total-wins}패)')
    logger.info(f'  평균수익 : {avg_ret:+.3f}%')
    logger.info(f'  Max DD   : {max_dd:.1f}%')
    reasons_str = '  '.join(f'{k}={v}' for k, v in by_reason.items())
    logger.info(f'  청산 사유: {reasons_str}')
    if commission > 0:
        logger.info(f'  수수료   : {commission*100:.2f}%/거래 (왕복)')
    logger.info(f'{"─"*55}')

    # 레버리지 케이스: 방향별 요약
    if is_leverage_case and 'direction' in df_all.columns:
        for direction, grp in df_all.groupby('direction'):
            w = int(grp['win'].sum())
            logger.info(
                f'  [{direction.upper()}] {len(grp):,}건  승률{w/len(grp)*100:.1f}%  '
                f'평균{grp["ret_pct"].mean():+.3f}%'
            )

    # ─── 기간별 분석 ──────────────────────────────────────────────────────────────
    _print_period_analysis(df_all, position_fraction)

    # ─── 최종 CSV 저장 ────────────────────────────────────────────────────────────
    ts  = datetime.now().strftime('%Y%m%d_%H%M')
    out = OUT_DIR / f'replay_{case_name}_{ts}.csv'
    df_all.to_csv(out, index=False, encoding='utf-8-sig')
    logger.success(f'최종 결과 저장: {out}')
    logger.info(f'체크포인트: {ckpt_csv.name} / {ckpt_done.name}')


def _calc_max_dd(ret_pct_series: pd.Series, position_fraction: float = 0.01) -> float:
    equity = 10_000 * (1 + ret_pct_series / 100 * position_fraction).cumprod()
    dd = (equity / equity.cummax() - 1).min() * 100
    return float(dd)


def get_financials_at(ticker: str, date: pd.Timestamp,
                      fin_dir: Path = FINANCIALS_DIR) -> dict:
    """특정 날짜 기준 직전 분기 재무 스냅샷 반환 (point-in-time)."""
    path = fin_dir / f'{ticker}.parquet'
    if not path.exists():
        return {}
    try:
        df = pd.read_parquet(path)
        if df.index.tz is not None:
            cmp = date.tz_localize(None) if date.tz is None else date.tz_convert(None)
            past = df[df.index.tz_localize(None) <= cmp]
        else:
            cmp = date.tz_localize(None) if date.tz is not None else date
            past = df[df.index <= cmp]
        if past.empty:
            return {}
        row = past.iloc[-1]
        return {k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
    except Exception:
        return {}


def _print_period_analysis(df_trades: pd.DataFrame, position_fraction: float = 0.01) -> None:
    df = df_trades.copy()
    df['entry_dt'] = pd.to_datetime(df['entry_time'], utc=True)
    df = df.sort_values('entry_dt').reset_index(drop=True)

    # 복리 equity curve
    df['equity'] = 10_000 * (1 + df['ret_pct'] / 100 * position_fraction).cumprod()
    df['year']   = df['entry_dt'].dt.year
    df['month']  = df['entry_dt'].dt.to_period('M')
    df['qtr']    = df['entry_dt'].dt.to_period('Q')

    logger.info(f'\n{"─"*55}')
    logger.info(f'  [기간별 수익 분석]  복리 {position_fraction*100:.0f}%/거래, 초기 $10,000')

    # 연도별
    logger.info('  ■ 연도별')
    prev_eq = 10_000.0
    for yr, g in df.groupby('year'):
        end_eq   = float(g['equity'].iloc[-1])
        yr_ret   = (end_eq / prev_eq - 1) * 100
        win_rate = g['win'].mean() * 100
        logger.info(
            f'    {yr}  {len(g):4d}건  승률{win_rate:.0f}%  '
            f'복리수익{yr_ret:+.1f}%  (${prev_eq:,.0f}→${end_eq:,.0f})'
        )
        prev_eq = end_eq

    # 분기별
    logger.info('  ■ 분기별')
    prev_eq = 10_000.0
    for qtr, g in df.groupby('qtr'):
        end_eq   = float(g['equity'].iloc[-1])
        qtr_ret  = (end_eq / prev_eq - 1) * 100
        win_rate = g['win'].mean() * 100
        logger.info(f'    {qtr}  {len(g):4d}건  승률{win_rate:.0f}%  복리수익{qtr_ret:+.1f}%')
        prev_eq = end_eq

    # 월별 best / worst (복리)
    monthly = df.groupby('month').agg(
        count   =('ret_pct', 'count'),
        m_ret   =('ret_pct', lambda x: ((1 + x / 100 * position_fraction).prod() - 1) * 100),
        win_rate=('win', 'mean'),
    )
    best_m  = monthly['m_ret'].idxmax()
    worst_m = monthly['m_ret'].idxmin()
    logger.info('  ■ 월별 극값')
    logger.info(
        f'    최고달: {best_m}  {monthly.loc[best_m,"m_ret"]:+.2f}%'
        f'  ({int(monthly.loc[best_m,"count"])}건  승률{monthly.loc[best_m,"win_rate"]*100:.0f}%)'
    )
    logger.info(
        f'    최저달: {worst_m}  {monthly.loc[worst_m,"m_ret"]:+.2f}%'
        f'  ({int(monthly.loc[worst_m,"count"])}건  승률{monthly.loc[worst_m,"win_rate"]*100:.0f}%)'
    )

    # 누적 + CAGR
    final_eq = float(df['equity'].iloc[-1])
    t0       = df['entry_dt'].min()
    t1       = df['entry_dt'].max()
    years    = max((t1 - t0).days / 365.25, 0.05)
    cagr     = ((final_eq / 10_000) ** (1 / years) - 1) * 100
    logger.info('  ■ 누적')
    logger.info(f'    기간  : {t0.date()} ~ {t1.date()}  ({years:.1f}년)')
    logger.info(f'    최종  : ${final_eq:,.0f}  ({(final_eq/10_000-1)*100:+.1f}%)')
    logger.info(f'    CAGR  : {cagr:+.1f}%')
    logger.info(f'{"─"*55}')
