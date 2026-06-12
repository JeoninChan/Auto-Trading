"""
프로젝트 컨텍스트 내보내기 — 제미나이/GPT/Grok 토론용

사용:
  python3 day_trading/export_context.py
  python3 day_trading/export_context.py --question "트리플 배리어 상단 8% vs ATR 기반 동적 배리어 비교"
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

BASE_DIR  = Path(__file__).parent
MODEL_DIR = BASE_DIR / 'models'
OUT_DIR   = BASE_DIR / 'context_exports'
OUT_DIR.mkdir(exist_ok=True)

SYSTEM_OVERVIEW = """
## 프로젝트 개요
- 목적: NASDAQ 주식 자동매매 (단타 모델)
- 실행 환경: MacBook M4 (학습), Lenovo Legion 5 (자동매매)
- 언어: Python, LightGBM + XGBoost 앙상블

## 모델 구조
- 이진 분류: "이 자리에서 상단 배리어(익절 목표)를 먼저 터치할 확률"
- 레이블: Triple Barrier Method (López de Prado, 2018)
  - +1: 익절 배리어 먼저 터치 → 학습 레이블 1
  - -1/0: 손절 or 타임아웃 → 학습 레이블 0
- 캘리브레이션: IsotonicRegression (val set으로 proba 눈금 보정)
- 앙상블: (LGB proba + XGB proba) / 2

## 케이스 분류
| 케이스 | 종목 범위 | forward(1h) | 용도 |
|--------|----------|-------------|------|
| smallcap | mc < $1.4B | 3봉 | 잡주 단타 |
| small_mid | mc < $17B | 5봉 | 잡주+중형 |
| mid_large | mc ≥ $1.4B | 7봉 | 중형+우량 |
| all | 전체 | 3봉 | 범용 |
| leverage_mid_large | mc ≥ $1.4B | 8봉 | 레버리지 ETF 방향 |

## 피처 세트
- 제거: RSI, MACD, Stochastic (후행 지표)
- 유지: ATR, Bollinger Bands, ADX, EMA(9/21/50), OBV
- 추가:
  - SPY 상대 강도 (rs_vs_spy_1, rs_vs_spy_20, spy_regime)
  - ORB (Opening Range Breakout): orb_break_up, orb_break_down, orb_vol_ratio
  - VWAP 거리 (vwap_dist)
  - 매물대 노드 (hv_node_above, hv_node_dist) — 240봉 최대 거래량 가격

## 청산 전략
- 손절: -3% (config 조정 가능)
- 익절: +8% (고정, 신호 기반 익절은 백테스트 과적합 위험으로 미사용)
- 타임아웃: forward_bars 소진
- 레버리지 케이스: proba ≥ 0.75 → 롱 ETF (NVDL/TQQQ), proba ≤ 0.25 → 숏 ETF (NVDS/SQQQ)

## 검증 기준 (미통과 시 실거래 금지)
- Sharpe > 1.5, Max DD > -20%, Win Rate > 52%, Profit Factor > 1.5
"""


def get_model_status() -> str:
    lines = ['## 현재 모델 파일 상태']
    cases = ['smallcap', 'small_mid', 'mid_large', 'all', 'leverage_mid_large']
    for case in cases:
        lgb_ok = (MODEL_DIR / f'model_{case}_lgb.txt').exists()
        xgb_ok = (MODEL_DIR / f'model_{case}_xgb.pkl').exists()
        cal_ok = (MODEL_DIR / f'calibrator_{case}_lgb.pkl').exists()
        status = '✅' if lgb_ok and xgb_ok else '❌'
        cal_str = '(캘리브레이터 있음)' if cal_ok else '(캘리브레이터 없음)'
        lines.append(f'- [{status}] {case} {cal_str}')
    return '\n'.join(lines)


def get_backtest_summary() -> str:
    results_path = MODEL_DIR / 'backtest_results.json'
    if not results_path.exists():
        return '## 백테스트 결과\n- 결과 없음 (아직 학습 전)'

    results = json.loads(results_path.read_text())
    if not results:
        return '## 백테스트 결과\n- 결과 없음'

    lines = ['## 최근 백테스트 결과 (최신 10건)']
    for r in results[-10:]:
        status = '✅ PASS' if r.get('passed') else '❌ FAIL'
        lines.append(
            f"- [{r['case']} / {r['model']}]  {status}  "
            f"승률{r['win_rate']*100:.1f}%  Sharpe{r['sharpe']:.2f}  "
            f"MaxDD{r['max_drawdown']*100:.1f}%  신호율{r['signal_rate']*100:.1f}%"
            f"  ({r['timestamp'][:10]})"
        )
    return '\n'.join(lines)


def export_context(question: str | None = None) -> Path:
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    out_path = OUT_DIR / f'context_{ts}.md'

    sections = [
        f'# 주식 자동매매 프로젝트 컨텍스트\n생성: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        SYSTEM_OVERVIEW,
        get_model_status(),
        get_backtest_summary(),
    ]

    if question:
        sections.append(f'## 토론 질문\n{question}')
        sections.append(
            '## 답변 요청 형식\n'
            '1. 찬성 의견 및 근거 (구체적 수치나 논문 있으면 포함)\n'
            '2. 반대 의견 및 근거\n'
            '3. 최종 권장 방향 및 구현 시 주의사항'
        )

    content = '\n\n---\n\n'.join(sections)
    out_path.write_text(content, encoding='utf-8')
    print(f'컨텍스트 파일 생성: {out_path}')
    print('\n' + '─' * 60)
    print('제미나이/GPT에 붙여넣을 내용:')
    print('─' * 60)
    print(content[:3000] + ('...(이하 파일 참조)' if len(content) > 3000 else ''))
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--question', default=None, help='AI에게 물어볼 토론 질문')
    args = parser.parse_args()
    export_context(question=args.question)


if __name__ == '__main__':
    main()
