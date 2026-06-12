# 프로젝트 맥락 및 현재 상태 (세션 3 인계용)

## 이번 세션 목표
레버리지 리플레이 수익률 -43% 개선 + 실거래 현실화 작업

---

## 이번 세션에서 수정한 것

### 1. `day_trading/backtest/leverage_replay.py`

#### DEFAULT_ETF_PAIRS 확장 (7쌍 → 48쌍, 96 ETFs)
브로드마켓 3x: TQQQ/SQQQ, SPXL/SPXS, UPRO/SPXU, UDOW/SDOW, UMDD/SMDD, URTY/SRTY
브로드마켓 2x: QLD/QID, SSO/SDS, DDM/DXD, MVV/MZZ, SAA/SDD, UWM/TWM
섹터 3x: TECL/TECS, FAS/FAZ, TNA/TZA, SOXL/SOXS, LABU/LABD, ERX/ERY, NUGT/DUST, DRN/DRV, WEBL/WEBS, GUSH/DRIP, JNUG/JDST, HIBL/HIBS, YINN/YANG, EDC/EDZ, FNGU/FNGD, INDL/INDZ
섹터 2x: ROM/REW, UCO/SCO, UGL/GLL, AGQ/ZSL, BOIL/KOLD, UYG/SKF, RXL/RXD, UYM/SMN, UCC/SCC, UPW/SDP
근사쌍: CURE/RXD, UTSL/SDP, DPST/FAZ, MIDU/MZZ, NAIL/DRV, DFEN/SDS
VIX: UVXY/SVXY
단일종목: NVDL/NVDS, TSLL/TSLS, AAPU/AAPD, AMZU/AMZD, GGLL/GGLS, METU/METD, MSFO/MSFD

#### Fix 1 — 롤링 P90/P10 자동 임계값 (핵심 수정)
- 기존: 하드코딩 `long_thresh=0.60`, `short_thresh=0.40` (전 기간 미래 참조)
- 변경: 직전 500봉 proba 캐시 → P90=long_thresh, P10=short_thresh (look-ahead 제거)
- 데이터 50봉 미만이면 fallback: `base_threshold * 1.2 / 0.8`

#### Fix 2 — SPY ADX 추세 필터
- SPY ADX > 35일 때만 레버리지 ETF 진입 허용
- 변천: >25 (원래) → >30 (1차 강화) → >35 (현재)
- 주석으로 이전 값 보존 (롤백 가능)

#### Fix 3 — 숏 조건 수정
- 기존: `market_p <= short_thresh` (항상 숏 가능)
- 변경: `market_p <= short_thresh and spy_bull == 0` (하락장에서만 숏)
- 이유: 나스닥 우상향 구조에서 항상 숏 허용은 비현실적

#### Fix 4 — 서킷브레이커 (-15%)
- 고점 대비 -15% 낙폭 시 신규 진입 중단
- `equity_peak` 추적, 낙폭 회복 시 자동 재개

#### Fix 5 — 멀티포지션
- 최대 5개 동시 포지션, 각 8% equity
- `positions` dict로 관리, 중복 진입 방지

---

### 2. `day_trading/backtest/portfolio_replay.py`

#### 실시간 BUY/SELL 거래 로그 추가
```
[BUY]  NVDL     @  $47.20 | proba=0.731 | 잔고: $10,472 (+4.7%)
[SELL] NVDL     @  $51.10 | ✓  +8.20%  PnL:  +$859 | 익절  | 잔고: $11,331 (+13.3%)
[STOP] NVDL     @  $45.80 | ✗  -3.10%  PnL:  -$324 | 손절  | 잔고: $10,148 (+1.5%)
```

#### ADX 필터 + 서킷브레이커 파라미터 추가
- `adx_filter: int = 0` — 0이면 비활성화, >0이면 SPY ADX ≤ 값일 때 진입 차단
- `circuit_breaker: float = 0.15` — 레버리지와 동일 로직

---

### 3. `day_trading/local_trainer.py`

#### --train-cutoff 플래그 (캘리브레이터 오염 방지)
- 사용법: `--train-cutoff 2025-06-01`
- 이 날짜 이후 데이터는 학습/캘리브레이션에서 제외
- replay 기간 데이터가 학습에 섞이는 문제 방지
- 버그 수정: `pd.Timestamp(cutoff, tz='UTC')` — tz 없으면 UTC 비교 에러 발생

#### --download-leverage-etfs 플래그
- DEFAULT_ETF_PAIRS 96개 전부 yfinance로 다운로드
- 기존 파일 있으면 skip (skip=True)

#### --circuit-breaker 플래그
- 기본값 0.15 (15%), 0이면 비활성화
- leverage_replay와 portfolio_replay 양쪽에 전달

---

### 4. `retrain_all.sh` 생성
```bash
#!/bin/bash
cd /Users/changpt/Downloads/stock_trading
python3 day_trading/local_trainer.py --skip-download --cases leverage_mid_large mid_large smallcap small_mid all --train-cutoff 2025-06-01 --no-backtest
```
- 한 번에 전 케이스 재학습
- zsh 멀티라인 명령어 오류 방지용

---

## 수정 과정에서 발견/수정한 버그

| 버그 | 원인 | 수정 |
|------|------|------|
| `UnboundLocalError: _get_market_proba` | proba 캐시 사전계산 블록이 중첩 함수 정의보다 앞에 삽입됨 | 블록을 함수 정의 2개 뒤로 이동 |
| `TypeError: Invalid comparison datetime64[ns, UTC] and Timestamp` | `pd.Timestamp(train_cutoff)` timezone-naive | `tz='UTC'` 추가 |
| zsh `command not found: leverage_mid_large` | `\` 멀티라인 명령어를 zsh가 분리 해석 | retrain_all.sh 스크립트로 해결 |

---

## 수익률 개선 추이

| 시점 | 수익률 | 주요 변경 |
|------|--------|----------|
| 초기 | -43% | 하드코딩 임계값 0.60/0.40 |
| 1차 개선 | -24% | 롤링 P90/P10 임계값 적용 |
| 2차 개선 | -14% | ADX>35, 숏=하락장only, 서킷브레이커 |
| 재학습 후 | 측정 중 | train-cutoff 2025-06-01 적용 |

---

## 아키텍처 확인 사항

### leverage 모델 학습 데이터
- `local_trainer.py` 237번 줄: `'leverage_mid_large': mid + large`
- ETF 자체 데이터로 학습하는 게 **아님**
- mid+large cap 본주 데이터(S&P500)로 학습 → ETF에 추론 적용
- ETF 데이터 부족 문제 이미 설계상 해결되어 있음

### ETF 추가 시 재학습 불필요
- 모델은 RSI/MACD/ATR 등 기술적 지표를 학습하는 것 (티커 무관)
- 새 ETF 추가 시 데이터 다운로드만 하면 기존 모델로 바로 추론 가능
- 재학습은 나중에 여유될 때 해도 됨

---

## 현재 실행 명령어

### 전체 재학습 (--no-backtest)
```bash
bash retrain_all.sh
```

### 레버리지 리플레이 (2년치)
```bash
python3 day_trading/local_trainer.py --skip-download --leverage-replay --replay-start 2024-06-12
```

### 미드라지 리플레이
```bash
python3 day_trading/local_trainer.py --skip-download --replay --cases mid_large
```

### ETF 전체 다운로드
```bash
python3 day_trading/local_trainer.py --download-leverage-etfs
```

---

## 미완료 / 다음 세션 과제

### 즉시 할 것
1. 단일종목 ETF 쌍 확장 — CONL/CONS, MSTU/MSTD 등 롱+숏 쌍 둘 다 있는 것만 추가
   - 기준: 롱 ETF + 인버스 ETF 둘 다 있을 것 (한쪽만 있으면 스킵)
   - yfinance 데이터 존재 여부로 자동 필터링

### 다음 단계
2. 레버리지 재학습 후 리플레이 결과 비교 (현재 리플레이 실행 중)
3. 미드라지 포트폴리오 리플레이 결과 확인
4. 모의투자 API 연동 (KIS, 키움) 준비
5. KR 가격 수집 실패 원인 파악 (pykrx failed.csv)
6. KR 뉴스 수집 0건 오류 수정

---

## 디렉터리 구조 (단타 관련)
```
day_trading/
├── config.py
├── local_trainer.py          ← 학습/리플레이 메인 진입점
├── backtest/
│   ├── leverage_replay.py    ← 레버리지 ETF 전용 리플레이
│   └── portfolio_replay.py   ← 일반 종목 포트폴리오 리플레이
├── models/
│   ├── model_leverage_mid_large_lgb.txt
│   ├── model_leverage_mid_large_xgb.pkl
│   ├── calibrator_leverage_mid_large_lgb.pkl
│   ├── calibrator_leverage_mid_large_xgb.pkl
│   ├── threshold_leverage_mid_large.json   ← {lgb, xgb, ensemble} 최적 임계값
│   └── ... (mid_large, smallcap, small_mid, all 동일 구조)
└── data/
    └── us/prices/            ← 티커별 .parquet (일봉)
```

```
retrain_all.sh                ← 전 케이스 한번에 재학습
```

---

## 개발 규칙
- API 키는 절대 하드코딩 금지 → .env 사용
- 블랙리스트 체크는 주문 함수 내부에서 반드시 수행
- 백테스트 통과 전 실거래 진입 금지
- 훈련/검증/테스트 기간 시간순 분할 (랜덤 분할 금지)
- 주석은 기본 없음 (WHY가 명확할 때만)
