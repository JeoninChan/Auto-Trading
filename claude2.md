# 프로젝트 맥락 및 현재 상태 (세션 2 인계용)

## 프로젝트 개요
미국/한국 주식 자동매매 시스템. 3모델 구조.
- 우량주 모델: 재무 기반 텐버거 탐색 → 보고서 → 사람이 매수
- 급등주 모델: 이슈/거래량 폭발 포착 → 자동매매 or 알림
- 단타 모델: 기술적 지표 기반 당일 매매

API: 한국 → 키움 REST, 미국 → KIS(한국투자증권) REST
데이터: yfinance(미국), pykrx(한국), DART(공시), SEC EDGAR
AI 분석: Gemini, Claude, Grok, GPT 앙상블 (0~200점 상대비교)

---

## 이번 세션에서 만든 것

### 1. `screener/us_smallmid_screen.py` — 기존 하드필터 스크리너
- PEG 칼럼 추가 (매출성장률 뒤에 삽입)
- 필터 기준: 시총 5000억~28조, D/E≤100, 유동비율≥1.0, 매출총이익률≥10%, 영업이익률≥10%, 순이익률≥5%, 매출성장률≥20%

### 2. `screener z-score/` 폴더 — 연속 점수 기반 스크리너

#### `us_zscore_raw.py`
- 유니버스 전체 → 유동비율 < 1.3 제외 → 순수 지표값만 저장
- 출력: `output/us_zscore_raw_*.{xlsx,json,txt}`
- 정렬: 시가총액 내림차순

#### `us_zscore_score.py`
- 유니버스 전체 → 유동비율 < 1.3 제외 → 점수화
- 가중치: 시총20% + PEG30% + 매출성장률20% + 순이익률15% + 부채비율10% + 매총이익률5%
- 칼럼 배치: 각 지표 바로 뒤에 점수 칼럼 삽입
- 종합점수(가중합) + 평균점수(단순평균) 둘 다 포함
- 마지막 행: [평균] 전 종목 점수 평균
- 출력: `output/us_zscore_scored_*.{xlsx,json,txt}`
- 정렬: 종합점수 내림차순

#### 점수 곡선 (`Normal-distribution-score.py` 기반)
| 지표 | 방식 | 기준점 |
|------|------|--------|
| 시가총액 | Plateau + Gaussian Falloff | 5000억~28조=100점 |
| PEG | 선형 | ≤0.5→100, 1.0→75, 1.5→50, 2.5+→0 |
| 매출성장률 | CDF σ=15 | 20%=50점 |
| 순이익률 | CDF σ=10 | 5%=50점 |
| 부채비율 | 역방향 CDF σ=40 | 100%=50점 |
| 매출총이익률 | CDF σ=15 | 10%=50점 |

### 3. `screener z-score/score.py` — 일봉 기술적 분석 점수화
- 인자로 scored 엑셀 파일 경로를 받음
- 각 티커에 대해 yfinance 200일 일봉 수집
- pandas-ta로 RSI, MACD, VWAP, 볼린저밴드, MA배열, 거래량 계산
- 0~100 기술점수 + 적극매수/매수/중립/매도/적극매도 신호
- 원본 엑셀 파일에 `기술점수`, `기술신호` 칼럼 추가 후 덮어씀
- `[평균]` 행 자동 스킵

```bash
python3 "screener z-score/score.py" "screener z-score/output/us_zscore_scored_20260530_1730.xlsx"
```

#### 기술적 지표 가중치
| 지표 | 가중치 |
|------|--------|
| RSI(14) | 25% |
| MACD | 20% |
| VWAP | 20% |
| 볼린저밴드 %B | 15% |
| MA(50/200) 배열 | 10% |
| 거래량 | 10% |

---

## 현재 디렉터리 구조
```
stock_trading/
├── CLAUDE.md
├── claude2.md                  ← 이 파일
├── Normal-distribution-score.py  ← 점수 곡선 시각화
├── screener/
│   ├── us_smallmid_screen.py   ← 하드필터 스크리너
│   └── output/                 ← .gitignore 예정
├── screener z-score/
│   ├── us_zscore_raw.py
│   ← us_zscore_score.py
│   ├── score.py                ← 기술적 분석 점수화
│   └── output/                 ← .gitignore 예정
│       └── us_zscore_scored_20260530_1730.xlsx  (현재 파일)
└── datasets/
    └── us/
        └── universe.parquet
```

---

## 미완료 항목

### 즉시 해야 할 것
1. `.gitignore`에 output 폴더 추가 (`screener/output/`, `screener z-score/output/`)
2. `us_zscore_raw.py`, `us_zscore_score.py` 양쪽 `_to_excel()` openpyxl 포맷 코드 복원
   - 사용자가 롤백해서 현재 셀루프 없는 단순버전으로 돌아가있음
   - 복원 코드:
   ```python
   def _to_excel(df: pd.DataFrame, path: Path):
       with pd.ExcelWriter(path, engine="openpyxl") as writer:
           df.to_excel(writer, index=False, sheet_name="스크리닝결과")
           ws = writer.sheets["스크리닝결과"]
           for cell in ws[1]:
               cell.font = cell.font.copy(bold=True)
           for col in ws.columns:
               width = max(len(str(c.value or "")) for c in col) + 4
               ws.column_dimensions[col[0].column_letter].width = min(width, 45)
       logger.info(f"Excel → {path}")
   ```

### 다음 단계 (CLAUDE.md 로드맵 기준)
- KR 가격 수집 실패 원인 파악 (pykrx 오류, failed.csv)
- KR 뉴스 수집 0건 오류 수정
- 전체 유니버스 수집 스크립트 완성
- pump_scanner 폴더 구조 코드 작성
- 백테스트 프레임워크 준비

---

## 개발 규칙 (CLAUDE.md에서)
- API 키는 절대 하드코딩 금지 → .env 사용
- 주석은 기본 없음 (WHY가 명확할 때만)
- 파일 상단 docstring 쓰지 않음
- 블랙리스트 체크는 주문 함수 내부에서 반드시 수행

---

## 설치된 의존성
```bash
pip3 install yfinance pandas loguru tqdm scipy openpyxl pandas-ta
```
