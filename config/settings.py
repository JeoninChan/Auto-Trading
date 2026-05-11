import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "datasets"
LOG_DIR  = BASE_DIR / "logs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# API 키
DART_API_KEY    = os.getenv("DART_API_KEY", "")
KIS_APP_KEY     = os.getenv("KIS_APP_KEY", "")
KIS_APP_SECRET  = os.getenv("KIS_APP_SECRET", "")
KIWOOM_APP_KEY  = os.getenv("KIWOOM_APP_KEY", "")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY", "")
GROK_API_KEY      = os.getenv("GROK_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")

# 수집 설정
US_PRICE_PERIOD   = "5y"    # yfinance 기간
US_PRICE_INTERVAL = "1d"    # 일봉 (단타용 분봉은 별도)
KR_PRICE_START    = "20200101"

SEC_RATE_LIMIT    = 0.12    # SEC EDGAR: 10 req/sec 제한 → 0.12초 간격
YFINANCE_DELAY    = 0.3
NAVER_DELAY       = 0.5

BLACKLIST_PATH = BASE_DIR / "blacklist" / "blacklist.json"
