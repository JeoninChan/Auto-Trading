SECTOR_ETFS = {
    # 기술
    "반도체":       ["SOXX", "SMH", "SOXQ"],
    "AI인프라":     ["GRID", "AIQ", "BOTZ", "WTAI", "IRBO"],
    "클라우드SaaS": ["SKYY", "WCLD", "CLOU"],
    "사이버보안":   ["HACK", "CIBR", "BUG"],
    "로보틱스":     ["ROBO", "ARKQ"],
    "5G통신":       ["FIVG", "NXTG"],

    # 에너지/자원
    "원자력":       ["NLR", "URNM", "URA"],
    "신재생에너지": ["ICLN", "QCLN", "ACES"],
    "수소":         ["HDRO", "HYDR", "HJEN"],
    "석유가스":     ["XLE", "XOP", "OIH"],
    "금속광물":     ["XME", "COPX", "PICK"],
    "금/귀금속":    ["GDX", "GDXJ", "SIL"],
    "농업/식량":    ["MOO", "DBA", "SOIL"],
    "수자원":       ["PHO", "FIW", "CGW"],

    # 헬스케어/바이오
    "바이오":       ["XBI", "IBB", "ARKG"],
    "제약":         ["XPH", "PPH", "PJP"],
    "헬스케어":     ["XLV", "VHT", "IYH"],
    "의료기기":     ["IHI", "MDEV"],

    # 소비/라이프스타일
    "소비재":       ["XLY", "VCR"],
    "리테일/패션":  ["XRT", "PMR"],
    "게임이스포츠": ["HERO", "ESPO", "NERD"],
    "미디어엔터":   ["XLC", "PBS", "IYC"],
    "여행항공":     ["JETS", "AWAY", "TRYP"],
    "레저스포츠":   ["HAVE"],

    # 금융
    "핀테크":       ["ARKF", "FINX", "IPAY"],
    "은행금융":     ["XLF", "KBE", "KRE"],
    "블록체인크립토": ["BLOK", "BKCH", "BITQ"],

    # 인프라/산업
    "방산":         ["ITA", "XAR", "DFEN"],
    "우주":         ["UFO", "ROKT"],
    "EV자율주행":   ["LIT", "DRIV", "KARS", "IDRV"],
    "건설인프라":   ["PAVE", "ITB", "PKB"],
    "운송물류":     ["XTN", "IYT", "FTXR"],
    "유틸리티":     ["XLU", "VPU"],
    "부동산리츠":   ["VNQ", "XLRE", "IYR"],
    "소재화학":     ["XLB", "VAW", "MXI"],

    # 지역/신흥국
    "중국테크":     ["MCHI", "KWEB", "CQQQ"],
    "인도":         ["INDA", "PIN", "SMIN"],
    "신흥국":       ["EEM", "VWO"],
}

SECTOR_KEYWORDS = {
    "반도체":       ["semiconductor", "chip", "wafer", "TSMC", "foundry", "fab", "DRAM", "NAND", "GPU", "ASIC"],
    "AI인프라":     ["data center", "AI infrastructure", "power supply", "cooling", "hyperscaler", "NVIDIA", "GPU cluster"],
    "클라우드SaaS": ["cloud", "SaaS", "subscription software", "AWS", "Azure", "multi-cloud", "ARR"],
    "사이버보안":   ["cybersecurity", "zero trust", "ransomware", "endpoint security", "SIEM", "SOC", "firewall"],
    "로보틱스":     ["robotics", "automation", "industrial robot", "cobots", "warehouse automation"],
    "5G통신":       ["5G", "spectrum", "telecom", "base station", "mmWave", "network slicing"],
    "원자력":       ["nuclear", "SMR", "small modular reactor", "uranium", "fusion", "fission", "reactor"],
    "신재생에너지": ["solar", "wind", "renewable", "clean energy", "battery storage", "ESS", "green hydrogen"],
    "수소":         ["hydrogen", "fuel cell", "electrolyzer", "green hydrogen", "H2"],
    "석유가스":     ["oil", "gas", "LNG", "pipeline", "upstream", "downstream", "refinery", "OPEC"],
    "금속광물":     ["copper", "lithium", "cobalt", "nickel", "rare earth", "critical minerals", "mining"],
    "금/귀금속":    ["gold", "silver", "precious metals", "gold mining", "bullion"],
    "농업/식량":    ["agriculture", "food security", "crop", "fertilizer", "precision farming", "agtech", "grain"],
    "수자원":       ["water", "desalination", "wastewater", "water infrastructure", "aquifer"],
    "바이오":       ["biotech", "gene therapy", "clinical trial", "FDA approval", "oncology", "genomics", "CRISPR", "mRNA"],
    "제약":         ["pharma", "drug", "pipeline", "FDA", "NDA", "BLA", "generic", "specialty pharma"],
    "헬스케어":     ["healthcare", "hospital", "health insurance", "managed care", "telehealth"],
    "의료기기":     ["medical device", "surgical robot", "implant", "diagnostic", "wearable health"],
    "소비재":       ["consumer discretionary", "retail", "e-commerce", "luxury", "spending"],
    "리테일/패션":  ["retail", "fashion", "apparel", "luxury brands", "DTC", "outlet"],
    "게임이스포츠": ["gaming", "esports", "video game", "mobile game", "metaverse", "VR gaming"],
    "미디어엔터":   ["media", "streaming", "content", "entertainment", "studio", "OTT", "Netflix"],
    "여행항공":     ["travel", "airline", "hotel", "tourism", "cruise", "booking", "TSA"],
    "레저스포츠":   ["leisure", "sports", "fitness", "outdoor", "recreation"],
    "핀테크":       ["fintech", "payments", "digital banking", "neobank", "BNPL", "remittance"],
    "은행금융":     ["bank", "interest rate", "fed", "credit", "lending", "mortgage"],
    "블록체인크립토": ["crypto", "blockchain", "bitcoin", "ethereum", "DeFi", "NFT", "stablecoin", "web3"],
    "방산":         ["defense", "military", "missile", "DoD", "Pentagon", "NATO", "weapons", "drone warfare"],
    "우주":         ["space", "satellite", "launch vehicle", "lunar", "NASA", "SpaceX", "orbit", "asteroid"],
    "EV자율주행":   ["electric vehicle", "EV", "autonomous driving", "self-driving", "battery", "charging infrastructure"],
    "건설인프라":   ["construction", "infrastructure", "bridge", "CHIPS Act", "IRA", "government spending"],
    "운송물류":     ["logistics", "freight", "trucking", "supply chain", "last-mile delivery", "port"],
    "유틸리티":     ["utility", "electric grid", "power generation", "regulated utility", "rate base"],
    "부동산리츠":   ["REIT", "real estate", "data center REIT", "industrial REIT", "office vacancy"],
    "소재화학":     ["materials", "chemical", "specialty chemical", "plastics", "paint"],
    "중국테크":     ["China tech", "Alibaba", "Tencent", "ByteDance", "Chinese regulation", "ADR"],
    "인도":         ["India", "Modi", "PLI scheme", "Indian market", "NSE", "BSE"],
    "신흥국":       ["emerging market", "EM", "developing economy", "MSCI EM"],
}

TRACKED_FUNDS = {
    "Citadel":          "0001423298",
    "Point72":          "0001418396",
    "Renaissance":      "0001037389",
    "ARK Invest":       "0001697748",
    "Tiger Global":     "0001167483",
    "Coatue":           "0001336528",
    "Dragoneer":        "0001548681",
    "Scion (Burry)":    "0001341439",
    "Pershing Square":  "0001336528",
    "Appaloosa":        "0001656456",
    "Third Point":      "0001040273",
    "Lone Pine":        "0001166408",
}

REDDIT_SUBS = [
    "wallstreetbets", "stocks", "investing",
    "StockMarket", "SecurityAnalysis", "options",
    "smallstreetbets", "pennystocks",
]

DELAY = 0.5
OUTPUT_DIR = "signals"
