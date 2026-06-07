HOT_THEMES = ["AI인프라", "우주"]

THEME_KEYWORDS = {
    "AI인프라": {
        "power_supply":  ["power supply", "data center power", "electricity", "grid", "utility", "Vistra", "Constellation", "NRG", "AES"],
        "data_center":   ["data center", "colocation", "hyperscale", "Equinix", "Iron Mountain", "Digital Realty", "QTS"],
        "cooling":       ["cooling", "thermal management", "liquid cooling", "HVAC", "Vertiv", "Modine"],
        "semiconductor": ["GPU", "AI chip", "inferencing", "training chip", "NVIDIA", "AMD", "Marvell", "Broadcom"],
    },
    "우주": {
        "launch":    ["launch vehicle", "rocket", "launch services", "Rocket Lab", "Redwire", "ABL Space", "SpaceX supplier"],
        "satellite": ["satellite", "LEO constellation", "broadband satellite", "AST SpaceMobile", "Iridium", "Spire Global", "Planet Labs"],
        "lunar":     ["lunar", "moon", "lunar lander", "NASA Artemis", "CLPS", "Intuitive Machines", "Astrobotic"],
    },
    "원자력": {
        "smr":     ["small modular reactor", "SMR", "NuScale", "X-energy", "nuclear power"],
        "uranium": ["uranium", "enrichment", "Centrus", "Cameco", "fuel fabrication"],
    },
    "바이오": {
        "gene":    ["gene therapy", "CRISPR", "genomics", "cell therapy", "CAR-T"],
        "oncology":["oncology", "cancer", "tumor", "immuno-oncology", "checkpoint"],
        "rare":    ["rare disease", "orphan drug", "enzyme replacement"],
    },
    "방산": {
        "missile": ["missile", "hypersonic", "interceptor", "ICBM", "Lockheed", "RTX"],
        "drone":   ["drone", "UAV", "unmanned aerial", "AeroVironment", "Joby"],
        "cyber":   ["cyber warfare", "electronic warfare", "SIGINT"],
    },
    "EV자율주행": {
        "ev":      ["electric vehicle", "EV", "battery pack", "charging station"],
        "autonomy":["autonomous driving", "self-driving", "LiDAR", "radar sensor"],
        "battery": ["battery", "lithium", "cathode", "anode", "solid state"],
    },
    "신재생에너지": {
        "solar":   ["solar", "photovoltaic", "PV panel", "inverter"],
        "wind":    ["wind turbine", "offshore wind", "onshore wind"],
        "storage": ["energy storage", "ESS", "grid battery", "BESS"],
    },
    "핀테크": {
        "payments": ["payments", "digital wallet", "mobile payment", "neobank"],
        "crypto":   ["crypto", "blockchain", "DeFi", "stablecoin", "web3"],
    },
    "바이오의약": {
        "pharma":  ["drug discovery", "clinical trial", "FDA", "NDA", "BLA"],
        "med_device": ["medical device", "surgical robot", "implant", "diagnostic"],
    },
    "사이버보안": {
        "zero_trust": ["zero trust", "SASE", "endpoint", "identity security"],
        "cloud_sec":  ["cloud security", "CASB", "CNAPP", "data protection"],
    },
}

UNIVERSE_PATH = "datasets/us/universe.parquet"
DELAY = 0.4
