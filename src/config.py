"""
Central configuration for the AI Subject Matter Expert.
All settings are read from environment variables (via .env).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")


# ── LLM ──────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
PRIMARY_LLM = os.getenv("PRIMARY_LLM", "claude-sonnet-4-6")

# ── HuggingFace ───────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_REPO_ID = os.getenv("HF_REPO_ID", "")

# ── External APIs ─────────────────────────────────────────
COURTLISTENER_API_TOKEN = os.getenv("COURTLISTENER_API_TOKEN", "")
CASELAW_API_KEY = os.getenv("CASELAW_API_KEY", "")

# ── Data paths ────────────────────────────────────────────
RAW_DATA_DIR = ROOT / os.getenv("RAW_DATA_DIR", "data/raw")
PROCESSED_DATA_DIR = ROOT / os.getenv("PROCESSED_DATA_DIR", "data/processed")
VECTOR_DB_DIR = ROOT / os.getenv("VECTOR_DB_DIR", "data/vector_db")

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

# ── Domain ────────────────────────────────────────────────
DOMAIN = os.getenv("DOMAIN", "divorce")
DOMAIN_DESCRIPTION = os.getenv(
    "DOMAIN_DESCRIPTION",
    "divorce law and family law proceedings in the United States"
)
SEARCH_KEYWORDS = [
    kw.strip()
    for kw in os.getenv(
        "SEARCH_KEYWORDS",
        "divorce,custody,alimony,child support,marital property,separation agreement"
    ).split(",")
    if kw.strip()
]

# ── Scraper settings ──────────────────────────────────────
MAX_CASES_PER_SOURCE = int(os.getenv("MAX_CASES_PER_SOURCE", "500"))
SCRAPER_DELAY_MIN = float(os.getenv("SCRAPER_DELAY_MIN", "1.0"))
SCRAPER_DELAY_MAX = float(os.getenv("SCRAPER_DELAY_MAX", "3.0"))

# ── Embeddings ────────────────────────────────────────────
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = f"{DOMAIN}_cases"

# ── RAG ───────────────────────────────────────────────────
DEFAULT_K_CASES = 5
DEFAULT_K_PRINCIPLES = 3


def active_llm_provider() -> str:
    """Detect which LLM provider is configured."""
    if ANTHROPIC_API_KEY:
        return "anthropic"
    if GOOGLE_API_KEY:
        return "google"
    if OPENAI_API_KEY:
        return "openai"
    raise ValueError(
        "No LLM API key found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY in .env"
    )
