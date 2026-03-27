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
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
PRIMARY_LLM = os.getenv("PRIMARY_LLM", "claude-sonnet-4-6")

# ── HuggingFace ───────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_REPO_ID = os.getenv("HF_REPO_ID", "")

# ── Search APIs (for web research) ────────────────────────
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")   # optional — better quality
# DuckDuckGo needs no key and is always available

# ── Data paths ────────────────────────────────────────────
RAW_DATA_DIR = ROOT / os.getenv("RAW_DATA_DIR", "data/raw")
PROCESSED_DATA_DIR = ROOT / os.getenv("PROCESSED_DATA_DIR", "data/processed")
VECTOR_DB_DIR = ROOT / os.getenv("VECTOR_DB_DIR", "data/vector_db")

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

# ── Subject ───────────────────────────────────────────────
SUBJECT = os.getenv("SUBJECT", "artificial intelligence")
SUBJECT_DESCRIPTION = os.getenv(
    "SUBJECT_DESCRIPTION",
    "artificial intelligence, machine learning, and deep learning"
)
SUBJECT_KEYWORDS = [
    kw.strip()
    for kw in os.getenv(
        "SUBJECT_KEYWORDS",
        "artificial intelligence,machine learning,deep learning,neural networks,LLM,GPT,transformers"
    ).split(",")
    if kw.strip()
]

# ── Research settings ─────────────────────────────────────
MAX_DOCS_PER_SOURCE = int(os.getenv("MAX_DOCS_PER_SOURCE", "200"))
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "10"))  # per query
RESEARCH_DELAY_MIN = float(os.getenv("RESEARCH_DELAY_MIN", "1.0"))
RESEARCH_DELAY_MAX = float(os.getenv("RESEARCH_DELAY_MAX", "3.0"))

# ── Embeddings ────────────────────────────────────────────
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = SUBJECT.lower().replace(" ", "_")[:30]

# ── RAG ───────────────────────────────────────────────────
DEFAULT_K_DOCS = int(os.getenv("DEFAULT_K_DOCS", "5"))
DEFAULT_K_LEARNINGS = int(os.getenv("DEFAULT_K_LEARNINGS", "3"))


def active_llm_provider() -> str:
    """Detect which LLM provider is configured."""
    if ANTHROPIC_API_KEY:
        return "anthropic"
    if GOOGLE_API_KEY:
        return "google"
    if GROQ_API_KEY:
        return "groq"
    if OPENAI_API_KEY:
        return "openai"
    raise ValueError(
        "No LLM API key found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, or GROQ_API_KEY in .env"
    )
