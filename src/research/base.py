"""
Base document collector — shared infrastructure for all research sources.

Provides:
  - HTTP requests with retry + rate-limit sleep
  - Save/load JSON documents to data/raw/
  - Deduplication via source_id tracking
"""

import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

from src.config import RAW_DATA_DIR, RESEARCH_DELAY_MIN, RESEARCH_DELAY_MAX

logger = logging.getLogger(__name__)


class BaseCollector:
    """Abstract base for all document collectors."""

    SOURCE_NAME: str = "unknown"

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        delay_min: float = RESEARCH_DELAY_MIN,
        delay_max: float = RESEARCH_DELAY_MAX,
        max_retries: int = 3,
    ):
        self.output_dir = output_dir or (RAW_DATA_DIR / self.SOURCE_NAME)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (compatible; AI-SME-Bot/1.0; "
                "+https://github.com/albertgd/ai-subject-matter-expert)"
            )
        })

    # ── HTTP ──────────────────────────────────────────────
    def _get(self, url: str, **kwargs) -> requests.Response:
        """GET with retry and exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                resp = self._session.get(url, timeout=15, **kwargs)
                resp.raise_for_status()
                return resp
            except requests.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise
                wait = 2 ** attempt + random.uniform(0, 1)
                logger.debug(f"Retry {attempt+1} for {url}: {e} — waiting {wait:.1f}s")
                time.sleep(wait)

    def _sleep(self):
        """Polite delay between requests."""
        time.sleep(random.uniform(self.delay_min, self.delay_max))

    # ── Persistence ───────────────────────────────────────
    def save_document(self, doc: Dict) -> Path:
        """Save a document dict as JSON. Returns the file path."""
        source_id = doc.get("source_id", "unknown")
        path = self.output_dir / f"{source_id}.json"
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def already_collected(self, source_id: str) -> bool:
        """Return True if this document was already saved."""
        return (self.output_dir / f"{source_id}.json").exists()

    def load_all(self) -> List[Dict]:
        """Load all saved documents from output_dir."""
        docs = []
        for path in sorted(self.output_dir.glob("*.json")):
            try:
                docs.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
        return docs
