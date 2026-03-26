"""
Base scraper class with shared utilities: rate limiting, retry logic, output saving.
"""

import json
import logging
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log

from src.config import RAW_DATA_DIR, SCRAPER_DELAY_MIN, SCRAPER_DELAY_MAX

logger = logging.getLogger(__name__)


class BaseScraper(ABC):
    """
    Abstract base class for all content scrapers.

    Each subclass must implement `scrape()` which returns a list of raw case dicts.
    Each raw case dict should contain at minimum:
        - source_id:    Unique ID within the source (e.g. opinion ID, case number)
        - source_name:  Name of the data source (e.g. "CourtListener")
        - title:        Case or document title
        - url:          Canonical URL of the original document
        - text:         Full extracted text
        - date:         Publication / decision date (ISO string or empty)
        - metadata:     Dict of additional source-specific fields
    """

    SOURCE_NAME: str = "unknown"

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        delay_min: float = SCRAPER_DELAY_MIN,
        delay_max: float = SCRAPER_DELAY_MAX,
        max_retries: int = 3,
    ):
        self.output_dir = (output_dir or RAW_DATA_DIR) / self.SOURCE_NAME
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.max_retries = max_retries
        self.session = self._build_session()

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        )

    # ── Abstract interface ────────────────────────────────
    @abstractmethod
    def scrape(self, max_cases: int = 100) -> List[Dict]:
        """Collect up to max_cases raw case dicts and return them."""

    # ── HTTP helpers ──────────────────────────────────────
    def _build_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (compatible; AI-SME-Research-Bot/1.0; "
                "+https://github.com/albertgd/ai-subject-matter-expert)"
            ),
            "Accept": "application/json, text/html, */*",
            "Accept-Language": "en-US,en;q=0.9",
        })
        return session

    def _get(self, url: str, params: Optional[Dict] = None, **kwargs) -> requests.Response:
        """GET with retry and exponential backoff."""
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=30, **kwargs)
                resp.raise_for_status()
                return resp
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    wait = 60 * attempt
                    logger.warning(f"Rate limited on attempt {attempt}. Waiting {wait}s...")
                    time.sleep(wait)
                elif attempt == self.max_retries:
                    raise
                else:
                    logger.warning(f"HTTP error on attempt {attempt}: {e}. Retrying...")
                    time.sleep(2 ** attempt)
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries:
                    raise
                logger.warning(f"Request error on attempt {attempt}: {e}. Retrying...")
                time.sleep(2 ** attempt)

    def _sleep(self):
        """Polite random sleep between requests."""
        time.sleep(random.uniform(self.delay_min, self.delay_max))

    # ── Output helpers ────────────────────────────────────
    def save_case(self, case: Dict) -> Path:
        """Save a single case dict as JSON to output_dir."""
        filename = f"{case['source_id']}.json".replace("/", "_").replace(":", "_")
        path = self.output_dir / filename
        path.write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def save_all(self, cases: List[Dict]) -> int:
        """Save all cases and return count of saved files."""
        saved = 0
        for case in cases:
            try:
                self.save_case(case)
                saved += 1
            except Exception as e:
                logger.warning(f"Failed to save {case.get('source_id', '?')}: {e}")
        logger.info(f"Saved {saved}/{len(cases)} cases to {self.output_dir}")
        return saved

    def load_saved(self) -> List[Dict]:
        """Load all previously saved JSON cases from output_dir."""
        cases = []
        for path in sorted(self.output_dir.glob("*.json")):
            try:
                cases.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")
        return cases

    def already_scraped(self, source_id: str) -> bool:
        """Check if a case has already been scraped (for resumable runs)."""
        filename = f"{source_id}.json".replace("/", "_").replace(":", "_")
        return (self.output_dir / filename).exists()
