"""
Generic web page fetcher.

Fetches any URL and extracts clean text using BeautifulSoup.
Skips PDFs, binary files, and very short pages.
"""

import hashlib
import logging
import re
from typing import Optional, Dict
from urllib.parse import urlparse

from src.research.base import BaseCollector

logger = logging.getLogger(__name__)

# Tags whose content is noise
_NOISE_TAGS = [
    "script", "style", "nav", "header", "footer",
    "aside", "form", "noscript", "iframe", "advertisement",
]

_MIN_TEXT_LENGTH = 300


class WebFetcher(BaseCollector):
    """Fetches and cleans arbitrary web pages."""

    SOURCE_NAME = "web"

    def fetch(self, url: str, title_hint: str = "") -> Optional[Dict]:
        """
        Fetch a URL and return a raw document dict, or None if unusable.

        Document fields:
          source_id, source_name, title, url, text, date, author
        """
        if self._is_binary_url(url):
            return None

        source_id = f"web_{self._url_to_id(url)}"
        if self.already_collected(source_id):
            logger.debug(f"Already fetched: {url}")
            return None

        try:
            resp = self._get(url)
        except Exception as e:
            logger.debug(f"Fetch failed {url}: {e}")
            return None

        content_type = resp.headers.get("Content-Type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            return None

        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.error("beautifulsoup4 not installed")
            return None

        soup = BeautifulSoup(resp.text, "lxml")

        # Remove noise
        for tag in soup.find_all(_NOISE_TAGS):
            tag.decompose()

        # Title
        title = title_hint
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True) or title
        if not title:
            t = soup.find("title")
            if t:
                title = t.get_text(strip=True).split("|")[0].split("-")[0].strip()

        # Main text — try semantic tags first
        text = ""
        for selector in ["article", "main", "[role='main']", "div.content",
                         "div.post-content", "div.entry-content", "body"]:
            el = soup.select_one(selector)
            if el:
                text = el.get_text(separator="\n", strip=True)
                if len(text) >= _MIN_TEXT_LENGTH:
                    break

        if len(text) < _MIN_TEXT_LENGTH:
            return None

        # Normalise whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Date — look for common patterns
        date = ""
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',
            r'(?:January|February|March|April|May|June|July|August|'
            r'September|October|November|December)\s+\d{1,2},\s+\d{4}',
            r'\d{1,2}/\d{1,2}/\d{4}',
        ]
        for pat in date_patterns:
            m = re.search(pat, text[:2000])
            if m:
                date = m.group(0)
                break

        # Author
        author = ""
        author_el = soup.select_one(
            "[rel='author'], .author, .byline, [itemprop='author']"
        )
        if author_el:
            author = author_el.get_text(strip=True)[:200]

        doc = {
            "source_id": source_id,
            "source_name": urlparse(url).netloc,
            "title": title[:500],
            "url": url,
            "date": date,
            "author": author,
            "text": text,
        }
        self.save_document(doc)
        return doc

    # ── Utilities ─────────────────────────────────────────
    @staticmethod
    def _url_to_id(url: str) -> str:
        """Short stable ID from a URL."""
        return hashlib.md5(url.encode()).hexdigest()[:16]

    @staticmethod
    def _is_binary_url(url: str) -> bool:
        binary_exts = {".pdf", ".doc", ".docx", ".xls", ".xlsx",
                       ".ppt", ".pptx", ".zip", ".tar", ".gz",
                       ".jpg", ".jpeg", ".png", ".gif", ".mp4", ".mp3"}
        return any(url.lower().endswith(ext) for ext in binary_exts)
