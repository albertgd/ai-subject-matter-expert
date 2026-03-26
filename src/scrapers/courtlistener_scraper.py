"""
CourtListener Scraper — Free REST API for US court opinions.

CourtListener (https://www.courtlistener.com) is a free, open-source legal
research platform run by the Free Law Project. It provides a REST API with
millions of US federal and state court opinions.

API docs: https://www.courtlistener.com/help/api/rest/

Rate limits:
  - Anonymous: 5,000 requests/day
  - Authenticated: 30,000 requests/day (set COURTLISTENER_API_TOKEN in .env)

No API key required for basic use.
"""

import logging
from typing import Dict, List, Optional

from src.config import COURTLISTENER_API_TOKEN, MAX_CASES_PER_SOURCE, SEARCH_KEYWORDS
from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)

API_BASE = "https://www.courtlistener.com/api/rest/v4"


class CourtListenerScraper(BaseScraper):
    """
    Scrapes US court opinions from the CourtListener REST API.

    Targets family law / divorce opinions using keyword search.
    Fetches full opinion text when available.
    """

    SOURCE_NAME = "courtlistener"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if COURTLISTENER_API_TOKEN:
            self.session.headers["Authorization"] = f"Token {COURTLISTENER_API_TOKEN}"
            logger.info("CourtListener: using authenticated access (higher rate limit).")
        else:
            logger.info("CourtListener: using anonymous access (5,000 req/day limit).")

    # ── Public interface ──────────────────────────────────
    def scrape(self, max_cases: int = MAX_CASES_PER_SOURCE) -> List[Dict]:
        """
        Search CourtListener for family law / divorce opinions.

        Returns list of raw case dicts suitable for processing.
        """
        logger.info(f"CourtListener: starting scrape (max {max_cases} cases)...")

        # Build search query from domain keywords
        query = " OR ".join(f'"{kw}"' for kw in SEARCH_KEYWORDS[:4])

        cases = []
        page_url = f"{API_BASE}/search/"
        params = {
            "q": query,
            "type": "o",           # opinions
            "order_by": "score desc",
            "stat_Precedential": "on",
            "filed_after": "2000-01-01",
        }

        seen_ids = set()
        page_count = 0

        while page_url and len(cases) < max_cases:
            page_count += 1
            logger.info(f"  Fetching search page {page_count}...")

            try:
                resp = self._get(page_url, params=params if page_count == 1 else None)
                data = resp.json()
            except Exception as e:
                logger.error(f"  Search failed: {e}")
                break

            results = data.get("results", [])
            if not results:
                break

            for hit in results:
                if len(cases) >= max_cases:
                    break

                opinion_id = str(hit.get("id", ""))
                if not opinion_id or opinion_id in seen_ids:
                    continue
                seen_ids.add(opinion_id)

                if self.already_scraped(f"cl_{opinion_id}"):
                    logger.debug(f"  Skipping already-scraped opinion {opinion_id}")
                    continue

                case = self._build_case_from_hit(hit)
                if case:
                    cases.append(case)
                    self.save_case(case)
                    self._sleep()

            # Pagination
            page_url = data.get("next")
            params = None  # params are encoded in the next URL

        logger.info(f"CourtListener: collected {len(cases)} new cases.")
        return cases

    def scrape_by_ids(self, opinion_ids: List[str]) -> List[Dict]:
        """Fetch specific opinions by their CourtListener ID."""
        cases = []
        for oid in opinion_ids:
            case = self._fetch_opinion(oid)
            if case:
                cases.append(case)
                self.save_case(case)
            self._sleep()
        return cases

    # ── Private helpers ───────────────────────────────────
    def _build_case_from_hit(self, hit: Dict) -> Optional[Dict]:
        """Convert a search result hit into a raw case dict."""
        opinion_id = str(hit.get("id", ""))
        if not opinion_id:
            return None

        # Extract text from search snippet or fetch full opinion
        text = hit.get("snippet", "") or ""
        absolute_url = hit.get("absolute_url", "")
        full_url = f"https://www.courtlistener.com{absolute_url}" if absolute_url else ""

        # Try to get full text from the cluster
        cluster_id = hit.get("cluster_id") or hit.get("cluster", {})
        if isinstance(cluster_id, dict):
            cluster_id = cluster_id.get("id", "")

        # Build case with available data; fetch full text separately
        case = {
            "source_id": f"cl_{opinion_id}",
            "source_name": "CourtListener",
            "title": hit.get("caseName") or hit.get("case_name", "Untitled"),
            "url": full_url,
            "date": hit.get("dateFiled") or hit.get("date_filed", ""),
            "court": hit.get("court") or hit.get("court_id", ""),
            "court_full": hit.get("court_citation_string", ""),
            "citation": hit.get("citation", []),
            "judges": hit.get("judge", ""),
            "status": hit.get("status", ""),
            "text": text,
            "metadata": {
                "opinion_id": opinion_id,
                "cluster_id": str(cluster_id) if cluster_id else "",
                "docket_number": hit.get("docketNumber") or hit.get("docket_number", ""),
                "nature_of_suit": hit.get("suitNature") or hit.get("nature_of_suit", ""),
                "attorneys": hit.get("attorney", ""),
                "score": hit.get("score", 0),
            },
        }

        # Fetch full opinion text if snippet is too short
        if len(text) < 500 and opinion_id:
            full_text = self._fetch_opinion_text(opinion_id)
            if full_text:
                case["text"] = full_text

        return case if case["text"] else None

    def _fetch_opinion_text(self, opinion_id: str) -> str:
        """Fetch the full text of a single opinion."""
        try:
            resp = self._get(f"{API_BASE}/opinions/{opinion_id}/")
            data = resp.json()
            # Try various text fields in order of preference
            for field in ("plain_text", "html_with_citations", "html_lawbox",
                          "html_columbia", "html", "xml_harvard"):
                text = data.get(field, "")
                if text and len(text.strip()) > 200:
                    # Strip HTML if needed
                    if field.startswith("html") or field.startswith("xml"):
                        text = self._strip_html(text)
                    return text
        except Exception as e:
            logger.debug(f"Could not fetch opinion {opinion_id}: {e}")
        return ""

    def _fetch_opinion(self, opinion_id: str) -> Optional[Dict]:
        """Fetch a single opinion by ID."""
        try:
            resp = self._get(f"{API_BASE}/opinions/{opinion_id}/")
            data = resp.json()

            text = ""
            for field in ("plain_text", "html_with_citations", "html_lawbox",
                          "html_columbia", "html", "xml_harvard"):
                raw = data.get(field, "")
                if raw and len(raw.strip()) > 200:
                    text = self._strip_html(raw) if field != "plain_text" else raw
                    break

            if not text:
                return None

            cluster_url = data.get("cluster", "")
            cluster_data = {}
            if cluster_url:
                try:
                    cluster_resp = self._get(cluster_url)
                    cluster_data = cluster_resp.json()
                except Exception:
                    pass

            return {
                "source_id": f"cl_{opinion_id}",
                "source_name": "CourtListener",
                "title": cluster_data.get("case_name", data.get("author_str", "Unknown")),
                "url": f"https://www.courtlistener.com/opinion/{opinion_id}/",
                "date": cluster_data.get("date_filed", ""),
                "court": data.get("cluster", ""),
                "text": text,
                "metadata": {
                    "opinion_id": opinion_id,
                    "type": data.get("type", ""),
                    "per_curiam": data.get("per_curiam", False),
                    "author": data.get("author_str", ""),
                },
            }
        except Exception as e:
            logger.error(f"Failed to fetch opinion {opinion_id}: {e}")
            return None

    @staticmethod
    def _strip_html(html: str) -> str:
        """Strip HTML tags from text."""
        try:
            from bs4 import BeautifulSoup
            return BeautifulSoup(html, "lxml").get_text(separator="\n").strip()
        except Exception:
            import re
            return re.sub(r"<[^>]+>", " ", html).strip()
