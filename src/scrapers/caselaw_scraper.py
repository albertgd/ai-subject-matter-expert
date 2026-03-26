"""
Harvard Caselaw Access Project (CAP) Scraper.

The Harvard Caselaw Access Project provides free API access to millions of
US court cases from 1658 to the present. Full text is freely accessible
for all public-domain cases (pre-1929) and via researcher registration
for modern cases.

API docs: https://api.case.law/v1/
Register for full text: https://case.law/user/register/

Rate limits:
  - Anonymous: 500 cases/day (metadata only)
  - Registered (free): Full text, 500 cases/day
  - Research partners: Bulk access

Set CASELAW_API_KEY in .env for full text access.
"""

import logging
from typing import Dict, List, Optional

from src.config import CASELAW_API_KEY, MAX_CASES_PER_SOURCE, SEARCH_KEYWORDS
from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)

API_BASE = "https://api.case.law/v1"


class CaseLawAccessScraper(BaseScraper):
    """
    Scrapes US court cases from the Harvard Caselaw Access Project API.
    """

    SOURCE_NAME = "caselaw_access"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if CASELAW_API_KEY:
            self.session.headers["Authorization"] = f"Token {CASELAW_API_KEY}"
            logger.info("CaseLaw Access: using authenticated access (full text).")
        else:
            logger.info("CaseLaw Access: using anonymous access (metadata only, 500/day).")

    # ── Public interface ──────────────────────────────────
    def scrape(self, max_cases: int = MAX_CASES_PER_SOURCE) -> List[Dict]:
        """
        Search for family law / divorce cases.
        Returns list of raw case dicts.
        """
        logger.info(f"CaseLaw Access: starting scrape (max {max_cases} cases)...")

        cases = []
        seen_ids: set = set()

        for keyword in SEARCH_KEYWORDS[:3]:  # Use first 3 keywords to stay within rate limits
            if len(cases) >= max_cases:
                break

            remaining = max_cases - len(cases)
            keyword_cases = self._search_keyword(keyword, max_results=min(remaining, 100))
            for case in keyword_cases:
                if case["source_id"] not in seen_ids:
                    seen_ids.add(case["source_id"])
                    cases.append(case)

        logger.info(f"CaseLaw Access: collected {len(cases)} new cases.")
        return cases

    # ── Private helpers ───────────────────────────────────
    def _search_keyword(self, keyword: str, max_results: int = 100) -> List[Dict]:
        """Search for cases matching a keyword."""
        cases = []
        url = f"{API_BASE}/cases/"
        params = {
            "search": keyword,
            "jurisdiction": "us",
            "full_case": "true" if CASELAW_API_KEY else "false",
            "page_size": min(100, max_results),
            "ordering": "-decision_date",
            "decision_date_min": "2000-01-01",
        }

        page_count = 0
        while url and len(cases) < max_results:
            page_count += 1
            logger.info(f"  Keyword '{keyword}': page {page_count}...")

            try:
                resp = self._get(url, params=params if page_count == 1 else None)
                data = resp.json()
            except Exception as e:
                logger.error(f"  Search failed for '{keyword}': {e}")
                break

            results = data.get("results", [])
            if not results:
                break

            for hit in results:
                if len(cases) >= max_results:
                    break

                case_id = str(hit.get("id", ""))
                if not case_id:
                    continue

                source_id = f"cap_{case_id}"
                if self.already_scraped(source_id):
                    continue

                case = self._build_case(hit, keyword)
                if case:
                    cases.append(case)
                    self.save_case(case)
                    self._sleep()

            url = data.get("next")
            params = None

        return cases

    def _build_case(self, hit: Dict, search_keyword: str = "") -> Optional[Dict]:
        """Convert an API result into a raw case dict."""
        case_id = str(hit.get("id", ""))
        if not case_id:
            return None

        # Extract text — available only with API key
        casebody = hit.get("casebody") or {}
        opinions = casebody.get("data", {}).get("opinions", []) if isinstance(casebody.get("data"), dict) else []
        text = ""
        for opinion in opinions:
            text += opinion.get("text", "") + "\n\n"

        # If no full text, use a summary from available metadata
        if not text.strip():
            # Build a minimal record from metadata (still useful for indexing)
            abstract = hit.get("preview", [])
            text = " ".join(abstract) if isinstance(abstract, list) else str(abstract)

        if not text.strip():
            return None

        court = hit.get("court") or {}
        jurisdiction = hit.get("jurisdiction") or {}

        return {
            "source_id": f"cap_{case_id}",
            "source_name": "Harvard Caselaw Access",
            "title": hit.get("name_abbreviation") or hit.get("name", "Untitled"),
            "url": hit.get("frontend_url", f"https://cite.case.law/cases/{case_id}/"),
            "date": hit.get("decision_date", ""),
            "court": court.get("name", "") if isinstance(court, dict) else str(court),
            "jurisdiction": jurisdiction.get("name_long", "") if isinstance(jurisdiction, dict) else str(jurisdiction),
            "citations": [c.get("cite", "") for c in (hit.get("citations") or [])],
            "text": text.strip(),
            "metadata": {
                "case_id": case_id,
                "docket_number": hit.get("docket_number", ""),
                "reporter": hit.get("reporter", {}).get("full_name", "") if isinstance(hit.get("reporter"), dict) else "",
                "volume": hit.get("volume", {}).get("volume_number", "") if isinstance(hit.get("volume"), dict) else "",
                "first_page": hit.get("first_page", ""),
                "last_page": hit.get("last_page", ""),
                "search_keyword": search_keyword,
            },
        }
