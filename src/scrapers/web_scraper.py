"""
General Web Scraper — public legal information sites.

Scrapes publicly available divorce/family law information from:
  - Justia (justia.com) — US case law, statutes, and legal guides
  - FindLaw (findlaw.com) — Legal articles and guides
  - BAILII (bailii.org) — British and Irish case law
  - Google Scholar (scholar.google.com) — Legal opinions (via search)

These sources provide legal knowledge, case summaries, and statutes
to supplement the API-based case law scrapers.
"""

import logging
import re
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from src.config import MAX_CASES_PER_SOURCE, SEARCH_KEYWORDS
from .base_scraper import BaseScraper

logger = logging.getLogger(__name__)


class WebScraper(BaseScraper):
    """
    Scrapes public legal information from multiple websites.
    Focuses on easily accessible HTML pages with no authentication needed.
    """

    SOURCE_NAME = "web"

    # Target sources mapped to their scrape methods
    SOURCES = {
        "justia": "https://law.justia.com",
        "bailii": "https://www.bailii.org",
    }

    def scrape(self, max_cases: int = MAX_CASES_PER_SOURCE) -> List[Dict]:
        """Scrape all configured web sources."""
        all_cases = []
        per_source = max(10, max_cases // len(self.SOURCES))

        for source_name in self.SOURCES:
            if len(all_cases) >= max_cases:
                break
            try:
                method = getattr(self, f"_scrape_{source_name}", None)
                if method:
                    cases = method(max_cases=per_source)
                    all_cases.extend(cases)
                    logger.info(f"  {source_name}: collected {len(cases)} documents")
            except Exception as e:
                logger.error(f"  {source_name} scraper failed: {e}")

        return all_cases[:max_cases]

    # ── Justia ────────────────────────────────────────────
    def _scrape_justia(self, max_cases: int = 100) -> List[Dict]:
        """
        Scrape divorce-related case law from Justia.
        Uses Justia's US Law section for family law cases.
        """
        cases = []
        base_url = "https://law.justia.com"

        # Justia family law topic pages
        family_law_urls = [
            "https://law.justia.com/topics/family-law/divorce/",
            "https://law.justia.com/topics/family-law/child-custody/",
            "https://law.justia.com/topics/family-law/spousal-support-alimony/",
            "https://law.justia.com/topics/family-law/marital-property-division/",
        ]

        seen_urls = set()
        for topic_url in family_law_urls:
            if len(cases) >= max_cases:
                break
            try:
                topic_cases = self._scrape_justia_topic(topic_url, seen_urls, max_per_page=30)
                cases.extend(topic_cases)
                logger.info(f"  Justia {topic_url.split('/')[-2]}: {len(topic_cases)} docs")
                self._sleep()
            except Exception as e:
                logger.warning(f"  Justia topic failed ({topic_url}): {e}")

        return cases[:max_cases]

    def _scrape_justia_topic(self, url: str, seen_urls: set, max_per_page: int = 30) -> List[Dict]:
        """Scrape a Justia topic/category page for case links."""
        cases = []
        try:
            resp = self._get(url)
            soup = BeautifulSoup(resp.text, "lxml")
        except Exception as e:
            logger.warning(f"    Failed to fetch {url}: {e}")
            return cases

        # Find case links on the page
        case_links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            full_url = urljoin(url, href)
            if self._is_justia_case_url(full_url) and full_url not in seen_urls:
                case_links.append((full_url, a.get_text(strip=True)))
                seen_urls.add(full_url)
            if len(case_links) >= max_per_page:
                break

        for case_url, link_text in case_links:
            if len(cases) >= max_per_page:
                break
            case = self._scrape_justia_case(case_url, link_text)
            if case:
                cases.append(case)
                self.save_case(case)
                self._sleep()

        return cases

    def _scrape_justia_case(self, url: str, title_hint: str = "") -> Optional[Dict]:
        """Scrape a single Justia case page."""
        source_id = f"justia_{self._url_to_id(url)}"
        if self.already_scraped(source_id):
            return None

        try:
            resp = self._get(url)
            soup = BeautifulSoup(resp.text, "lxml")
        except Exception as e:
            logger.debug(f"    Failed to fetch {url}: {e}")
            return None

        # Extract title
        title = ""
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)
        if not title:
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True).split("|")[0].strip() if title_tag else title_hint

        # Extract main content
        text = ""
        for selector in [
            "div.has-padding-content-wrapper-medium",
            "div#casecontent",
            "div.case-content",
            "main",
            "article",
            "div.content",
        ]:
            content_div = soup.select_one(selector)
            if content_div:
                # Remove navigation, headers, footers
                for tag in content_div.select("nav, header, footer, .ad, .advertisement, script, style"):
                    tag.decompose()
                text = content_div.get_text(separator="\n", strip=True)
                if len(text) > 300:
                    break

        if len(text) < 200:
            return None

        # Extract date
        date = ""
        date_patterns = [
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}',
            r'\d{4}-\d{2}-\d{2}',
            r'\d{1,2}/\d{1,2}/\d{4}',
        ]
        for pattern in date_patterns:
            m = re.search(pattern, text[:2000])
            if m:
                date = m.group(0)
                break

        # Extract court
        court = ""
        court_patterns = [
            r'(?:Supreme Court|Court of Appeals|District Court|Circuit Court|Family Court)[^\n]*',
            r'(?:SUPREME COURT|COURT OF APPEALS)[^\n]*',
        ]
        for pattern in court_patterns:
            m = re.search(pattern, text[:1000], re.IGNORECASE)
            if m:
                court = m.group(0).strip()[:200]
                break

        return {
            "source_id": source_id,
            "source_name": "Justia",
            "title": title[:500],
            "url": url,
            "date": date,
            "court": court,
            "text": text,
            "metadata": {"scraped_from": "justia"},
        }

    # ── BAILII (British and Irish) ────────────────────────
    def _scrape_bailii(self, max_cases: int = 50) -> List[Dict]:
        """
        Scrape divorce-related cases from BAILII (British and Irish Legal Info).
        Uses BAILII's free search interface.
        """
        cases = []
        for keyword in ["divorce", "ancillary relief", "financial remedy"]:
            if len(cases) >= max_cases:
                break
            try:
                url = "https://www.bailii.org/cgi-bin/find_cases.pl"
                params = {
                    "query": f"{keyword} family law",
                    "method": "boolean",
                    "courts": "All",
                    "startrecord": "1",
                }
                resp = self._get(url, params=params)
                keyword_cases = self._parse_bailii_results(resp.text, max_per_keyword=15)
                cases.extend(keyword_cases)
                self._sleep()
            except Exception as e:
                logger.warning(f"  BAILII search failed for '{keyword}': {e}")

        return cases[:max_cases]

    def _parse_bailii_results(self, html: str, max_per_keyword: int = 15) -> List[Dict]:
        """Parse BAILII search results page."""
        cases = []
        soup = BeautifulSoup(html, "lxml")
        base_url = "https://www.bailii.org"

        # Find case links in results
        for a in soup.find_all("a", href=True):
            if len(cases) >= max_per_keyword:
                break
            href = a["href"]
            if self._is_bailii_case_url(href):
                full_url = urljoin(base_url, href)
                case = self._scrape_bailii_case(full_url)
                if case:
                    cases.append(case)
                    self.save_case(case)
                    self._sleep()

        return cases

    def _scrape_bailii_case(self, url: str) -> Optional[Dict]:
        """Scrape a single BAILII case."""
        source_id = f"bailii_{self._url_to_id(url)}"
        if self.already_scraped(source_id):
            return None

        try:
            resp = self._get(url)
            soup = BeautifulSoup(resp.text, "lxml")
        except Exception as e:
            logger.debug(f"    Failed to fetch {url}: {e}")
            return None

        # Extract title
        title = ""
        for tag in soup.find_all(["h1", "h2", "title"]):
            t = tag.get_text(strip=True)
            if len(t) > 10:
                title = t[:300]
                break

        # Extract main text
        body = soup.find("body")
        if not body:
            return None
        for tag in body.select("nav, header, footer, script, style, .ads"):
            tag.decompose()
        text = body.get_text(separator="\n", strip=True)

        if len(text) < 300:
            return None

        # Extract date
        date = ""
        m = re.search(r'\[\d{4}\]', title)
        if m:
            date = m.group(0)[1:-1]

        return {
            "source_id": source_id,
            "source_name": "BAILII",
            "title": title,
            "url": url,
            "date": date,
            "court": self._extract_bailii_court(url),
            "text": text,
            "metadata": {"jurisdiction": "UK/Ireland", "scraped_from": "bailii"},
        }

    # ── Utilities ─────────────────────────────────────────
    @staticmethod
    def _url_to_id(url: str) -> str:
        """Convert a URL to a filesystem-safe ID."""
        parsed = urlparse(url)
        path = parsed.path.strip("/").replace("/", "_").replace(".", "_")
        return (parsed.netloc.split(".")[0] + "_" + path)[:100]

    @staticmethod
    def _is_justia_case_url(url: str) -> bool:
        """Check if a URL looks like a Justia case page."""
        return (
            "law.justia.com" in url
            and any(seg in url for seg in ["/cases/", "/us/", "/federal/", "/state/"])
            and not url.endswith(".pdf")
        )

    @staticmethod
    def _is_bailii_case_url(href: str) -> bool:
        """Check if a href looks like a BAILII case (not a page/nav link)."""
        if not href.endswith(".html"):
            return False
        # Must contain 'cases' in the path and look like a case number (digits)
        import re
        return (
            ("bailii.org" in href or href.startswith("/"))
            and "/cases/" in href
            and bool(re.search(r'/\d+\.html$', href))
        )

    @staticmethod
    def _extract_bailii_court(url: str) -> str:
        """Extract court name from BAILII URL path."""
        parts = url.split("/")
        court_map = {
            "ewca": "England and Wales Court of Appeal",
            "ewhc": "England and Wales High Court",
            "uksc": "UK Supreme Court",
            "ukhl": "UK House of Lords",
            "nica": "Court of Appeal in Northern Ireland",
            "niqb": "High Court of Justice in Northern Ireland",
        }
        for part in parts:
            for code, name in court_map.items():
                if code in part.lower():
                    return name
        return "UK/Irish Court"
