"""Web scrapers for collecting public case law and legal documents."""
from .base_scraper import BaseScraper
from .courtlistener_scraper import CourtListenerScraper
from .caselaw_scraper import CaseLawAccessScraper
from .web_scraper import WebScraper

__all__ = [
    "BaseScraper",
    "CourtListenerScraper",
    "CaseLawAccessScraper",
    "WebScraper",
]
