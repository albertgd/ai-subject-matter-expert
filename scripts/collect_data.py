"""
collect_data.py — Run all scrapers to collect public court cases.

Usage:
    python scripts/collect_data.py                   # all sources, 500 cases each
    python scripts/collect_data.py --max 50          # limit per source
    python scripts/collect_data.py --courtlistener   # only CourtListener
    python scripts/collect_data.py --caselaw         # only Harvard Caselaw Access
    python scripts/collect_data.py --web             # only Justia/BAILII
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import MAX_CASES_PER_SOURCE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("collect_data")


def run_courtlistener(max_cases: int):
    logger.info(f"=== CourtListener scraper (max {max_cases}) ===")
    from src.scrapers.courtlistener_scraper import CourtListenerScraper
    scraper = CourtListenerScraper()
    cases = scraper.scrape(max_cases=max_cases)
    logger.info(f"CourtListener: {len(cases)} new cases collected.")
    return len(cases)


def run_caselaw_access(max_cases: int):
    logger.info(f"=== Harvard Caselaw Access scraper (max {max_cases}) ===")
    from src.scrapers.caselaw_scraper import CaseLawAccessScraper
    scraper = CaseLawAccessScraper()
    cases = scraper.scrape(max_cases=max_cases)
    logger.info(f"Caselaw Access: {len(cases)} new cases collected.")
    return len(cases)


def run_web_scraper(max_cases: int):
    logger.info(f"=== Web scraper (Justia/BAILII) (max {max_cases}) ===")
    from src.scrapers.web_scraper import WebScraper
    scraper = WebScraper()
    cases = scraper.scrape(max_cases=max_cases)
    logger.info(f"Web scraper: {len(cases)} new cases collected.")
    return len(cases)


def main():
    parser = argparse.ArgumentParser(description="Collect court cases from public sources.")
    parser.add_argument("--max", type=int, default=MAX_CASES_PER_SOURCE,
                        help="Maximum cases per source")
    parser.add_argument("--courtlistener", action="store_true",
                        help="Run CourtListener scraper only")
    parser.add_argument("--caselaw", action="store_true",
                        help="Run Harvard Caselaw Access scraper only")
    parser.add_argument("--web", action="store_true",
                        help="Run web scraper (Justia/BAILII) only")
    args = parser.parse_args()

    # If no specific source specified, run all
    run_all = not (args.courtlistener or args.caselaw or args.web)
    total = 0

    if run_all or args.courtlistener:
        try:
            total += run_courtlistener(args.max)
        except Exception as e:
            logger.error(f"CourtListener failed: {e}")

    if run_all or args.caselaw:
        try:
            total += run_caselaw_access(args.max)
        except Exception as e:
            logger.error(f"Caselaw Access failed: {e}")

    if run_all or args.web:
        try:
            total += run_web_scraper(args.max)
        except Exception as e:
            logger.error(f"Web scraper failed: {e}")

    print(f"\nTotal new cases collected: {total}")
    print(f"Raw data saved to: {ROOT / 'data' / 'raw'}")
    print("\nNext step: python scripts/process_data.py")


if __name__ == "__main__":
    main()
