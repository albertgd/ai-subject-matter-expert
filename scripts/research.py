"""
Research script — gather public information about the configured subject.

Uses AI to generate smart search queries, then fetches web pages
and Wikipedia articles to build the raw document corpus.

Usage:
    python scripts/research.py                   # all sources
    python scripts/research.py --web             # web only (AI researcher)
    python scripts/research.py --wikipedia       # Wikipedia only
    python scripts/research.py --max 100         # limit per source
    python scripts/research.py --queries 15      # number of AI-generated queries
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import SUBJECT, MAX_DOCS_PER_SOURCE, LANGUAGE, REGION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description=f"Research '{SUBJECT}' from public internet sources"
    )
    parser.add_argument("--web", action="store_true", help="Run AI web researcher")
    parser.add_argument("--wikipedia", action="store_true", help="Scrape Wikipedia")
    parser.add_argument(
        "--max", type=int, default=MAX_DOCS_PER_SOURCE,
        help="Max documents per source (default: %(default)s)"
    )
    parser.add_argument(
        "--queries", type=int, default=10,
        help="Number of AI-generated search queries (default: %(default)s)"
    )
    parser.add_argument(
        "--language", type=str, default=LANGUAGE,
        help="Language for research: 'en' or 'es' (default: %(default)s)"
    )
    parser.add_argument(
        "--region", type=str, default=REGION,
        help="Region filter: anywhere | spain | catalunya | europe (default: %(default)s)"
    )
    args = parser.parse_args()

    # Default: run all sources
    run_all = not (args.web or args.wikipedia)

    total = 0

    logger.info(f"Language: {args.language} | Region: {args.region}")

    if run_all or args.wikipedia:
        logger.info(f"=== Wikipedia: gathering articles about '{SUBJECT}' ===")
        try:
            from src.research.wikipedia import WikipediaSource
            wiki = WikipediaSource(language=args.language, region=args.region)
            docs = wiki.collect(max_docs=args.max)
            logger.info(f"Wikipedia: {len(docs)} documents collected")
            total += len(docs)
        except Exception as e:
            logger.error(f"Wikipedia source failed: {e}")

    if run_all or args.web:
        logger.info(f"=== AI Web Researcher: searching for '{SUBJECT}' ===")
        try:
            from src.research.ai_researcher import AIResearcher
            researcher = AIResearcher(language=args.language, region=args.region)
            docs = researcher.research(
                subject=SUBJECT,
                n_queries=args.queries,
                max_docs=args.max,
            )
            logger.info(f"AI Researcher: {len(docs)} documents collected")
            total += len(docs)
        except Exception as e:
            logger.error(f"AI Researcher failed: {e}")

    logger.info(f"\nResearch complete. Total documents collected: {total}")
    logger.info("Next step: python scripts/process_data.py")


if __name__ == "__main__":
    main()
