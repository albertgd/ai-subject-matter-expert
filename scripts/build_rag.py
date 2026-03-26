"""
build_rag.py — Build the ChromaDB vector knowledge base from processed documents.

Reads all JSON files from data/processed/ and indexes them into three
ChromaDB collections:
  - {subject}_documents : full content chunks
  - {subject}_learnings : distilled key insights
  - {subject}_summaries : concise document overviews

Usage:
    python scripts/build_rag.py             # incremental (skip existing)
    python scripts/build_rag.py --rebuild   # delete and rebuild from scratch
    python scripts/build_rag.py --stats     # just show current stats
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("build_rag")


def main():
    parser = argparse.ArgumentParser(description="Build the RAG knowledge base.")
    parser.add_argument("--rebuild", action="store_true",
                        help="Delete existing index and rebuild from scratch")
    parser.add_argument("--stats", action="store_true",
                        help="Show current knowledge base stats and exit")
    args = parser.parse_args()

    from src.rag.vector_store import VectorStore
    from src.rag.indexer import Indexer

    store = VectorStore()
    indexer = Indexer(store=store)

    if args.stats:
        stats = store.stats()
        print("\nKnowledge Base Statistics:")
        for collection, count in stats.items():
            print(f"  {collection}: {count:,} chunks")
        print(f"  TOTAL: {sum(stats.values()):,} chunks")
        return

    print("Building RAG knowledge base...")
    print(f"Processed cases directory: {ROOT / 'data' / 'processed'}")

    if args.rebuild:
        print("Force rebuild: deleting existing collections...")
        store.reset_all()
        counts = indexer.index_all(force_rebuild=False)
    else:
        # Incremental: only index new cases
        existing = store.stats()
        if sum(existing.values()) > 0:
            print(f"Existing index found: {existing}")
            print("Running incremental indexing (new documents only)...")
            counts = indexer.index_new()
        else:
            print("No existing index. Building from scratch...")
            counts = indexer.index_all(force_rebuild=False)

    print("\nKnowledge base build complete:")
    for collection, count in counts.items():
        print(f"  {collection}: {count:,} chunks added")

    final_stats = store.stats()
    print("\nFinal knowledge base stats:")
    for collection, count in final_stats.items():
        print(f"  {collection}: {count:,} chunks")
    print(f"  TOTAL: {sum(final_stats.values()):,} chunks")
    print(f"\nKnowledge base saved to: {ROOT / 'data' / 'vector_db'}")
    print("\nYou can now start the app: streamlit run src/app.py")


if __name__ == "__main__":
    main()
