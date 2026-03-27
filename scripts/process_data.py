"""
process_data.py — Clean, anonymize, and structure raw documents.

For each raw document:
  1. Clean text (boilerplate, whitespace, encoding)
  2. Remove PII (Presidio + spaCy, with regex fallback)
  3. Structure with LLM (extract summary, key_points, learnings, topics)
  4. Save to data/processed/

Usage:
    python scripts/process_data.py                   # process only new raw docs (default)
    python scripts/process_data.py --reprocess       # force reprocess all docs, even existing ones
    python scripts/process_data.py --limit 100       # process first N
    python scripts/process_data.py --fast-model      # use cheaper LLM
    python scripts/process_data.py --no-structure    # skip LLM structuring
"""

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("process_data")


def load_raw_documents(limit: int = 0) -> list:
    """Load all raw document JSON files."""
    docs = []
    for path in sorted(RAW_DATA_DIR.rglob("*.json")):
        try:
            docs.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as e:
            logger.warning(f"Could not load {path}: {e}")
        if limit and len(docs) >= limit:
            break
    logger.info(f"Loaded {len(docs)} raw documents.")
    return docs


def already_processed(source_id: str) -> bool:
    safe_id = source_id.replace("/", "_").replace(":", "_")
    return (PROCESSED_DATA_DIR / f"{safe_id}.json").exists()


def save_processed(doc: dict):
    source_id = doc.get("source_id", "unknown")
    safe_id = source_id.replace("/", "_").replace(":", "_")
    path = PROCESSED_DATA_DIR / f"{safe_id}.json"
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="Process raw documents.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only first N documents (0 = all)")
    parser.add_argument("--reprocess", action="store_true",
                        help="Force reprocess all docs, even ones already in data/processed/")
    parser.add_argument("--fast-model", action="store_true",
                        help="Use cheap/fast LLM model (Haiku / GPT-4o-mini)")
    parser.add_argument("--no-structure", action="store_true",
                        help="Skip LLM structuring (only clean + PII removal)")
    args = parser.parse_args()

    logger.info("Loading text cleaner...")
    from src.processors.text_cleaner import TextCleaner
    cleaner = TextCleaner()

    logger.info("Loading PII remover...")
    from src.processors.pii_remover import PIIRemover
    pii_remover = PIIRemover()

    structurer = None
    if not args.no_structure:
        logger.info("Loading content structurer...")
        from src.processors.content_structurer import ContentStructurer
        model_override = None
        if args.fast_model:
            import os
            if os.getenv("ANTHROPIC_API_KEY"):
                model_override = "claude-haiku-4-5-20251001"
            elif os.getenv("OPENAI_API_KEY"):
                model_override = "gpt-4o-mini"
        structurer = ContentStructurer(model_name=model_override)

    docs = load_raw_documents(limit=args.limit)
    processed = skipped = failed = 0

    for i, doc in enumerate(docs, 1):
        source_id = doc.get("source_id", f"doc_{i}")

        if not args.reprocess and already_processed(source_id):
            skipped += 1
            continue

        logger.info(f"[{i}/{len(docs)}] Processing {source_id}...")

        try:
            # Step 1: Clean text
            cleaned_text = cleaner.clean(doc.get("text", ""))
            if not cleaned_text:
                logger.warning("  Empty after cleaning, skipping.")
                failed += 1
                continue
            doc["text"] = cleaned_text

            # Step 2: Remove PII
            doc = pii_remover.anonymize_document(doc)

            # Step 3: LLM structuring
            if structurer:
                doc = structurer.structure(doc)

            save_processed(doc)
            processed += 1

        except Exception as e:
            logger.error(f"  Failed: {e}")
            failed += 1

    print(f"\nProcessing complete:")
    print(f"  Processed: {processed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {failed}")
    print(f"\nProcessed documents saved to: {PROCESSED_DATA_DIR}")
    print("\nNext step: python scripts/build_rag.py")


if __name__ == "__main__":
    main()
