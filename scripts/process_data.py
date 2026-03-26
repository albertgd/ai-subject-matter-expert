"""
process_data.py — Clean, anonymize, and structure scraped cases.

For each raw case:
  1. Clean text (boilerplate, whitespace, encoding)
  2. Remove PII (Presidio + spaCy, with regex fallback)
  3. Structure with LLM (extract facts, ruling, reasoning, learnings)
  4. Save to data/processed/

Usage:
    python scripts/process_data.py                   # process all raw cases
    python scripts/process_data.py --limit 100       # process first N
    python scripts/process_data.py --skip-structured # skip already-processed
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


def load_raw_cases(limit: int = 0) -> list:
    """Load all raw case JSON files."""
    cases = []
    for path in sorted(RAW_DATA_DIR.rglob("*.json")):
        try:
            cases.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as e:
            logger.warning(f"Could not load {path}: {e}")
        if limit and len(cases) >= limit:
            break
    logger.info(f"Loaded {len(cases)} raw cases.")
    return cases


def already_processed(source_id: str) -> bool:
    """Check if a case has already been processed."""
    safe_id = source_id.replace("/", "_").replace(":", "_")
    return (PROCESSED_DATA_DIR / f"{safe_id}.json").exists()


def save_processed(case: dict):
    """Save a processed case to disk."""
    source_id = case.get("source_id", "unknown")
    safe_id = source_id.replace("/", "_").replace(":", "_")
    path = PROCESSED_DATA_DIR / f"{safe_id}.json"
    path.write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="Process raw scraped cases.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only first N cases (0 = all)")
    parser.add_argument("--skip-structured", action="store_true",
                        help="Skip cases already processed")
    parser.add_argument("--fast-model", action="store_true",
                        help="Use cheap/fast LLM model (Haiku / GPT-4o-mini)")
    parser.add_argument("--no-structure", action="store_true",
                        help="Skip LLM structuring (only clean + PII removal)")
    args = parser.parse_args()

    # Load components
    logger.info("Loading text cleaner...")
    from src.processors.text_cleaner import TextCleaner
    cleaner = TextCleaner()

    logger.info("Loading PII remover...")
    from src.processors.pii_remover import PIIRemover
    pii_remover = PIIRemover()

    structurer = None
    if not args.no_structure:
        logger.info("Loading case structurer...")
        from src.processors.case_structurer import CaseStructurer
        model_override = None
        if args.fast_model:
            # Use cheaper models
            import os
            if os.getenv("ANTHROPIC_API_KEY"):
                model_override = "claude-haiku-4-5-20251001"
            elif os.getenv("OPENAI_API_KEY"):
                model_override = "gpt-4o-mini"
        structurer = CaseStructurer(model_name=model_override)

    # Process cases
    cases = load_raw_cases(limit=args.limit)
    processed = 0
    skipped = 0
    failed = 0

    for i, case in enumerate(cases, 1):
        source_id = case.get("source_id", f"case_{i}")

        if args.skip_structured and already_processed(source_id):
            skipped += 1
            continue

        logger.info(f"[{i}/{len(cases)}] Processing {source_id}...")

        try:
            # Step 1: Clean text
            cleaned_text = cleaner.clean(case.get("text", ""))
            if not cleaned_text:
                logger.warning(f"  Empty after cleaning, skipping.")
                failed += 1
                continue
            case["text"] = cleaned_text

            # Step 2: Remove PII
            case = pii_remover.anonymize_case(case)

            # Step 3: LLM structuring
            if structurer:
                case = structurer.structure(case)

            # Save
            save_processed(case)
            processed += 1

        except Exception as e:
            logger.error(f"  Failed: {e}")
            failed += 1

    print(f"\nProcessing complete:")
    print(f"  Processed: {processed}")
    print(f"  Skipped:   {skipped}")
    print(f"  Failed:    {failed}")
    print(f"\nProcessed cases saved to: {PROCESSED_DATA_DIR}")
    print("\nNext step: python scripts/build_rag.py")


if __name__ == "__main__":
    main()
