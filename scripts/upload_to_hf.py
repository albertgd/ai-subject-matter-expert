"""
upload_to_hf.py — Upload the cleaned dataset to HuggingFace Hub.

Reads all processed JSON files, assembles a HuggingFace Dataset,
and uploads to the Hub.

Dataset fields:
  source_id, source_name, title, url, date, court, jurisdiction,
  text, facts, ruling, reasoning, learnings, summary, practice_areas,
  structured (bool)

Usage:
    python scripts/upload_to_hf.py --repo-id username/divorce-cases-en
    python scripts/upload_to_hf.py --repo-id username/repo --private
    python scripts/upload_to_hf.py --repo-id username/repo --no-full-text
    python scripts/upload_to_hf.py --local-only  # save JSONL/parquet locally only
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import PROCESSED_DATA_DIR, DOMAIN, HF_TOKEN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("upload_to_hf")

HF_OUTPUT_DIR = ROOT / "data" / "hf_dataset"
HF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_processed_cases(include_full_text: bool = True) -> list:
    """Load all processed cases and build dataset records."""
    records = []

    for path in sorted(PROCESSED_DATA_DIR.rglob("*.json")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))

            record = {
                "source_id": case.get("source_id", ""),
                "source_name": case.get("source_name", ""),
                "title": case.get("title", ""),
                "url": case.get("url", ""),
                "date": case.get("date", ""),
                "court": case.get("court", ""),
                "jurisdiction": case.get("jurisdiction", ""),
                "citations": json.dumps(case.get("citations", []) or case.get("citation", [])),
                "practice_areas": case.get("practice_areas", []),
                "facts": case.get("facts", ""),
                "ruling": case.get("ruling", ""),
                "reasoning": case.get("reasoning", ""),
                "learnings": case.get("learnings", ""),
                "summary": case.get("summary", ""),
                "structured": bool(case.get("structured", False)),
            }

            if include_full_text:
                record["text"] = case.get("text", "")

            # Skip records with no useful content
            if not any([record["facts"], record["ruling"], record["learnings"],
                        record.get("text", ""), record["summary"]]):
                continue

            records.append(record)
        except Exception as e:
            logger.warning(f"Skipping {path}: {e}")

    logger.info(f"Loaded {len(records)} records.")
    return records


def build_dataset_card(repo_id: str, record_count: int, description: str = "") -> str:
    """Generate a HuggingFace dataset card."""
    return f"""---
annotations_creators:
  - machine-generated
language:
  - en
license: cc-by-4.0
multilinguality:
  - monolingual
source_datasets:
  - original
task_categories:
  - text-classification
  - question-answering
tags:
  - legal
  - court-cases
  - {DOMAIN}
  - family-law
  - rag
  - legal-ai
pretty_name: "{DOMAIN.title()} Law Court Cases (English)"
dataset_info:
  splits:
    - name: train
      num_examples: {record_count}
---

# {DOMAIN.title()} Law Court Cases Dataset

{description or f"Public {DOMAIN} law court cases collected from free, open-source legal databases. PII-anonymized and structured with LLM."}

## Dataset Description

This dataset contains **{record_count} court cases** related to {DOMAIN} law,
collected from publicly available sources:

- **CourtListener** — US federal & state court opinions (Free Law Project)
- **Harvard Caselaw Access Project** — US historical cases
- **Justia** — US case law and legal guides
- **BAILII** — British and Irish legal decisions

### How Was It Built?

1. **Scraping** — Automated collection from public legal APIs and websites
2. **Cleaning** — Boilerplate removal, whitespace normalization, encoding fixes
3. **PII Anonymization** — Microsoft Presidio + spaCy NER; person names → `[PERSON_N]`, SSN/phone/email scrubbed
4. **Structuring** — LLM (Claude/GPT-4) extracts structured fields from each opinion
5. **Publishing** — Assembled into HuggingFace Dataset format

## Dataset Fields

| Field | Description |
|---|---|
| `source_id` | Unique ID (e.g. `cl_12345`, `cap_67890`) |
| `source_name` | Origin database (CourtListener, Justia, etc.) |
| `title` | Case name |
| `url` | Link to original opinion |
| `date` | Decision date |
| `court` | Court name |
| `jurisdiction` | Jurisdiction (US, UK, etc.) |
| `practice_areas` | List of relevant legal topics |
| `facts` | Key facts and background (PII-anonymized) |
| `ruling` | Court's final decision |
| `reasoning` | Legal reasoning and principles applied |
| `learnings` | Distilled legal principles for RAG/fine-tuning |
| `summary` | Concise 3-5 sentence summary |
| `text` | Full case text (PII-anonymized) |
| `structured` | Whether LLM extraction was applied |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", split="train")
print(ds)

# Load into a RAG system
for case in ds:
    print(case["title"])
    print(case["learnings"])
```

## License

This dataset is released under **CC BY 4.0**.
Source cases are public court records.
"""


def main():
    parser = argparse.ArgumentParser(description="Upload dataset to HuggingFace Hub.")
    parser.add_argument("--repo-id", default="",
                        help="HuggingFace repo ID (e.g. username/divorce-cases-en)")
    parser.add_argument("--private", action="store_true",
                        help="Make dataset private")
    parser.add_argument("--no-full-text", action="store_true",
                        help="Exclude full text field (saves space)")
    parser.add_argument("--local-only", action="store_true",
                        help="Save locally without uploading to HF")
    parser.add_argument("--description", default="",
                        help="Dataset description for the card")
    args = parser.parse_args()

    # Load data
    records = load_processed_cases(include_full_text=not args.no_full_text)
    if not records:
        print("ERROR: No processed cases found. Run process_data.py first.")
        sys.exit(1)

    print(f"Building HuggingFace dataset from {len(records)} records...")

    try:
        from datasets import Dataset
        ds = Dataset.from_list(records)
        print(f"Dataset schema:\n{ds}")
    except ImportError:
        print("ERROR: 'datasets' package not installed. Run: pip install datasets")
        sys.exit(1)

    # Save locally
    jsonl_path = HF_OUTPUT_DIR / f"{DOMAIN}_cases.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved JSONL → {jsonl_path}")

    ds.save_to_disk(str(HF_OUTPUT_DIR / "dataset"))
    print(f"Saved HF dataset → {HF_OUTPUT_DIR / 'dataset'}")

    if args.local_only:
        print("\nLocal-only mode. Dataset saved. Skipping HuggingFace upload.")
        return

    # Upload to HuggingFace
    if not args.repo_id:
        print("No --repo-id specified. Use --local-only to skip upload, or provide a repo ID.")
        sys.exit(1)

    token = HF_TOKEN or os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set. Add it to .env or set as environment variable.")
        sys.exit(1)

    print(f"\nUploading to HuggingFace: {args.repo_id} ...")
    try:
        ds.push_to_hub(
            repo_id=args.repo_id,
            token=token,
            private=args.private,
        )
        print(f"Dataset uploaded!")
        print(f"View at: https://huggingface.co/datasets/{args.repo_id}")
    except Exception as e:
        print(f"Upload failed: {e}")
        sys.exit(1)

    # Upload dataset card
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        card = build_dataset_card(args.repo_id, len(records), args.description)
        card_path = HF_OUTPUT_DIR / "README.md"
        card_path.write_text(card, encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
            token=token,
        )
        print("Dataset card uploaded.")
    except Exception as e:
        logger.warning(f"Could not upload dataset card: {e}")

    print(f"\nDone! {len(records)} cases uploaded to {args.repo_id}")


if __name__ == "__main__":
    main()
