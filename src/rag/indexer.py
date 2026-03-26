"""
Indexer — loads processed case files into the vector store.

Reads processed JSON files from data/processed/ and indexes them
into ChromaDB collections via VectorStore.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.config import PROCESSED_DATA_DIR
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class Indexer:
    """
    Loads processed case JSON files and indexes them into ChromaDB.

    Usage:
        indexer = Indexer()
        stats = indexer.index_all()   # index everything in processed/
        print(stats)
    """

    def __init__(
        self,
        store: Optional[VectorStore] = None,
        processed_dir: Optional[Path] = None,
    ):
        self.store = store or VectorStore()
        self.processed_dir = processed_dir or PROCESSED_DATA_DIR

    def index_all(self, force_rebuild: bool = False) -> Dict[str, int]:
        """
        Index all processed cases.

        Args:
            force_rebuild: If True, delete existing collections and reindex.

        Returns:
            Dict of {collection_name: count_added}.
        """
        if force_rebuild:
            logger.info("Force rebuild: deleting existing collections...")
            self.store.reset_all()

        # Check if already indexed
        existing_counts = self.store.stats()
        total_existing = sum(existing_counts.values())

        if total_existing > 0 and not force_rebuild:
            logger.info(f"Found existing index: {existing_counts}. Use force_rebuild=True to rebuild.")
            return existing_counts

        cases = self.load_processed_cases()
        if not cases:
            logger.warning(f"No processed cases found in {self.processed_dir}.")
            return {}

        logger.info(f"Indexing {len(cases)} cases into vector store...")
        counts = self.store.add_from_cases(cases)
        logger.info(f"Indexing complete: {counts}")
        return counts

    def index_new(self) -> Dict[str, int]:
        """
        Index only cases not yet in the store (incremental updates).
        """
        cases = self.load_processed_cases()
        if not cases:
            return {}

        # Get existing source IDs to avoid duplicates
        existing_ids = self._get_indexed_source_ids()
        new_cases = [c for c in cases if c.get("source_id") not in existing_ids]

        if not new_cases:
            logger.info("No new cases to index.")
            return {}

        logger.info(f"Indexing {len(new_cases)} new cases...")
        return self.store.add_from_cases(new_cases)

    def load_processed_cases(self) -> List[Dict]:
        """Load all processed case JSON files from disk."""
        cases = []
        # Search all subdirectories
        for path in sorted(self.processed_dir.rglob("*.json")):
            try:
                case = json.loads(path.read_text(encoding="utf-8"))
                cases.append(case)
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

        logger.info(f"Loaded {len(cases)} processed cases from {self.processed_dir}")
        return cases

    def _get_indexed_source_ids(self) -> set:
        """Get source_ids already present in the opinions collection."""
        try:
            store = self.store.get_collection("opinions")
            result = store._collection.get(include=["metadatas"])
            return {m.get("source_id") for m in result.get("metadatas", []) if m.get("source_id")}
        except Exception:
            return set()

    def index_cases_directly(self, cases: List[Dict]) -> Dict[str, int]:
        """
        Index a list of case dicts directly (without loading from disk).
        Useful for pipeline scripts that have cases in memory.
        """
        logger.info(f"Indexing {len(cases)} cases directly...")
        return self.store.add_from_cases(cases)
