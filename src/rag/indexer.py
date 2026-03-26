"""
Indexer — loads processed document files into the vector store.

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
    Loads processed document JSON files and indexes them into ChromaDB.

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
        Index all processed documents.

        Args:
            force_rebuild: If True, delete existing collections and reindex.

        Returns:
            Dict of {collection_name: count_added}.
        """
        if force_rebuild:
            logger.info("Force rebuild: deleting existing collections...")
            self.store.reset_all()

        existing_counts = self.store.stats()
        total_existing = sum(existing_counts.values())

        if total_existing > 0 and not force_rebuild:
            logger.info(
                f"Found existing index: {existing_counts}. Use force_rebuild=True to rebuild."
            )
            return existing_counts

        documents = self.load_processed_documents()
        if not documents:
            logger.warning(f"No processed documents found in {self.processed_dir}.")
            return {}

        logger.info(f"Indexing {len(documents)} documents into vector store...")
        counts = self.store.add_from_documents(documents)
        logger.info(f"Indexing complete: {counts}")
        return counts

    def index_new(self) -> Dict[str, int]:
        """Index only documents not yet in the store (incremental updates)."""
        documents = self.load_processed_documents()
        if not documents:
            return {}

        existing_ids = self._get_indexed_source_ids()
        new_docs = [d for d in documents if d.get("source_id") not in existing_ids]

        if not new_docs:
            logger.info("No new documents to index.")
            return {}

        logger.info(f"Indexing {len(new_docs)} new documents...")
        return self.store.add_from_documents(new_docs)

    def load_processed_documents(self) -> List[Dict]:
        """Load all processed document JSON files from disk."""
        documents = []
        for path in sorted(self.processed_dir.rglob("*.json")):
            try:
                documents.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

        logger.info(f"Loaded {len(documents)} processed documents from {self.processed_dir}")
        return documents

    # backward-compat alias
    def load_processed_cases(self) -> List[Dict]:
        return self.load_processed_documents()

    def _get_indexed_source_ids(self) -> set:
        """Get source_ids already present in the documents collection."""
        try:
            store = self.store.get_collection("documents")
            result = store._collection.get(include=["metadatas"])
            return {
                m.get("source_id")
                for m in result.get("metadatas", [])
                if m.get("source_id")
            }
        except Exception:
            return set()

    def index_documents_directly(self, documents: List[Dict]) -> Dict[str, int]:
        """Index a list of document dicts directly (without loading from disk)."""
        logger.info(f"Indexing {len(documents)} documents directly...")
        return self.store.add_from_documents(documents)
