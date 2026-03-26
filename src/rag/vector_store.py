"""
Vector store — ChromaDB-backed semantic search over legal cases.

Collections:
  - {domain}_opinions    : Full case chunks (facts, reasoning, ruling)
  - {domain}_learnings   : Distilled legal knowledge / principles
  - {domain}_summaries   : Case summaries for quick overview retrieval

All collections use the same multilingual sentence-transformer embeddings
so cross-lingual queries work out of the box.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from src.config import COLLECTION_NAME, EMBED_MODEL, VECTOR_DB_DIR

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages ChromaDB collections for case law retrieval.

    Usage:
        store = VectorStore()
        store.add_documents(docs, collection="opinions")
        results = store.search("custody best interests", k=5)
    """

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        embed_model: str = EMBED_MODEL,
        collection_prefix: str = COLLECTION_NAME,
        embeddings=None,  # Optional pre-built embeddings object (useful for testing)
    ):
        self.persist_dir = persist_dir or VECTOR_DB_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_prefix = collection_prefix
        self._embed_model_name = embed_model
        self._embeddings = embeddings  # If provided, skip lazy loading
        self._stores: Dict[str, object] = {}

    # ── Embeddings (lazy init) ────────────────────────────
    @property
    def embeddings(self):
        if self._embeddings is None:
            logger.info(f"Loading embedding model: {self._embed_model_name}")
            from langchain_huggingface import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(model_name=self._embed_model_name)
        return self._embeddings

    # ── Collection management ─────────────────────────────
    def _collection_name(self, suffix: str) -> str:
        return f"{self.collection_prefix}_{suffix}"

    def get_collection(self, suffix: str = "opinions"):
        """Get or create a ChromaDB collection."""
        if suffix not in self._stores:
            from langchain_chroma import Chroma
            self._stores[suffix] = Chroma(
                collection_name=self._collection_name(suffix),
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_dir),
            )
        return self._stores[suffix]

    def collection_count(self, suffix: str = "opinions") -> int:
        """Return number of documents in a collection."""
        try:
            store = self.get_collection(suffix)
            return store._collection.count()
        except Exception:
            return 0

    # ── Write ─────────────────────────────────────────────
    def add_documents(
        self,
        documents: List[Document],
        collection: str = "opinions",
        batch_size: int = 100,
    ) -> int:
        """
        Add documents to a collection in batches.

        Returns number of documents added.
        """
        if not documents:
            return 0

        from langchain_chroma import Chroma

        col_name = self._collection_name(collection)
        persist_dir = str(self.persist_dir)

        # Check if the collection already has documents
        existing_store = self.get_collection(collection)
        try:
            existing_count = existing_store._collection.count()
        except Exception:
            existing_count = 0

        added = 0

        if existing_count == 0:
            # Use from_documents for the first population — more reliable initialization
            first_batch = documents[:batch_size]
            try:
                new_store = Chroma.from_documents(
                    documents=first_batch,
                    embedding=self.embeddings,
                    collection_name=col_name,
                    persist_directory=persist_dir,
                )
                self._stores[collection] = new_store
                added = len(first_batch)
                logger.info(f"  Created collection '{col_name}' with {added} docs.")
            except Exception as e:
                logger.error(f"  Failed to create collection '{col_name}': {e}")
                return 0
            remaining = documents[batch_size:]
        else:
            remaining = documents

        # Add remaining batches incrementally
        store = self._stores[collection]
        for i in range(0, len(remaining), batch_size):
            batch = remaining[i:i + batch_size]
            try:
                store.add_documents(batch)
                added += len(batch)
                logger.info(f"  Added batch: {added}/{len(documents)} docs to '{col_name}'")
            except Exception as e:
                logger.error(f"  Failed to add batch to '{col_name}': {e}")

        return added

    def add_from_cases(self, cases: List[Dict]) -> Dict[str, int]:
        """
        Index a list of processed case dicts into all collections.

        Returns dict of {collection: count_added}.
        """
        opinion_docs = []
        learning_docs = []
        summary_docs = []

        for case in cases:
            source_id = case.get("source_id", "unknown")
            base_meta = {
                "source_id": source_id,
                "source_name": case.get("source_name", ""),
                "title": case.get("title", "")[:500],
                "url": case.get("url", ""),
                "date": case.get("date", ""),
                "court": case.get("court", ""),
                "practice_areas": ",".join(case.get("practice_areas", [])),
            }

            title = case.get("title", source_id)

            # Opinion chunks: facts + reasoning + ruling
            case_opinion_count = 0
            for section in ["facts", "reasoning", "ruling"]:
                content = case.get(section, "").strip()
                if content and len(content) > 30:
                    opinion_docs.append(Document(
                        page_content=f"[{section.upper()} — {title}]\n\n{content}",
                        metadata={**base_meta, "section": section},
                    ))
                    case_opinion_count += 1

            # Fallback: use full text if no structured sections were added
            if case_opinion_count == 0:
                full_text = case.get("text", "").strip()
                if full_text:
                    # Chunk the full text
                    chunks = self._chunk_text(full_text, max_chars=2000, overlap=200)
                    for j, chunk in enumerate(chunks):
                        opinion_docs.append(Document(
                            page_content=f"[TEXT — {title}]\n\n{chunk}",
                            metadata={**base_meta, "section": "text", "chunk": j},
                        ))

            # Learnings
            learnings = case.get("learnings", "").strip()
            if learnings and len(learnings) > 50:
                learning_docs.append(Document(
                    page_content=f"[LEARNINGS — {title}]\n\n{learnings}",
                    metadata={**base_meta, "section": "learnings"},
                ))

            # Summary
            summary = case.get("summary", "").strip()
            if summary and len(summary) > 50:
                summary_docs.append(Document(
                    page_content=f"[SUMMARY — {title}]\n\n{summary}",
                    metadata={**base_meta, "section": "summary"},
                ))

        counts = {}
        if opinion_docs:
            counts["opinions"] = self.add_documents(opinion_docs, collection="opinions")
        if learning_docs:
            counts["learnings"] = self.add_documents(learning_docs, collection="learnings")
        if summary_docs:
            counts["summaries"] = self.add_documents(summary_docs, collection="summaries")

        return counts

    # ── Search ────────────────────────────────────────────
    def search(
        self,
        query: str,
        k: int = 5,
        collection: str = "opinions",
        filter_dict: Optional[Dict] = None,
    ) -> List[Document]:
        """Semantic similarity search."""
        store = self.get_collection(collection)
        return store.similarity_search(query, k=k, filter=filter_dict)

    def search_with_score(
        self,
        query: str,
        k: int = 5,
        collection: str = "opinions",
        filter_dict: Optional[Dict] = None,
    ) -> List[Tuple[Document, float]]:
        """Semantic similarity search with relevance scores."""
        store = self.get_collection(collection)
        return store.similarity_search_with_score(query, k=k, filter=filter_dict)

    def multi_search(
        self,
        query: str,
        k_opinions: int = 4,
        k_learnings: int = 3,
        k_summaries: int = 2,
    ) -> Dict[str, List[Tuple[Document, float]]]:
        """Search across all collections and return merged results."""
        results = {}
        for collection, k in [
            ("opinions", k_opinions),
            ("learnings", k_learnings),
            ("summaries", k_summaries),
        ]:
            if self.collection_count(collection) > 0:
                results[collection] = self.search_with_score(query, k=k, collection=collection)
        return results

    def delete_collection(self, suffix: str):
        """Delete a collection (useful for rebuilding)."""
        store = self.get_collection(suffix)
        store.delete_collection()
        self._stores.pop(suffix, None)
        logger.info(f"Deleted collection: {self._collection_name(suffix)}")

    def reset_all(self):
        """Delete all collections (full rebuild)."""
        for suffix in ["opinions", "learnings", "summaries"]:
            try:
                self.delete_collection(suffix)
            except Exception:
                pass

    # ── Utilities ─────────────────────────────────────────
    @staticmethod
    def _chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= max_chars:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            # Try to break at sentence boundary
            if end < len(text):
                for sep in [". ", ".\n", "\n\n", "\n"]:
                    pos = text.rfind(sep, start + max_chars // 2, end)
                    if pos != -1:
                        end = pos + len(sep)
                        break
            chunks.append(text[start:end].strip())
            next_start = end - overlap
            # Guard against infinite loop: always advance by at least 1
            if next_start <= start:
                next_start = start + 1
            start = next_start

        return [c for c in chunks if c]

    def stats(self) -> Dict[str, int]:
        """Return document counts for all collections."""
        return {
            suffix: self.collection_count(suffix)
            for suffix in ["opinions", "learnings", "summaries"]
        }
