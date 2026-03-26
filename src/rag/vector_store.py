"""
Vector store — ChromaDB-backed semantic search over any subject's documents.

Collections:
  - {subject}_documents : Full document chunks (content)
  - {subject}_learnings : Distilled knowledge / key insights
  - {subject}_summaries : Document summaries for quick overview retrieval

All collections use multilingual sentence-transformer embeddings
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
    Manages ChromaDB collections for subject-matter document retrieval.

    Usage:
        store = VectorStore()
        store.add_documents(docs, collection="documents")
        results = store.search("what is backpropagation", k=5)
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

    def get_collection(self, suffix: str = "documents"):
        """Get or create a ChromaDB collection."""
        if suffix not in self._stores:
            from langchain_chroma import Chroma
            self._stores[suffix] = Chroma(
                collection_name=self._collection_name(suffix),
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_dir),
            )
        return self._stores[suffix]

    def collection_count(self, suffix: str = "documents") -> int:
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
        collection: str = "documents",
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

        existing_store = self.get_collection(collection)
        try:
            existing_count = existing_store._collection.count()
        except Exception:
            existing_count = 0

        added = 0

        if existing_count == 0:
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

    def add_from_documents(self, documents: List[Dict]) -> Dict[str, int]:
        """
        Index a list of processed document dicts into all collections.

        Document fields used:
          - text        → chunked into 'documents' collection
          - key_points  → 'learnings' collection
          - learnings   → appended to 'learnings' collection
          - summary     → 'summaries' collection

        Returns dict of {collection: count_added}.
        """
        content_docs = []
        learning_docs = []
        summary_docs = []

        for doc in documents:
            source_id = doc.get("source_id", "unknown")
            base_meta = {
                "source_id": source_id,
                "source_name": doc.get("source_name", ""),
                "title": doc.get("title", "")[:500],
                "url": doc.get("url", ""),
                "date": doc.get("date", ""),
                "author": doc.get("author", ""),
                "topics": ",".join(doc.get("topics", [])),
            }

            title = doc.get("title", source_id)

            # Content: full text, chunked
            text = doc.get("text", "").strip()
            if text and len(text) > 30:
                chunks = self._chunk_text(text, max_chars=2000, overlap=200)
                for j, chunk in enumerate(chunks):
                    content_docs.append(Document(
                        page_content=f"[{title}]\n\n{chunk}",
                        metadata={**base_meta, "section": "content", "chunk": j},
                    ))

            # Learnings: key_points + learnings fields
            learnings_parts = []
            key_points = doc.get("key_points", "").strip()
            learnings = doc.get("learnings", "").strip()
            if key_points and len(key_points) > 30:
                learnings_parts.append(key_points)
            if learnings and len(learnings) > 30:
                learnings_parts.append(learnings)

            if learnings_parts:
                combined = "\n\n".join(learnings_parts)
                learning_docs.append(Document(
                    page_content=f"[KEY INSIGHTS — {title}]\n\n{combined}",
                    metadata={**base_meta, "section": "learnings"},
                ))

            # Summary
            summary = doc.get("summary", "").strip()
            if summary and len(summary) > 50:
                summary_docs.append(Document(
                    page_content=f"[SUMMARY — {title}]\n\n{summary}",
                    metadata={**base_meta, "section": "summary"},
                ))

        counts = {}
        if content_docs:
            counts["documents"] = self.add_documents(content_docs, collection="documents")
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
        collection: str = "documents",
        filter_dict: Optional[Dict] = None,
    ) -> List[Document]:
        """Semantic similarity search."""
        store = self.get_collection(collection)
        return store.similarity_search(query, k=k, filter=filter_dict)

    def search_with_score(
        self,
        query: str,
        k: int = 5,
        collection: str = "documents",
        filter_dict: Optional[Dict] = None,
    ) -> List[Tuple[Document, float]]:
        """Semantic similarity search with relevance scores."""
        store = self.get_collection(collection)
        return store.similarity_search_with_score(query, k=k, filter=filter_dict)

    def multi_search(
        self,
        query: str,
        k_documents: int = 4,
        k_learnings: int = 3,
        k_summaries: int = 2,
    ) -> Dict[str, List[Tuple[Document, float]]]:
        """Search across all collections and return merged results."""
        results = {}
        for collection, k in [
            ("documents", k_documents),
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
        for suffix in ["documents", "learnings", "summaries"]:
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
            if end < len(text):
                for sep in [". ", ".\n", "\n\n", "\n"]:
                    pos = text.rfind(sep, start + max_chars // 2, end)
                    if pos != -1:
                        end = pos + len(sep)
                        break
            chunks.append(text[start:end].strip())
            next_start = end - overlap
            if next_start <= start:
                next_start = start + 1
            start = next_start

        return [c for c in chunks if c]

    def stats(self) -> Dict[str, int]:
        """Return document counts for all collections."""
        return {
            suffix: self.collection_count(suffix)
            for suffix in ["documents", "learnings", "summaries"]
        }
