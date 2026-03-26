"""
Retriever — unified interface for retrieving relevant documents from the vector store.

Combines results from multiple ChromaDB collections (documents, learnings, summaries)
and returns a structured context ready for the SME agent's LLM prompt.
"""

import logging
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from src.config import DEFAULT_K_DOCS, DEFAULT_K_LEARNINGS, SUBJECT
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieves relevant documents for a given query.

    Searches across:
      - documents collection: full content chunks
      - learnings collection: distilled key insights
      - summaries collection: concise document overviews

    Usage:
        retriever = Retriever()
        context, sources = retriever.retrieve("what is backpropagation")
    """

    def __init__(
        self,
        store: Optional[VectorStore] = None,
        k_documents: int = DEFAULT_K_DOCS,
        k_learnings: int = DEFAULT_K_LEARNINGS,
        k_summaries: int = 2,
    ):
        self.store = store or VectorStore()
        self.k_documents = k_documents
        self.k_learnings = k_learnings
        self.k_summaries = k_summaries

    def retrieve(
        self,
        query: str,
        topic_filter: Optional[str] = None,
        verbose: bool = False,
    ) -> Tuple[str, List[Dict]]:
        """
        Retrieve relevant documents and format as context string.

        Args:
            query:        The user's question or search query.
            topic_filter: Optional topic tag to filter results.
            verbose:      Log retrieved sources to console.

        Returns:
            (context_string, sources_list)
              - context_string: Formatted text ready to inject into LLM prompt
              - sources_list:   List of source metadata dicts for UI display
        """
        filter_dict = {"topics": topic_filter} if topic_filter else None

        all_results = self.store.multi_search(
            query=query,
            k_documents=self.k_documents,
            k_learnings=self.k_learnings,
            k_summaries=self.k_summaries,
        )

        context_parts = []
        sources = []

        # ── Learnings first (highest signal) ─────────────
        learning_results = all_results.get("learnings", [])
        if learning_results:
            context_parts.append(f"## KEY INSIGHTS ABOUT {SUBJECT.upper()}\n")
            for doc, score in learning_results:
                meta = doc.metadata
                context_parts.append(
                    f"**Source: {meta.get('title', 'Unknown')}**\n"
                    f"From: {meta.get('source_name', '')} | "
                    f"Date: {meta.get('date', '')}\n\n"
                    f"{doc.page_content}\n\n---\n"
                )
                sources.append(self._make_source(meta, score, "learnings"))

        # ── Document chunks ───────────────────────────────
        doc_results = all_results.get("documents", [])
        if doc_results:
            context_parts.append("## RELEVANT CONTENT\n")
            for doc, score in doc_results:
                meta = doc.metadata
                context_parts.append(
                    f"**{meta.get('title', 'Unknown')}**\n"
                    f"From: {meta.get('source_name', '')} | "
                    f"Date: {meta.get('date', '')}\n\n"
                    f"{doc.page_content}\n\n---\n"
                )
                sources.append(self._make_source(meta, score, "documents"))

        # ── Summaries ─────────────────────────────────────
        summary_results = all_results.get("summaries", [])
        if summary_results:
            context_parts.append("## DOCUMENT SUMMARIES\n")
            for doc, score in summary_results:
                meta = doc.metadata
                context_parts.append(
                    f"**{meta.get('title', 'Unknown')}** — {meta.get('date', '')}\n\n"
                    f"{doc.page_content}\n\n---\n"
                )
                sources.append(self._make_source(meta, score, "summary"))

        if verbose:
            logger.info(f"\nRetrieved {len(sources)} sources for query: {query[:80]}")
            for s in sources:
                logger.info(f"  [{s['collection']}] {s['title'][:60]} (score={s['score']:.3f})")

        context = (
            "\n".join(context_parts)
            if context_parts
            else "No relevant documents found in the knowledge base."
        )
        return context, sources

    def retrieve_for_topic(self, topic: str, n_results: int = 10) -> List[Dict]:
        """
        Retrieve top N documents for a broad topic (for browsing/exploration).
        Returns list of source metadata dicts sorted by relevance.
        """
        results = self.store.search_with_score(topic, k=n_results, collection="summaries")
        if not results:
            results = self.store.search_with_score(topic, k=n_results, collection="documents")

        sources = []
        seen_ids = set()
        for doc, score in results:
            sid = doc.metadata.get("source_id", "")
            if sid not in seen_ids:
                seen_ids.add(sid)
                sources.append(self._make_source(doc.metadata, score, "search"))

        return sources

    # ── Private ───────────────────────────────────────────
    @staticmethod
    def _make_source(meta: Dict, score: float, collection: str) -> Dict:
        return {
            "source_id": meta.get("source_id", ""),
            "title": meta.get("title", "Unknown"),
            "source_name": meta.get("source_name", ""),
            "url": meta.get("url", ""),
            "date": meta.get("date", ""),
            "author": meta.get("author", ""),
            "topics": meta.get("topics", ""),
            "score": round(float(score), 3),
            "collection": collection,
        }

    def is_ready(self) -> bool:
        """Check if the knowledge base has any indexed documents."""
        return sum(self.store.stats().values()) > 0

    def knowledge_base_stats(self) -> Dict[str, int]:
        """Return collection sizes."""
        return self.store.stats()
