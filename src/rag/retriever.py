"""
Retriever — unified interface for retrieving relevant documents from the vector store.

Combines results from multiple ChromaDB collections (opinions, learnings, summaries)
and returns a structured context ready for the SME agent's LLM prompt.
"""

import logging
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from src.config import DEFAULT_K_CASES, DEFAULT_K_PRINCIPLES
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieves relevant legal documents for a given query.

    Searches across:
      - opinions collection: case facts, reasoning, rulings
      - learnings collection: distilled legal principles
      - summaries collection: concise case overviews

    Usage:
        retriever = Retriever()
        context, sources = retriever.retrieve("child custody best interests standard")
    """

    def __init__(
        self,
        store: Optional[VectorStore] = None,
        k_opinions: int = DEFAULT_K_CASES,
        k_learnings: int = DEFAULT_K_PRINCIPLES,
        k_summaries: int = 2,
    ):
        self.store = store or VectorStore()
        self.k_opinions = k_opinions
        self.k_learnings = k_learnings
        self.k_summaries = k_summaries

    def retrieve(
        self,
        query: str,
        practice_area: Optional[str] = None,
        verbose: bool = False,
    ) -> Tuple[str, List[Dict]]:
        """
        Retrieve relevant documents and format as context string.

        Args:
            query:          The user's question or search query.
            practice_area:  Optional filter (e.g. "custody", "alimony").
            verbose:        Log retrieved sources to console.

        Returns:
            (context_string, sources_list)
              - context_string: Formatted text ready to inject into LLM prompt
              - sources_list:   List of source metadata dicts for UI display
        """
        filter_dict = {"practice_areas": practice_area} if practice_area else None

        all_results = self.store.multi_search(
            query=query,
            k_opinions=self.k_opinions,
            k_learnings=self.k_learnings,
            k_summaries=self.k_summaries,
        )

        context_parts = []
        sources = []

        # ── Learnings first (highest signal) ─────────────
        learning_results = all_results.get("learnings", [])
        if learning_results:
            context_parts.append("## LEGAL PRINCIPLES & LEARNINGS\n")
            for doc, score in learning_results:
                meta = doc.metadata
                context_parts.append(
                    f"**Case: {meta.get('title', 'Unknown')}**\n"
                    f"Source: {meta.get('source_name', '')} | "
                    f"Date: {meta.get('date', '')} | "
                    f"Court: {meta.get('court', '')}\n\n"
                    f"{doc.page_content}\n\n---\n"
                )
                sources.append(self._make_source(meta, score, "learnings"))

        # ── Opinion chunks ────────────────────────────────
        opinion_results = all_results.get("opinions", [])
        if opinion_results:
            context_parts.append("## CASE LAW — OPINIONS\n")
            for doc, score in opinion_results:
                meta = doc.metadata
                context_parts.append(
                    f"**{meta.get('title', 'Unknown Case')}**\n"
                    f"Source: {meta.get('source_name', '')} | "
                    f"Date: {meta.get('date', '')} | "
                    f"Court: {meta.get('court', '')} | "
                    f"Section: {meta.get('section', '')}\n\n"
                    f"{doc.page_content}\n\n---\n"
                )
                sources.append(self._make_source(meta, score, meta.get("section", "opinion")))

        # ── Summaries ─────────────────────────────────────
        summary_results = all_results.get("summaries", [])
        if summary_results:
            context_parts.append("## CASE SUMMARIES\n")
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

        context = "\n".join(context_parts) if context_parts else "No relevant cases found in the knowledge base."
        return context, sources

    def retrieve_for_topic(self, topic: str, n_results: int = 10) -> List[Dict]:
        """
        Retrieve top N cases for a broad topic (for browsing/exploration).
        Returns list of source metadata dicts sorted by relevance.
        """
        results = self.store.search_with_score(topic, k=n_results, collection="summaries")
        if not results:
            results = self.store.search_with_score(topic, k=n_results, collection="opinions")

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
            "court": meta.get("court", ""),
            "section": meta.get("section", ""),
            "practice_areas": meta.get("practice_areas", ""),
            "score": round(float(score), 3),
            "collection": collection,
        }

    def is_ready(self) -> bool:
        """Check if the knowledge base has any indexed documents."""
        return sum(self.store.stats().values()) > 0

    def knowledge_base_stats(self) -> Dict[str, int]:
        """Return collection sizes."""
        return self.store.stats()
