"""
AI-powered researcher.

Uses an LLM to:
  1. Generate diverse, targeted search queries for the subject
  2. Evaluate which search results are worth fetching
  3. Extract structured knowledge from raw web content

Flow:
    AIResearcher.research(subject)
      → generate_queries()        # LLM generates N queries
      → search(query)             # DuckDuckGo / Tavily
      → fetch(url)                # WebFetcher
      → documents saved to data/raw/ai_researcher/
"""

import json
import logging
from typing import Dict, List, Optional

from src.config import (
    SUBJECT, SUBJECT_DESCRIPTION, SUBJECT_KEYWORDS,
    MAX_DOCS_PER_SOURCE, MAX_SEARCH_RESULTS,
    LANGUAGE, REGION,
)
from src.research.base import BaseCollector
from src.research.search import search
from src.research.web_fetcher import WebFetcher

logger = logging.getLogger(__name__)

_DEFAULT_N_QUERIES = 10
_MAX_URLS_PER_QUERY = 5

_REGION_CONTEXT = {
    "spain":     {"en": "in Spain (Spanish law and context)",  "es": "en España (derecho y contexto español)"},
    "catalunya": {"en": "in Catalonia, Spain",                 "es": "en Catalunya (derecho catalán y español)"},
    "europe":    {"en": "in Europe (EU law and context)",      "es": "en Europa (derecho europeo y de la UE)"},
    "anywhere":  {"en": "",                                    "es": ""},
}


class AIResearcher(BaseCollector):
    """
    Drives web research using an LLM as the research brain.

    The LLM:
      - Generates smart, varied search queries
      - Filters irrelevant URLs before fetching
    """

    SOURCE_NAME = "ai_researcher"

    def __init__(self, language: str = LANGUAGE, region: str = REGION, **kwargs):
        super().__init__(**kwargs)
        self._language = language
        self._region   = region
        self._llm = None
        self._fetcher = WebFetcher()

    @property
    def llm(self):
        if self._llm is None:
            from src.agents.llm_factory import build_llm
            self._llm = build_llm()
        return self._llm

    # ── Public API ────────────────────────────────────────
    def research(
        self,
        subject: str = SUBJECT,
        n_queries: int = _DEFAULT_N_QUERIES,
        max_docs: int = MAX_DOCS_PER_SOURCE,
    ) -> List[Dict]:
        """
        Full research pipeline for a subject.
        Returns list of raw document dicts.
        """
        logger.info(f"Starting AI research on: '{subject}' (lang={self._language}, region={self._region})")

        queries = self.generate_queries(subject, n=n_queries)
        logger.info(f"Generated {len(queries)} search queries")

        docs = []
        seen_urls: set = set()

        for i, query in enumerate(queries):
            if len(docs) >= max_docs:
                break
            logger.info(f"  Query {i+1}/{len(queries)}: {query}")

            results = search(query, max_results=MAX_SEARCH_RESULTS)
            relevant = self._filter_relevant(results, subject)

            for result in relevant[:_MAX_URLS_PER_QUERY]:
                if len(docs) >= max_docs:
                    break
                url = result.get("url", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)

                doc = self._fetcher.fetch(url, title_hint=result.get("title", ""))
                if doc:
                    docs.append(doc)
                    logger.info(f"    Fetched: {doc['title'][:60]}")
                    self._sleep()

        logger.info(f"AI research complete: {len(docs)} documents collected")
        return docs

    # ── Query generation ──────────────────────────────────
    def generate_queries(self, subject: str, n: int = _DEFAULT_N_QUERIES) -> List[str]:
        """Ask the LLM to generate n diverse search queries for the subject."""
        region_ctx = _REGION_CONTEXT.get(self._region, {}).get(self._language, "")
        region_note = f"\nRegion focus: {region_ctx}" if region_ctx else ""

        if self._language == "es":
            lang_instruction = (
                "IMPORTANTE: Genera las consultas de búsqueda en ESPAÑOL. "
                "Usa terminología española/latinoamericana adecuada para el tema."
            )
            example_note = 'Ejemplo: ["consulta 1 en español", "consulta 2 en español", ...]'
        else:
            lang_instruction = ""
            example_note = 'Example: ["query 1", "query 2", ...]'

        prompt = f"""You are a research assistant. Generate {n} diverse, specific web search queries
to gather comprehensive information about: {subject}

Context: {SUBJECT_DESCRIPTION}
Key topics: {", ".join(SUBJECT_KEYWORDS[:8])}{region_note}

Requirements:
- Cover different aspects: fundamentals, recent developments, practical applications, key concepts
- Mix broad overview queries with specific technical queries
- Suitable for finding high-quality educational and reference content
- Do NOT include site: operators
{lang_instruction}

Return ONLY a JSON array of {n} query strings, nothing else.
{example_note}"""

        try:
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()

            # Extract JSON array
            start = content.find("[")
            end = content.rfind("]") + 1
            if start != -1 and end > start:
                queries = json.loads(content[start:end])
                if isinstance(queries, list):
                    return [str(q) for q in queries if q][:n]
        except Exception as e:
            logger.warning(f"LLM query generation failed: {e}")

        # Fallback: build queries from keywords
        return self._fallback_queries(subject, n)

    # ── Relevance filtering ───────────────────────────────
    def _filter_relevant(
        self, results: List[Dict], subject: str
    ) -> List[Dict]:
        """Filter search results: require keyword hit AND a non-blocked domain."""
        if not results:
            return []

        from src.research.web_fetcher import WebFetcher
        subject_lower = subject.lower()
        keyword_lower = [kw.lower() for kw in SUBJECT_KEYWORDS]

        scored = []
        for r in results:
            url = r.get("url", "")
            # Drop blocked domains at query-result stage (before even fetching)
            if not url or WebFetcher._is_blocked_domain(url):
                continue

            text = (r.get("title", "") + " " + r.get("snippet", "")).lower()
            score = sum(1 for kw in keyword_lower if kw in text)
            if subject_lower in text:
                score += 3

            # Require at least one keyword match in title+snippet
            if score == 0:
                continue

            scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored]

    # ── Fallback query builder ─────────────────────────────
    def _fallback_queries(self, subject: str, n: int) -> List[str]:
        region_ctx = _REGION_CONTEXT.get(self._region, {}).get(self._language, "")
        suffix = f" {region_ctx}" if region_ctx else ""

        if self._language == "es":
            templates = [
                f"{subject} resumen{suffix}",
                f"{subject} introducción{suffix}",
                f"cómo funciona {subject}{suffix}",
                f"{subject} conceptos clave{suffix}",
                f"{subject} aplicaciones{suffix}",
                f"{subject} historia{suffix}",
                f"{subject} investigación reciente{suffix}",
                f"{subject} ejemplos{suffix}",
                f"qué es {subject}{suffix}",
                f"{subject} fundamentos{suffix}",
                f"{subject} guía completa{suffix}",
                f"{subject} temas avanzados{suffix}",
            ]
        else:
            templates = [
                f"{subject} overview{suffix}",
                f"{subject} introduction{suffix}",
                f"how does {subject} work{suffix}",
                f"{subject} key concepts{suffix}",
                f"{subject} applications{suffix}",
                f"{subject} history{suffix}",
                f"{subject} latest research{suffix}",
                f"{subject} examples{suffix}",
                f"{subject} tutorial{suffix}",
                f"what is {subject}{suffix}",
                f"{subject} fundamentals{suffix}",
                f"{subject} advanced topics{suffix}",
            ]
        return templates[:n]
