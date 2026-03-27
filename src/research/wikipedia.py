"""
Wikipedia source — reliable, structured, free, no API key needed.

Uses the wikipedia-api library to search and fetch articles for any subject.
Produces one document per article section to keep chunks meaningful.
"""

import logging
from typing import Dict, List, Optional

from src.config import SUBJECT, SUBJECT_KEYWORDS, MAX_DOCS_PER_SOURCE, LANGUAGE, REGION
from src.research.base import BaseCollector

# Extra seed pages to fetch when a specific region is selected (Spanish)
_REGION_SEED_PAGES_ES = {
    "spain":     ["España", "Derecho español", "Tribunal Supremo de España", "Constitución española de 1978"],
    "catalunya": ["Cataluña", "Generalitat de Cataluña", "Dret civil català", "Estatuto de Autonomía de Cataluña"],
    "europe":    ["Unión Europea", "Tribunal de Justicia de la Unión Europea", "Derecho europeo", "Consejo de Europa"],
}
_REGION_SEED_PAGES_EN = {
    "spain":     ["Spain", "Spanish law", "Supreme Court of Spain"],
    "catalunya": ["Catalonia", "Generalitat de Catalunya", "Catalan law"],
    "europe":    ["European Union", "Court of Justice of the European Union", "European law"],
}

logger = logging.getLogger(__name__)


class WikipediaSource(BaseCollector):
    """Fetches Wikipedia articles related to the configured subject."""

    SOURCE_NAME = "wikipedia"

    def __init__(self, language: str = LANGUAGE, region: str = REGION, **kwargs):
        super().__init__(**kwargs)
        self._language = language
        self._region   = region
        wiki_lang = "es" if language == "es" else "en"
        try:
            import wikipediaapi
            self._wiki = wikipediaapi.Wikipedia(
                language=wiki_lang,
                user_agent=(
                    "AI-SME-Bot/1.0 "
                    "(https://github.com/albertgd/ai-subject-matter-expert)"
                ),
            )
        except ImportError:
            self._wiki = None
            logger.warning("wikipedia-api not installed — Wikipedia source disabled")

    def collect(self, max_docs: int = MAX_DOCS_PER_SOURCE) -> List[Dict]:
        """Collect Wikipedia articles for the subject keywords."""
        if self._wiki is None:
            return []

        docs = []
        seen_titles = set()

        # Build seed list: subject keywords + region-specific pages
        region_seeds: List[str] = []
        if self._region != "anywhere":
            seed_map = _REGION_SEED_PAGES_ES if self._language == "es" else _REGION_SEED_PAGES_EN
            region_seeds = seed_map.get(self._region, [])

        # Search using each keyword
        for keyword in [SUBJECT] + SUBJECT_KEYWORDS + region_seeds:
            if len(docs) >= max_docs:
                break
            try:
                page = self._wiki.page(keyword)
                if page.exists() and page.title not in seen_titles:
                    seen_titles.add(page.title)
                    article_docs = self._page_to_documents(page)
                    docs.extend(article_docs)
                    logger.info(f"  Wikipedia '{page.title}': {len(article_docs)} sections")
                    self._sleep()

                    # Also follow linked pages (one level deep)
                    for link_title in list(page.links.keys())[:5]:
                        if len(docs) >= max_docs:
                            break
                        if link_title in seen_titles:
                            continue
                        try:
                            linked = self._wiki.page(link_title)
                            if linked.exists():
                                seen_titles.add(link_title)
                                linked_docs = self._page_to_documents(linked)
                                docs.extend(linked_docs)
                                logger.info(
                                    f"  Wikipedia (linked) '{link_title}': "
                                    f"{len(linked_docs)} sections"
                                )
                                self._sleep()
                        except Exception as e:
                            logger.debug(f"  Linked page failed '{link_title}': {e}")
            except Exception as e:
                logger.warning(f"  Wikipedia failed for '{keyword}': {e}")

        return docs[:max_docs]

    def _page_to_documents(self, page) -> List[Dict]:
        """Convert a Wikipedia page into one document per top-level section."""
        docs = []
        wiki_lang = "es" if self._language == "es" else "en"
        base_url = f"https://{wiki_lang}.wikipedia.org/wiki/{page.title.replace(' ', '_')}"

        def _make_doc(title: str, text: str, section: str) -> Optional[Dict]:
            text = text.strip()
            if len(text) < 200:
                return None
            source_id = f"wiki_{page.title.replace(' ', '_')}_{section.replace(' ', '_')}"
            source_id = source_id[:100]
            if self.already_collected(source_id):
                return None
            doc = {
                "source_id": source_id,
                "source_name": "Wikipedia",
                "title": f"{page.title} — {section}" if section != "intro" else page.title,
                "url": base_url if section == "intro" else f"{base_url}#{section.replace(' ', '_')}",
                "date": "",
                "author": "Wikipedia contributors",
                "text": text,
            }
            self.save_document(doc)
            return doc

        # Article intro (summary)
        intro_doc = _make_doc(page.title, page.summary, "intro")
        if intro_doc:
            docs.append(intro_doc)

        # Top-level sections
        for section in page.sections:
            doc = _make_doc(page.title, section.text, section.title)
            if doc:
                docs.append(doc)
            # Sub-sections
            for sub in section.sections:
                sub_doc = _make_doc(page.title, sub.text, f"{section.title}_{sub.title}")
                if sub_doc:
                    docs.append(sub_doc)

        return docs
