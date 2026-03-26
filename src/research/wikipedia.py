"""
Wikipedia source — reliable, structured, free, no API key needed.

Uses the wikipedia-api library to search and fetch articles for any subject.
Produces one document per article section to keep chunks meaningful.
"""

import logging
from typing import Dict, List, Optional

from src.config import SUBJECT, SUBJECT_KEYWORDS, MAX_DOCS_PER_SOURCE
from src.research.base import BaseCollector

logger = logging.getLogger(__name__)


class WikipediaSource(BaseCollector):
    """Fetches Wikipedia articles related to the configured subject."""

    SOURCE_NAME = "wikipedia"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        try:
            import wikipediaapi
            self._wiki = wikipediaapi.Wikipedia(
                language="en",
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

        # Search using each keyword
        for keyword in [SUBJECT] + SUBJECT_KEYWORDS:
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
        base_url = f"https://en.wikipedia.org/wiki/{page.title.replace(' ', '_')}"

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
