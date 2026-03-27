"""
Search engine adapters.

Priority:
  1. Tavily  — if TAVILY_API_KEY is set (AI-optimized, returns clean content)
  2. DuckDuckGo — free, no key required (via duckduckgo-search)

Both return a list of dicts: {title, url, snippet}
"""

import logging
from typing import List, Dict

from src.config import TAVILY_API_KEY, MAX_SEARCH_RESULTS

logger = logging.getLogger(__name__)


def search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> List[Dict]:
    """
    Run a web search and return result dicts.

    Each result: {"title": str, "url": str, "snippet": str}
    """
    if TAVILY_API_KEY:
        return _tavily_search(query, max_results)
    return _ddg_search(query, max_results)


def _tavily_search(query: str, max_results: int) -> List[Dict]:
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)
        resp = client.search(query, max_results=max_results, search_depth="advanced")
        results = []
        for r in resp.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", r.get("snippet", "")),
            })
        logger.debug(f"Tavily: {len(results)} results for '{query}'")
        return results
    except Exception as e:
        logger.warning(f"Tavily search failed, falling back to DuckDuckGo: {e}")
        return _ddg_search(query, max_results)


def _ddg_search(query: str, max_results: int) -> List[Dict]:
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results, safesearch="on"):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        logger.debug(f"DuckDuckGo: {len(results)} results for '{query}'")
        return results
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {e}")
        return []
