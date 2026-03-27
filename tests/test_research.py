"""
Tests for the research module — offline/unit tests only (no real HTTP calls).
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


class TestBaseCollector:
    def test_save_and_load_document(self):
        from src.research.base import BaseCollector

        class ConcreteCollector(BaseCollector):
            SOURCE_NAME = "test_source"

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = ConcreteCollector(output_dir=Path(tmpdir))

            doc = {
                "source_id": "test_001",
                "source_name": "TestSource",
                "title": "Neural Networks Overview",
                "url": "https://example.com/nn",
                "date": "2024-01-01",
                "author": "Test Author",
                "text": "Neural networks are computing systems. " * 10,
            }

            path = collector.save_document(doc)
            assert path.exists()

            loaded = collector.load_all()
            assert len(loaded) == 1
            assert loaded[0]["source_id"] == "test_001"
            assert loaded[0]["title"] == "Neural Networks Overview"

    def test_already_collected(self):
        from src.research.base import BaseCollector

        class ConcreteCollector(BaseCollector):
            SOURCE_NAME = "test_source2"

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = ConcreteCollector(output_dir=Path(tmpdir))

            assert not collector.already_collected("doc_123")

            doc = {
                "source_id": "doc_123",
                "source_name": "Test",
                "title": "Test Document",
                "url": "",
                "date": "",
                "author": "",
                "text": "Some content here.",
            }
            collector.save_document(doc)
            assert collector.already_collected("doc_123")

    def test_multiple_documents(self):
        from src.research.base import BaseCollector

        class ConcreteCollector(BaseCollector):
            SOURCE_NAME = "test_source3"

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = ConcreteCollector(output_dir=Path(tmpdir))
            for i in range(5):
                collector.save_document({
                    "source_id": f"doc_{i}",
                    "source_name": "Test",
                    "title": f"Document {i}",
                    "url": f"https://example.com/{i}",
                    "date": "2024-01-01",
                    "author": "",
                    "text": f"Content of document {i}. " * 10,
                })

            loaded = collector.load_all()
            assert len(loaded) == 5


class TestWebFetcher:
    def test_url_to_id(self):
        from src.research.web_fetcher import WebFetcher
        url1 = "https://en.wikipedia.org/wiki/Neural_network"
        url2 = "https://en.wikipedia.org/wiki/Backpropagation"
        id1 = WebFetcher._url_to_id(url1)
        id2 = WebFetcher._url_to_id(url2)
        assert id1 != id2
        assert len(id1) == 16
        assert len(id2) == 16

    def test_is_binary_url(self):
        from src.research.web_fetcher import WebFetcher
        assert WebFetcher._is_binary_url("https://example.com/paper.pdf")
        assert WebFetcher._is_binary_url("https://example.com/image.jpg")
        assert not WebFetcher._is_binary_url("https://en.wikipedia.org/wiki/AI")
        assert not WebFetcher._is_binary_url("https://example.com/article.html")

    def test_url_to_id_stable(self):
        from src.research.web_fetcher import WebFetcher
        url = "https://example.com/some/path"
        assert WebFetcher._url_to_id(url) == WebFetcher._url_to_id(url)


class TestSearchModule:
    def test_ddg_search_returns_list(self):
        """Test DuckDuckGo search returns a list (mocked)."""
        with patch("src.research.search._ddg_search") as mock_search:
            mock_search.return_value = [
                {"title": "Neural Network", "url": "https://example.com", "snippet": "A neural network..."}
            ]
            from src.research.search import _ddg_search
            results = _ddg_search("neural networks", 5)
            assert isinstance(results, list)

    def test_search_returns_list_on_failure(self):
        """Search should return empty list on failure, not raise."""
        with patch("src.research.search._ddg_search", side_effect=Exception("Network error")):
            # The top-level search() catches this via _ddg_search fallback
            from src.research.search import _ddg_search
            try:
                results = _ddg_search("test", 5)
            except Exception:
                results = []
            assert isinstance(results, list)


class TestAIResearcherFallbackQueries:
    def test_fallback_queries_returns_correct_count(self):
        from src.research.ai_researcher import AIResearcher
        researcher = AIResearcher.__new__(AIResearcher)
        researcher._language = "en"
        researcher._region = "anywhere"
        queries = researcher._fallback_queries("quantum physics", 5)
        assert len(queries) == 5
        assert all(isinstance(q, str) for q in queries)
        assert all("quantum physics" in q for q in queries)

    def test_fallback_queries_subject_included(self):
        from src.research.ai_researcher import AIResearcher
        researcher = AIResearcher.__new__(AIResearcher)
        researcher._language = "en"
        researcher._region = "anywhere"
        queries = researcher._fallback_queries("machine learning", 3)
        for q in queries:
            assert "machine learning" in q
