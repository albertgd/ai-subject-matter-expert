"""
Tests for the RAG system: VectorStore, Indexer, Retriever.
Uses FakeEmbeddings (from conftest.py) to avoid loading large model.
"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── VectorStore tests ─────────────────────────────────────────────────────────
class TestVectorStore:
    def test_search_returns_results(self, populated_store):
        store, docs = populated_store
        results = store.search("what is a neural network", k=2, collection="documents")
        assert len(results) > 0
        known_titles = {d["title"] for d in docs}
        for r in results:
            assert r.metadata.get("title") in known_titles

    def test_search_with_score(self, populated_store):
        store, _ = populated_store
        results = store.search_with_score("gradient descent backpropagation", k=2, collection="learnings")
        assert len(results) > 0
        for doc, score in results:
            assert isinstance(score, float)
            assert hasattr(doc, "page_content")

    def test_multi_search(self, populated_store):
        store, _ = populated_store
        results = store.multi_search("transformer attention mechanism")
        assert len(results) > 0

    def test_stats(self, populated_store):
        store, _ = populated_store
        stats = store.stats()
        assert sum(stats.values()) > 0
        assert set(stats.keys()) == {"documents", "learnings", "summaries"}


def test_chunk_text_pure():
    """Pure unit test for VectorStore._chunk_text — no ChromaDB."""
    from src.rag.vector_store import VectorStore
    long_text = "A sentence. " * 100  # 1200 chars
    chunks = VectorStore._chunk_text(long_text, max_chars=300, overlap=30)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 400


def test_reset_all(fake_embeddings):
    """Test that reset_all clears all collections."""
    from src.rag.vector_store import VectorStore
    docs = [
        {
            "source_id": "r_001", "source_name": "Test", "title": "Reset Test",
            "url": "", "date": "", "author": "",
            "text": "Machine learning is a subset of artificial intelligence.",
            "summary": "ML is a core AI technique.",
            "key_points": "ML learns from data.", "learnings": "Data drives ML.",
            "topics": ["machine learning"], "structured": True,
        }
    ]
    with tempfile.TemporaryDirectory() as tmpd:
        store = VectorStore(
            persist_dir=Path(tmpd) / "rdb",
            collection_prefix="rst",
            embeddings=fake_embeddings,
        )
        store.add_from_documents(docs)
        assert sum(store.stats().values()) > 0
        store.reset_all()
        assert sum(store.stats().values()) == 0


# ── Indexer tests ─────────────────────────────────────────────────────────────
def test_indexer_load_and_index(fake_embeddings):
    """Test that Indexer loads and indexes processed document files."""
    from src.rag.vector_store import VectorStore
    from src.rag.indexer import Indexer

    docs = [
        {
            "source_id": "idx_001", "source_name": "Test", "title": "Index Test",
            "url": "", "date": "", "author": "",
            "text": "Deep learning uses multiple layers to learn representations from data.",
            "summary": "Deep learning overview.",
            "key_points": "Uses many layers.", "learnings": "Layers enable abstraction.",
            "topics": ["deep learning"], "structured": True,
        }
    ]

    with tempfile.TemporaryDirectory() as tmpd:
        tmpdir = Path(tmpd)
        processed_dir = tmpdir / "processed"
        processed_dir.mkdir()
        for doc in docs:
            (processed_dir / f"{doc['source_id']}.json").write_text(
                json.dumps(doc), encoding="utf-8"
            )

        store = VectorStore(
            persist_dir=tmpdir / "vdb",
            collection_prefix="idx_test",
            embeddings=fake_embeddings,
        )
        indexer = Indexer(store=store, processed_dir=processed_dir)

        loaded = indexer.load_processed_documents()
        assert len(loaded) == 1

        counts = indexer.index_all(force_rebuild=True)
        assert sum(counts.values()) > 0


# ── Retriever tests ───────────────────────────────────────────────────────────
class TestRetriever:
    def test_retrieve_returns_context_and_sources(self, populated_store):
        from src.rag.retriever import Retriever
        store, _ = populated_store
        retriever = Retriever(store=store, k_documents=2, k_learnings=2, k_summaries=1)
        context, sources = retriever.retrieve("neural network layers")
        assert isinstance(context, str) and len(context) > 50
        assert isinstance(sources, list)

    def test_retrieve_with_results(self, populated_store):
        from src.rag.retriever import Retriever
        store, _ = populated_store
        retriever = Retriever(store=store, k_documents=2, k_learnings=2, k_summaries=1)
        context, sources = retriever.retrieve("backpropagation gradient descent")
        assert len(context) > 50
        assert len(sources) > 0

    def test_is_ready(self, populated_store):
        from src.rag.retriever import Retriever
        store, _ = populated_store
        retriever = Retriever(store=store)
        assert retriever.is_ready()

    def test_is_not_ready_when_empty(self, fake_embeddings):
        from src.rag.vector_store import VectorStore
        from src.rag.retriever import Retriever
        with tempfile.TemporaryDirectory() as tmpd:
            store = VectorStore(
                persist_dir=Path(tmpd) / "empty_db",
                collection_prefix="empty",
                embeddings=fake_embeddings,
            )
            retriever = Retriever(store=store)
            assert not retriever.is_ready()

    def test_knowledge_base_stats(self, populated_store):
        from src.rag.retriever import Retriever
        store, _ = populated_store
        retriever = Retriever(store=store)
        stats = retriever.knowledge_base_stats()
        assert isinstance(stats, dict)
        assert sum(stats.values()) > 0
