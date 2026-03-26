"""Shared pytest fixtures for the test suite."""

import hashlib
import math
import tempfile
from pathlib import Path
from typing import List

import pytest
from langchain_core.embeddings import Embeddings as LCEmbeddings


class FakeEmbeddings(LCEmbeddings):
    """Deterministic fake embeddings — fast, no model download needed."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._hash_embed(text)

    @staticmethod
    def _hash_embed(text: str, dim: int = 384) -> List[float]:
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
        vec = []
        for i in range(dim):
            seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
            vec.append((seed / 0xFFFFFFFF) * 2 - 1)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


@pytest.fixture(scope="module")
def fake_embeddings():
    return FakeEmbeddings()


SAMPLE_DOCUMENTS = [
    {
        "source_id": "test_001", "source_name": "Wikipedia",
        "title": "Neural Networks", "url": "https://en.wikipedia.org/wiki/Neural_network",
        "date": "2024-01-01", "author": "Wikipedia contributors",
        "text": "A neural network is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates.",
        "summary": "Neural networks are computing systems inspired by biological neural networks that form animal brains.",
        "key_points": "1. Neural networks learn from examples.\n2. They consist of layers of nodes.\n3. Deep learning uses many hidden layers.",
        "learnings": "Neural networks are universal function approximators.",
        "topics": ["neural networks", "deep learning", "machine learning"],
        "structured": True,
    },
    {
        "source_id": "test_002", "source_name": "Wikipedia",
        "title": "Backpropagation", "url": "https://en.wikipedia.org/wiki/Backpropagation",
        "date": "2024-01-02", "author": "Wikipedia contributors",
        "text": "Backpropagation is an algorithm widely used in training feedforward artificial neural networks by computing gradients of the loss function with respect to the weights.",
        "summary": "Backpropagation computes gradients efficiently using the chain rule to train neural networks.",
        "key_points": "1. Uses chain rule of calculus.\n2. Updates weights via gradient descent.\n3. Requires differentiable activation functions.",
        "learnings": "Backpropagation enables efficient gradient computation in neural networks.",
        "topics": ["backpropagation", "neural networks", "gradient descent"],
        "structured": True,
    },
    {
        "source_id": "test_003", "source_name": "ArXiv",
        "title": "Transformers in NLP", "url": "https://arxiv.org/abs/1706.03762",
        "date": "2017-06-12", "author": "Vaswani et al.",
        "text": "The transformer architecture relies on attention mechanisms and has become the dominant approach in natural language processing tasks.",
        "summary": "Transformers use self-attention to process sequences in parallel, enabling state-of-the-art NLP.",
        "key_points": "1. Attention is all you need.\n2. Transformers replaced RNNs for many tasks.\n3. BERT and GPT are based on transformers.",
        "learnings": "Self-attention allows models to relate positions in a sequence to each other.",
        "topics": ["transformers", "NLP", "attention mechanism"],
        "structured": True,
    },
]


@pytest.fixture(scope="module")
def populated_store(fake_embeddings):
    """Module-scoped VectorStore populated with sample documents."""
    from src.rag.vector_store import VectorStore

    with tempfile.TemporaryDirectory() as tmpd:
        store = VectorStore(
            persist_dir=Path(tmpd) / "vdb",
            collection_prefix="fixture_docs",
            embeddings=fake_embeddings,
        )
        store.add_from_documents(SAMPLE_DOCUMENTS)
        yield store, SAMPLE_DOCUMENTS
