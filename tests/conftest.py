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


@pytest.fixture(scope="module")
def populated_store(fake_embeddings):
    """Module-scoped VectorStore populated with sample cases."""
    import json
    from src.rag.vector_store import VectorStore

    SAMPLE_CASES = [
        {
            "source_id": "test_001", "source_name": "TestSource",
            "title": "Anderson v. Anderson", "url": "https://example.com/001",
            "date": "2023-03-15", "court": "California Supreme Court",
            "text": "The court awarded joint legal custody considering best interests.",
            "facts": "The parties divorced after 12 years with two children aged 8 and 10.",
            "ruling": "Joint legal custody; primary physical custody to mother.",
            "reasoning": "Best interests standard applied. Mother was primary caregiver.",
            "learnings": "1. Best interests is paramount.\n2. Caregiver status matters.\n3. Joint custody preserves rights.",
            "summary": "Court awarded joint legal custody with primary physical to primary caregiver.",
            "practice_areas": ["custody", "divorce"], "structured": True,
        },
        {
            "source_id": "test_002", "source_name": "TestSource",
            "title": "Brown v. Brown", "url": "https://example.com/002",
            "date": "2022-07-22", "court": "New York Family Court",
            "text": "Equitable distribution of marital assets applied in this divorce case.",
            "facts": "Husband earned $150k; wife unemployed for 5 years raising children.",
            "ruling": "60/40 equitable distribution in wife's favor.",
            "reasoning": "Equitable distribution considers contributions and earning capacity.",
            "learnings": "1. Equal does not mean equitable.\n2. Non-economic contributions count.\n3. Future earning capacity matters.",
            "summary": "Court applied equitable distribution awarding wife 60% of marital assets.",
            "practice_areas": ["property_division", "divorce"], "structured": True,
        },
        {
            "source_id": "test_003", "source_name": "TestSource",
            "title": "Williams v. Williams", "url": "https://example.com/003",
            "date": "2021-11-30", "court": "Texas Court of Appeals",
            "text": "Rehabilitative alimony awarded after career sacrifice in long marriage.",
            "facts": "20-year marriage. Wife gave up career. Husband earns $200k.",
            "ruling": "Rehabilitative alimony for 3 years.",
            "reasoning": "Career sacrifice during long marriage supports alimony award.",
            "learnings": "1. Career sacrifice supports alimony.\n2. Rehabilitative alimony restores capacity.\n3. Duration tied to re-entry time.",
            "summary": "Court awarded 3-year rehabilitative alimony after career sacrifice in 20-year marriage.",
            "practice_areas": ["alimony", "spousal_support", "divorce"], "structured": True,
        },
    ]

    with tempfile.TemporaryDirectory() as tmpd:
        store = VectorStore(
            persist_dir=Path(tmpd) / "vdb",
            collection_prefix="fixture_cases",
            embeddings=fake_embeddings,
        )
        store.add_from_cases(SAMPLE_CASES)
        yield store, SAMPLE_CASES
