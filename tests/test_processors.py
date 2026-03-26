"""
Tests for data processors: TextCleaner, PIIRemover, ContentStructurer.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ── TextCleaner tests ─────────────────────────────────────────────────────────
class TestTextCleaner:
    def setup_method(self):
        from src.processors.text_cleaner import TextCleaner
        self.cleaner = TextCleaner()

    def test_basic_cleaning(self):
        raw = "  Hello   World  \n\n\n\nsome content here. " * 20
        result = self.cleaner.clean(raw)
        assert result is not None
        assert "Hello" in result

    def test_html_entity_removal(self):
        raw = "The model &amp; found that &quot;attention&quot; was paramount. " * 5
        result = self.cleaner.clean(raw)
        assert result is not None
        assert "&amp;" not in result
        assert "attention" in result

    def test_whitespace_normalization(self):
        sentence = "First sentence about deep learning and neural network matters. "
        raw = (sentence * 10) + "\n\n\n\n\n\n" + (sentence * 10)
        result = self.cleaner.clean(raw)
        assert result is not None
        assert "\n\n\n" not in result

    def test_returns_none_for_short_text(self):
        result = self.cleaner.clean("Too short")
        assert result is None

    def test_returns_none_for_empty(self):
        result = self.cleaner.clean("")
        assert result is None
        result = self.cleaner.clean(None)
        assert result is None

    def test_unicode_normalization(self):
        raw = "The model\u2019s output on \u201cattention\u201d was clear. " * 10
        result = self.cleaner.clean(raw)
        assert result is not None
        assert "\u2019" not in result
        assert "\u201c" not in result

    def test_truncate(self):
        from src.processors.text_cleaner import TextCleaner
        long_text = "x" * 200_000
        truncated = TextCleaner.truncate(long_text, max_chars=10_000)
        assert len(truncated) <= 10_200

    def test_boilerplate_removal(self):
        raw = (
            "ADVERTISEMENT\n"
            "In the study of neural networks, the model achieved high accuracy. "
            "The researchers found that attention mechanisms improve results. " * 10
        )
        result = self.cleaner.clean(raw)
        assert result is not None
        assert len(result) < len(raw)

    def test_clean_document(self):
        from src.processors.text_cleaner import TextCleaner
        cleaner = TextCleaner()
        doc = {
            "source_id": "test",
            "text": "  \n\nDeep learning has revolutionized AI research significantly. " * 10,
        }
        result = cleaner.clean_document(doc)
        assert result["text"]
        assert result["text"] == result["text"].strip()


# ── PIIRemover tests ──────────────────────────────────────────────────────────
class TestPIIRemover:
    def setup_method(self):
        from src.processors.pii_remover import PIIRemover
        self.remover = PIIRemover(use_presidio=False)

    def test_phone_number_removal(self):
        text = "Contact the researcher at 555-123-4567 for more information."
        result = self.remover.anonymize(text)
        assert "555-123-4567" not in result
        assert "[PHONE]" in result

    def test_email_removal(self):
        text = "Send documents to john.smith@example.com immediately."
        result = self.remover.anonymize(text)
        assert "john.smith@example.com" not in result
        assert "[EMAIL]" in result

    def test_ssn_removal(self):
        text = "The contributor's SSN is 123-45-6789."
        result = self.remover.anonymize(text)
        assert "123-45-6789" not in result
        assert "[SSN]" in result

    def test_preserves_technical_content(self):
        text = (
            "The transformer architecture was introduced in the paper "
            "'Attention is All You Need' by Google Brain researchers. " * 5
        )
        result = self.remover.anonymize(text)
        assert "transformer" in result
        assert "attention" in result.lower()

    def test_empty_text(self):
        assert self.remover.anonymize("") == ""
        assert self.remover.anonymize(None) is None  # noqa

    def test_anonymize_document(self):
        doc = {
            "source_id": "test_1",
            "text": "Contact john@test.com or call 555-999-8888 for the dataset.",
            "summary": "Dataset available at john@test.com.",
        }
        result = self.remover.anonymize_document(doc)
        assert "john@test.com" not in result["text"]
        assert "[EMAIL]" in result["text"]
        assert "555-999-8888" not in result["text"]
        assert "[PHONE]" in result["text"]


# ── ContentStructurer tests ───────────────────────────────────────────────────
class TestContentStructurerOffline:
    """Offline tests that don't make LLM API calls."""

    def test_empty_text_returns_unchanged(self):
        from src.processors.content_structurer import ContentStructurer

        class MockLLM:
            def invoke(self, messages):
                raise Exception("Should not be called")

        structurer = ContentStructurer.__new__(ContentStructurer)
        structurer.llm = MockLLM()

        doc = {"source_id": "test_empty", "text": ""}
        result = structurer.structure(doc)
        assert result["source_id"] == "test_empty"
        assert not result.get("structured")

    def test_json_parse_error_fallback(self):
        from src.processors.content_structurer import ContentStructurer
        from langchain_core.messages import AIMessage

        class MockLLM:
            def invoke(self, messages):
                return AIMessage(content="This is not valid JSON at all!")

        structurer = ContentStructurer.__new__(ContentStructurer)
        structurer.llm = MockLLM()

        doc = {
            "source_id": "test_json_fail",
            "text": "Deep learning uses multiple layers of neural networks. " * 30,
        }
        result = structurer.structure(doc)
        assert result["source_id"] == "test_json_fail"
        assert result.get("structured") is False
        assert result.get("summary")

    def test_successful_extraction(self):
        import json
        from src.processors.content_structurer import ContentStructurer
        from langchain_core.messages import AIMessage

        extracted = {
            "summary": "Neural networks learn from data using gradient descent.",
            "key_points": "1. Neural networks have layers.\n2. They use backpropagation.",
            "learnings": "Deeper networks learn more complex features.",
            "topics": ["neural networks", "deep learning"],
        }

        class MockLLM:
            def invoke(self, messages):
                return AIMessage(content=json.dumps(extracted))

        structurer = ContentStructurer.__new__(ContentStructurer)
        structurer.llm = MockLLM()

        doc = {
            "source_id": "test_ok",
            "text": "A neural network is a computational model. " * 30,
        }
        result = structurer.structure(doc)
        assert result["structured"] is True
        assert result["summary"] == extracted["summary"]
        assert result["topics"] == extracted["topics"]
