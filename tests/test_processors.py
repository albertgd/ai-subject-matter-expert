"""
Tests for data processors: TextCleaner, PIIRemover, CaseStructurer.
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
        raw = "The court ruled &amp; found that &quot;custody&quot; was paramount. " * 5
        result = self.cleaner.clean(raw)
        assert result is not None
        assert "&amp;" not in result
        assert "custody" in result

    def test_whitespace_normalization(self):
        sentence = "First sentence about legal proceedings and divorce law matters. "
        raw = (sentence * 10) + "\n\n\n\n\n\n" + (sentence * 10)
        result = self.cleaner.clean(raw)
        assert result is not None
        # Should not have more than 2 consecutive newlines
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
        raw = "The court\u2019s ruling on \u201ccustody\u201d was clear. " * 10
        result = self.cleaner.clean(raw)
        assert result is not None
        assert "\u2019" not in result  # Curly apostrophe replaced
        assert "\u201c" not in result  # Curly quote replaced

    def test_truncate(self):
        from src.processors.text_cleaner import TextCleaner
        long_text = "x" * 200_000
        truncated = TextCleaner.truncate(long_text, max_chars=10_000)
        assert len(truncated) <= 10_200  # Some slack for the truncation marker

    def test_boilerplate_removal(self):
        raw = (
            "ADVERTISEMENT\n"
            "In the matter of Smith v. Jones, the court held the following. "
            "The parties agreed to a custody arrangement. "
            "The court found in favor of the petitioner. " * 10
        )
        result = self.cleaner.clean(raw)
        assert result is not None
        # Boilerplate should be reduced
        assert len(result) < len(raw)


# ── PIIRemover tests ──────────────────────────────────────────────────────────
class TestPIIRemover:
    def setup_method(self):
        from src.processors.pii_remover import PIIRemover
        # Use regex-only mode for fast tests (no spaCy dependency)
        self.remover = PIIRemover(use_presidio=False)

    def test_phone_number_removal(self):
        text = "Contact John at 555-123-4567 for more information."
        result = self.remover.anonymize(text)
        assert "555-123-4567" not in result
        assert "[PHONE]" in result

    def test_email_removal(self):
        text = "Send documents to john.smith@example.com immediately."
        result = self.remover.anonymize(text)
        assert "john.smith@example.com" not in result
        assert "[EMAIL]" in result

    def test_ssn_removal(self):
        text = "The taxpayer's SSN is 123-45-6789."
        result = self.remover.anonymize(text)
        assert "123-45-6789" not in result
        assert "[SSN]" in result

    def test_preserves_court_content(self):
        text = (
            "The Supreme Court of California ruled that the custody arrangement "
            "was in the best interests of the children. " * 5
        )
        result = self.remover.anonymize(text)
        assert "Supreme Court" in result
        assert "custody" in result

    def test_empty_text(self):
        assert self.remover.anonymize("") == ""
        assert self.remover.anonymize(None) is None  # noqa

    def test_anonymize_case(self):
        case = {
            "source_id": "test_1",
            "text": "John Smith (john@test.com, 555-999-8888) filed for divorce.",
            "facts": "The petitioner John Smith filed on 01/01/2020.",
        }
        result = self.remover.anonymize_case(case)
        assert "john@test.com" not in result["text"]
        assert "[EMAIL]" in result["text"]
        assert "555-999-8888" not in result["text"]
        assert "[PHONE]" in result["text"]


# ── CaseStructurer tests ──────────────────────────────────────────────────────
class TestCaseStructurerOffline:
    """Offline tests that don't make LLM API calls."""

    def test_empty_text_returns_unchanged(self):
        """Test that cases with no text are returned unchanged."""
        from src.processors.case_structurer import CaseStructurer

        class MockLLM:
            def invoke(self, messages):
                raise Exception("Should not be called")

        structurer = CaseStructurer.__new__(CaseStructurer)
        structurer.llm = MockLLM()

        case = {"source_id": "test_empty", "text": ""}
        result = structurer.structure(case)
        assert result["source_id"] == "test_empty"
        # Should return unchanged (no text to structure)
        assert not result.get("structured")

    def test_json_parse_error_fallback(self):
        """Test that JSON parse errors result in graceful fallback."""
        from src.processors.case_structurer import CaseStructurer
        from langchain_core.messages import AIMessage

        class MockLLM:
            def invoke(self, messages):
                return AIMessage(content="This is not valid JSON at all!")

        structurer = CaseStructurer.__new__(CaseStructurer)
        structurer.llm = MockLLM()

        case = {
            "source_id": "test_json_fail",
            "text": "The court held that custody should be awarded. " * 30,
        }
        result = structurer.structure(case)
        assert result["source_id"] == "test_json_fail"
        assert result.get("structured") is False
        assert result.get("summary")  # Should have a fallback summary
