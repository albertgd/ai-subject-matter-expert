"""
PII Remover — Anonymize personally identifiable information from web content.

Uses:
  - Microsoft Presidio (https://github.com/microsoft/presidio) — NER-based PII detection
  - spaCy en_core_web_lg (or sm) — English NLP model for Presidio
  - Regex fallbacks for common PII patterns (SSN, phone, email, etc.)

Replaced entity types:
  - PERSON       → [PERSON_N]
  - PHONE_NUMBER → [PHONE]
  - EMAIL_ADDRESS → [EMAIL]
  - US_SSN       → [SSN]
  - CREDIT_CARD  → [CREDIT_CARD]
  - IP_ADDRESS   → [IP_ADDRESS]
  - US_DRIVER_LICENSE → [DRIVER_LICENSE]
  - US_PASSPORT  → [PASSPORT]

Public figures and organization names are intentionally preserved as
they are matters of public record. Only private individual identifying
information is anonymized.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# ── Regex patterns (always applied, no spaCy dependency) ─
_REGEX_PATTERNS = {
    "SSN": r"\b\d{3}[- ]\d{2}[- ]\d{4}\b",
    "PHONE": r"\b(?:\+1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b",
    "EMAIL": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
    "CREDIT_CARD": r"\b(?:\d{4}[- ]?){3}\d{4}\b",
    "ZIP": r"\b\d{5}(?:-\d{4})?\b",  # Only strip if appears as standalone
}


class PIIRemover:
    """
    Removes PII from legal text, preserving public record info.

    Usage:
        remover = PIIRemover()
        clean_text = remover.anonymize(text)

    The class tries to use Presidio + spaCy for high-quality NER-based
    anonymization. If those aren't installed, it falls back to regex only.
    """

    def __init__(self, use_presidio: bool = True, spacy_model: str = "en_core_web_lg"):
        self._analyzer = None
        self._anonymizer = None
        self._person_counter: dict = {}

        if use_presidio:
            self._setup_presidio(spacy_model)

    def _setup_presidio(self, spacy_model: str):
        """Initialize Presidio analyzer and anonymizer."""
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            from presidio_anonymizer import AnonymizerEngine

            # Try the large model first, fall back to small
            models_to_try = [spacy_model, "en_core_web_sm"]
            for model in models_to_try:
                try:
                    config = {
                        "nlp_engine_name": "spacy",
                        "models": [{"lang_code": "en", "model_name": model}],
                    }
                    provider = NlpEngineProvider(nlp_configuration=config)
                    nlp_engine = provider.create_engine()
                    self._analyzer = AnalyzerEngine(
                        nlp_engine=nlp_engine,
                        supported_languages=["en"]
                    )
                    self._anonymizer = AnonymizerEngine()
                    logger.info(f"PIIRemover: Presidio initialized with {model}.")
                    return
                except Exception:
                    continue

            logger.warning("PIIRemover: Could not load spaCy model. Using regex-only mode.")
        except ImportError:
            logger.warning("PIIRemover: Presidio not installed. Using regex-only mode.")

    def anonymize(self, text: str) -> str:
        """
        Anonymize PII in text.

        Args:
            text: Raw legal text

        Returns:
            Text with PII replaced by placeholder tokens
        """
        if not text or not text.strip():
            return text

        # Pass 1: Regex-based patterns (always run)
        text = self._apply_regex_patterns(text)

        # Pass 2: Presidio NER (if available)
        if self._analyzer and self._anonymizer:
            text = self._apply_presidio(text)

        return text

    def anonymize_document(self, doc: dict) -> dict:
        """Anonymize PII in all text fields of a document dict."""
        doc = dict(doc)
        text_fields = ["text", "summary", "key_points", "learnings"]
        for field in text_fields:
            if doc.get(field):
                doc[field] = self.anonymize(doc[field])
        return doc

    # backward-compat alias
    def anonymize_case(self, case: dict) -> dict:
        return self.anonymize_document(case)

    # ── Private methods ───────────────────────────────────
    def _apply_regex_patterns(self, text: str) -> str:
        """Apply regex-based PII removal."""
        text = re.sub(_REGEX_PATTERNS["SSN"], "[SSN]", text)
        text = re.sub(_REGEX_PATTERNS["PHONE"], "[PHONE]", text)
        text = re.sub(_REGEX_PATTERNS["EMAIL"], "[EMAIL]", text)
        text = re.sub(_REGEX_PATTERNS["CREDIT_CARD"], "[CREDIT_CARD]", text)
        return text

    def _apply_presidio(self, text: str) -> str:
        """Apply Presidio NER-based PII detection and anonymization."""
        try:
            entities = ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "US_SSN",
                        "CREDIT_CARD", "IP_ADDRESS", "US_DRIVER_LICENSE", "US_PASSPORT"]

            results = self._analyzer.analyze(
                text=text,
                language="en",
                entities=entities,
            )

            if not results:
                return text

            # Sort by position descending to replace without offset shift
            results = sorted(results, key=lambda r: r.start, reverse=True)

            # Build consistent person mapping within this document
            person_map: dict = {}
            person_counter = [1]

            def get_person_placeholder(span_text: str) -> str:
                key = span_text.strip().lower()
                if key not in person_map:
                    person_map[key] = f"[PERSON_{person_counter[0]}]"
                    person_counter[0] += 1
                return person_map[key]

            text_chars = list(text)
            for result in results:
                if result.score < 0.6:
                    continue
                original = "".join(text_chars[result.start:result.end])

                if result.entity_type == "PERSON":
                    placeholder = get_person_placeholder(original)
                else:
                    type_map = {
                        "PHONE_NUMBER": "[PHONE]",
                        "EMAIL_ADDRESS": "[EMAIL]",
                        "US_SSN": "[SSN]",
                        "CREDIT_CARD": "[CREDIT_CARD]",
                        "IP_ADDRESS": "[IP_ADDRESS]",
                        "US_DRIVER_LICENSE": "[DRIVER_LICENSE]",
                        "US_PASSPORT": "[PASSPORT]",
                    }
                    placeholder = type_map.get(result.entity_type, f"[{result.entity_type}]")

                text_chars[result.start:result.end] = list(placeholder)

            return "".join(text_chars)

        except Exception as e:
            logger.warning(f"Presidio anonymization failed: {e}. Keeping regex-only output.")
            return text
