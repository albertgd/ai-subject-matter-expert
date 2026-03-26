"""
Text cleaner — normalize and clean raw scraped web text.

Handles:
  - Boilerplate removal (navigation, disclaimers, ads)
  - Whitespace normalization
  - Unicode normalization
  - Encoding fixes
  - Removal of very short / garbage documents
"""

import re
import unicodedata
from typing import Optional


# Common legal document boilerplate patterns to strip
_BOILERPLATE_PATTERNS = [
    # Court header boilerplate
    r"NOTICE[:：]\s*This slip opinion.{0,200}",
    r"FOR PUBLICATION.{0,100}\n",
    r"NOT FOR PUBLICATION.{0,100}\n",
    r"UNPUBLISHED.{0,100}\n",
    # Navigation / website chrome
    r"(?:Home|About|Contact|Search|Login|Register|Sign[- ]in|Sign[- ]up)\s*[\|>»]\s*",
    r"Skip to (?:main )?content\.?",
    r"JavaScript (?:is )?(?:required|disabled).{0,200}",
    # Subscription / paywall
    r"(?:Subscribe|Subscription|Premium|Paywall|Sign up).{0,200}(?:to access|for access).{0,200}",
    # Advertisement labels
    r"ADVERTISEMENT\s*",
    r"Sponsored[- ](?:Content|Link|By).{0,200}\n",
    # Justia-specific
    r"Justia Opinion Summary and Annotations.*?(?=\n[A-Z]|\Z)",
    # CourtListener-specific
    r"(?:This opinion|This case) (?:was|is) (?:decided|published|available).{0,300}\n",
    # Common footers
    r"(?:Copyright|All rights reserved|Terms of (?:Use|Service)|Privacy Policy).{0,200}",
    r"(?:Last updated|Last modified):\s*\w+ \d+, \d{4}.{0,100}",
    # Citation noise
    r"\[\d+\]\s*(?=\n)",  # Footnote markers at line start
    r"_+\n",              # Horizontal separators
    r"-{4,}\n",           # Long dashes
    r"={4,}\n",           # Long equals
]

_BOILERPLATE_RE = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _BOILERPLATE_PATTERNS]

# Minimum document length to keep
MIN_TEXT_LENGTH = 200


class TextCleaner:
    """
    Cleans and normalizes raw scraped text for further processing.

    Usage:
        cleaner = TextCleaner()
        clean = cleaner.clean(raw_text)
    """

    def clean(self, text: str) -> Optional[str]:
        """
        Full cleaning pipeline.

        Returns cleaned text, or None if the document is too short / garbage.
        """
        if not text or not text.strip():
            return None

        text = self._fix_encoding(text)
        text = self._normalize_unicode(text)
        text = self._remove_html_entities(text)
        text = self._strip_boilerplate(text)
        text = self._normalize_whitespace(text)
        text = self._remove_excessive_punctuation(text)
        text = self._normalize_whitespace(text)  # Second pass after punctuation removal

        if len(text) < MIN_TEXT_LENGTH:
            return None

        return text.strip()

    def clean_document(self, doc: dict) -> dict:
        """Clean the text field of a raw document dict. Returns updated dict."""
        cleaned = self.clean(doc.get("text", ""))
        if cleaned:
            doc = dict(doc)  # don't mutate original
            doc["text"] = cleaned
        return doc

    # backward-compat alias
    def clean_case(self, case: dict) -> dict:
        return self.clean_document(case)

    # ── Private methods ───────────────────────────────────
    @staticmethod
    def _fix_encoding(text: str) -> str:
        """Fix common encoding issues."""
        # Fix Windows-1252 characters incorrectly decoded as Latin-1
        replacements = {
            "\u2018": "'", "\u2019": "'",  # Curly single quotes
            "\u201c": '"', "\u201d": '"',  # Curly double quotes
            "\u2013": "-", "\u2014": "--", # En/em dashes
            "\u2026": "...",               # Ellipsis
            "\u00a0": " ",                 # Non-breaking space
            "\u200b": "",                  # Zero-width space
            "\ufeff": "",                  # BOM
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """Normalize Unicode to NFC form."""
        return unicodedata.normalize("NFC", text)

    @staticmethod
    def _remove_html_entities(text: str) -> str:
        """Remove residual HTML entities."""
        import html
        text = html.unescape(text)
        # Remove remaining entities like &amp; that slipped through
        text = re.sub(r"&[a-zA-Z][a-zA-Z0-9]*;", " ", text)
        text = re.sub(r"&#\d+;", " ", text)
        return text

    @staticmethod
    def _strip_boilerplate(text: str) -> str:
        """Remove common boilerplate patterns."""
        for pattern in _BOILERPLATE_RE:
            text = pattern.sub("", text)
        return text

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Normalize whitespace: collapse multiple spaces/blank lines."""
        # Collapse multiple spaces to single space
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse 3+ newlines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove spaces at start/end of lines
        text = re.sub(r"^ +| +$", "", text, flags=re.MULTILINE)
        return text

    @staticmethod
    def _remove_excessive_punctuation(text: str) -> str:
        """Remove lines consisting only of punctuation or symbols."""
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip lines that are just symbols/punctuation
            if stripped and re.match(r'^[^\w\s]{3,}$', stripped):
                continue
            # Skip very short lines that look like page numbers
            if stripped and re.match(r'^\d{1,3}$', stripped):
                continue
            cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    @staticmethod
    def truncate(text: str, max_chars: int = 100_000) -> str:
        """Truncate very long texts to max_chars (keep first + last)."""
        if len(text) <= max_chars:
            return text
        half = max_chars // 2
        return text[:half] + "\n\n[...TEXT TRUNCATED...]\n\n" + text[-half:]
