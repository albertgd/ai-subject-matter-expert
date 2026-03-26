"""
Case Structurer — Use an LLM to extract structured knowledge from raw case text.

Takes a raw legal case (full opinion text) and extracts:
  - facts:       The relevant facts / background of the case
  - ruling:      The court's final decision / ruling
  - reasoning:   The legal reasoning and principles applied
  - learnings:   Distilled, agent-ready knowledge points for RAG

Also generates a concise summary suitable for embedding.
"""

import json
import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Extraction prompt template
_EXTRACTION_PROMPT = """You are a legal analyst extracting structured information from a court opinion.

Given the following court opinion, extract the following sections in JSON format:

1. "facts": The key facts and background of the case (2-4 paragraphs). Focus on what happened, who the parties are (use generic terms like "Husband", "Wife", "Parent 1", "Parent 2" — do NOT use real names), and what legal question is at issue.

2. "ruling": The court's decision/ruling (1-2 paragraphs). What did the court ultimately decide?

3. "reasoning": The legal reasoning, principles, and doctrines applied by the court (2-4 paragraphs). What legal standards were used? What precedents cited?

4. "learnings": 5-8 bullet points of distilled legal knowledge that a lawyer or AI assistant could apply to future similar cases. These should be general legal principles extracted from this case. Format as a numbered list.

5. "summary": A single paragraph (3-5 sentences) summarizing the case for a legal research index.

6. "practice_areas": A list of 3-5 relevant legal practice area tags from this list: [divorce, custody, alimony, property_division, child_support, spousal_support, prenuptial, modification, enforcement, contempt, domestic_violence, paternity, adoption, guardianship, annulment]

Respond with ONLY a valid JSON object. Do not include any text outside the JSON.

COURT OPINION:
{text}
"""


class CaseStructurer:
    """
    Extracts structured knowledge from raw legal case text using an LLM.

    Usage:
        structurer = CaseStructurer()
        structured = structurer.structure(case_dict)
    """

    def __init__(self, llm_provider: str = "auto", model_name: Optional[str] = None):
        self.llm = self._init_llm(llm_provider, model_name)

    def _init_llm(self, provider: str, model_name: Optional[str]):
        """Initialize LLM client."""
        from src.config import (
            ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY,
            PRIMARY_LLM, active_llm_provider
        )

        if provider == "auto":
            provider = active_llm_provider()

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model_name or PRIMARY_LLM,
                temperature=0.1,
                max_tokens=4096,
            )
        elif provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model_name or "gemini-2.5-flash",
                temperature=0.1,
            )
        else:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model_name or "gpt-4o-mini",
                temperature=0.1,
            )

    def structure(self, case: Dict, max_text_chars: int = 15_000) -> Dict:
        """
        Extract structured fields from a raw case dict.

        Args:
            case:           Raw case dict with at minimum a 'text' field.
            max_text_chars: Truncate input text to this length to control token usage.

        Returns:
            Updated case dict with added structured fields:
            facts, ruling, reasoning, learnings, summary, practice_areas
        """
        text = case.get("text", "").strip()
        if not text:
            logger.warning(f"Case {case.get('source_id', '?')} has no text, skipping.")
            return case

        # Truncate to avoid hitting context limits
        if len(text) > max_text_chars:
            # Keep beginning and end (most opinions have facts at start, ruling at end)
            half = max_text_chars // 2
            text = text[:half] + "\n\n[...MIDDLE OF OPINION OMITTED...]\n\n" + text[-half:]

        prompt = _EXTRACTION_PROMPT.format(text=text)

        try:
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            raw_json = response.content.strip()

            # Strip markdown code fences if present
            if raw_json.startswith("```"):
                raw_json = raw_json.split("```")[1]
                if raw_json.startswith("json"):
                    raw_json = raw_json[4:]

            extracted = json.loads(raw_json)

            case = dict(case)
            case["facts"] = extracted.get("facts", "")
            case["ruling"] = extracted.get("ruling", "")
            case["reasoning"] = extracted.get("reasoning", "")
            case["learnings"] = extracted.get("learnings", "")
            case["summary"] = extracted.get("summary", "")
            case["practice_areas"] = extracted.get("practice_areas", [])
            case["structured"] = True

            return case

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed for {case.get('source_id', '?')}: {e}")
            # Fallback: use raw text as summary
            case = dict(case)
            case["summary"] = text[:500]
            case["structured"] = False
            return case
        except Exception as e:
            logger.error(f"Structuring failed for {case.get('source_id', '?')}: {e}")
            case = dict(case)
            case["structured"] = False
            return case

    def structure_batch(self, cases: list, skip_structured: bool = True) -> list:
        """
        Structure a batch of cases.

        Args:
            cases:            List of raw case dicts
            skip_structured:  Skip cases that already have structured=True
        """
        results = []
        total = len(cases)
        for i, case in enumerate(cases, 1):
            if skip_structured and case.get("structured"):
                results.append(case)
                continue
            logger.info(f"Structuring case {i}/{total}: {case.get('source_id', '?')}")
            structured = self.structure(case)
            results.append(structured)
        return results
