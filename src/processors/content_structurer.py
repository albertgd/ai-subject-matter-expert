"""
Content Structurer — use an LLM to extract structured knowledge from raw web content.

Works for any subject. Takes a raw document (text from any web page) and extracts:
  - summary:    Concise overview of the content
  - key_points: Bullet-point distillation of the most important information
  - topics:     Subject tags relevant to the document
  - learnings:  Reusable knowledge nuggets for RAG retrieval
"""

import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """You are a knowledge extraction assistant. Extract structured information from the text below.

The text is about the subject: {subject}

Extract the following fields and return ONLY a valid JSON object:

1. "summary": A concise paragraph (3-5 sentences) summarizing the main content.

2. "key_points": A numbered list of 5-8 most important facts, concepts, or insights from the text. Each point should be self-contained and useful on its own.

3. "learnings": 3-6 reusable knowledge statements that could help answer questions about {subject}. These should be general, distilled insights rather than document-specific details.

4. "topics": A list of 3-6 topic tags that describe what this document covers (short noun phrases, e.g. ["neural networks", "backpropagation", "supervised learning"]).

Respond with ONLY a valid JSON object. No text outside the JSON.

TEXT:
{text}
"""


class ContentStructurer:
    """
    Extracts structured knowledge from raw document text using an LLM.

    Usage:
        structurer = ContentStructurer()
        doc = structurer.structure({"source_id": "...", "text": "..."})
    """

    def __init__(self, llm_provider: str = "auto", model_name: Optional[str] = None):
        self.llm = self._init_llm(llm_provider, model_name)

    def _init_llm(self, provider: str, model_name: Optional[str]):
        from src.config import PRIMARY_LLM, active_llm_provider

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

    def structure(self, document: Dict, max_text_chars: int = 15_000) -> Dict:
        """
        Extract structured fields from a raw document dict.

        Returns the document dict with added fields:
          summary, key_points, learnings, topics, structured (bool)
        """
        from src.config import SUBJECT

        text = document.get("text", "").strip()
        if not text:
            logger.warning(f"Document {document.get('source_id', '?')} has no text, skipping.")
            return document

        # Truncate to avoid hitting context limits (keep start + end)
        if len(text) > max_text_chars:
            half = max_text_chars // 2
            text = text[:half] + "\n\n[...MIDDLE OMITTED...]\n\n" + text[-half:]

        prompt = _EXTRACTION_PROMPT.format(subject=SUBJECT, text=text)

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

            document = dict(document)
            document["summary"] = extracted.get("summary", "")
            document["key_points"] = extracted.get("key_points", "")
            document["learnings"] = extracted.get("learnings", "")
            document["topics"] = extracted.get("topics", [])
            document["structured"] = True
            return document

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed for {document.get('source_id', '?')}: {e}")
            document = dict(document)
            document["summary"] = text[:500]
            document["structured"] = False
            return document
        except Exception as e:
            logger.error(f"Structuring failed for {document.get('source_id', '?')}: {e}")
            document = dict(document)
            document["structured"] = False
            return document

    def structure_batch(
        self, documents: List[Dict], skip_structured: bool = True
    ) -> List[Dict]:
        """Structure a batch of documents."""
        results = []
        total = len(documents)
        for i, doc in enumerate(documents, 1):
            if skip_structured and doc.get("structured"):
                results.append(doc)
                continue
            logger.info(f"Structuring {i}/{total}: {doc.get('source_id', '?')}")
            results.append(self.structure(doc))
        return results
