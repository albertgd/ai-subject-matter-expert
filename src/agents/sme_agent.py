"""
SME Agent — Subject Matter Expert conversational agent.

Grounded RAG agent that:
  1. Retrieves relevant documents and insights from ChromaDB
  2. Injects them into the LLM context
  3. Generates well-cited, grounded answers
  4. Maintains multi-turn conversation history

Works for ANY subject configured in .env (SUBJECT, SUBJECT_DESCRIPTION).
Supports Claude (Anthropic), GPT-4 (OpenAI), and Gemini (Google).

Usage:
    agent = SMEAgent()
    answer = agent.chat("What is backpropagation?")
    answer = agent.chat("How does it relate to gradient descent?")  # follow-up
    agent.reset()
"""

import logging
import sys
from typing import Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.config import SUBJECT, SUBJECT_DESCRIPTION, PRIMARY_LLM, GROQ_API_KEY, LANGUAGE, active_llm_provider
from src.rag.retriever import Retriever

logger = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT_TEMPLATE = """You are an expert AI assistant specializing in {subject_description}.

You have access to a knowledge base gathered from public internet sources about {subject}.

RULES:
1. Ground your answers in the retrieved documents provided in each message.
2. Cite specific sources by title and origin (e.g., "According to [Wikipedia — Neural Networks]...").
3. If the retrieved documents don't fully cover the question, clearly say so and supplement with your general knowledge, clearly distinguishing between the two.
4. Do NOT fabricate source names or misattribute information.
5. Structure your answers clearly using headers and bullet points where helpful.
6. Maintain context from earlier in the conversation for follow-up questions.

Your responses should be:
- Accurate and well-sourced
- Clear and well-organized
- Appropriately detailed based on the complexity of the question
{language_instruction}"""

_LANGUAGE_INSTRUCTIONS = {
    "es": (
        "\nIMPORTANTE: Responde SIEMPRE en español, independientemente del idioma "
        "de la pregunta. Usa terminología jurídica/técnica española cuando sea aplicable."
    ),
    "en": "",
}

_NO_KB_WARNING = """
NOTE: The knowledge base is empty or not yet built.
Running in general knowledge mode — answers are based on training data only.
To build the knowledge base, run: python scripts/build_rag.py

"""


class SMEAgent:
    """
    Conversational Subject Matter Expert agent for any configured subject.

    Args:
        llm_provider:  "anthropic" | "openai" | "google" | "auto"
        model_name:    Override model name
        k_documents:   Number of document chunks to retrieve per turn
        k_learnings:   Number of insight chunks to retrieve
        retriever:     Optional pre-built Retriever instance
    """

    def __init__(
        self,
        llm_provider: str = "auto",
        model_name: Optional[str] = None,
        k_documents: int = 4,
        k_learnings: int = 3,
        retriever: Optional[Retriever] = None,
    ):
        self.llm = self._init_llm(llm_provider, model_name)
        self.retriever = retriever or self._init_retriever(k_documents, k_learnings)
        self.k_documents = k_documents
        self.k_learnings = k_learnings

        self.history: List = [SystemMessage(
            content=_SYSTEM_PROMPT_TEMPLATE.format(
                subject=SUBJECT,
                subject_description=SUBJECT_DESCRIPTION,
                language_instruction=_LANGUAGE_INSTRUCTIONS.get(LANGUAGE, ""),
            )
        )]
        self._kb_ready = self.retriever.is_ready()

        if not self._kb_ready:
            logger.warning("Knowledge base is empty. Run 'python scripts/build_rag.py' to populate it.")

    # ── Main interface ────────────────────────────────────
    def chat(self, question: str, verbose: bool = False) -> str:
        """Ask a question and receive a grounded expert answer."""
        context, sources = self._retrieve(question, verbose=verbose)
        augmented = self._build_augmented_message(question, context)

        self.history.append(HumanMessage(content=augmented))
        response = self.llm.invoke(self.history)
        answer = response.content

        # Store clean question (not augmented) for follow-ups
        self.history[-1] = HumanMessage(content=question)
        self.history.append(AIMessage(content=answer))

        return answer

    def chat_with_sources(self, question: str, verbose: bool = False) -> Tuple[str, List[Dict]]:
        """Same as chat() but also returns the source list."""
        context, sources = self._retrieve(question, verbose=verbose)
        augmented = self._build_augmented_message(question, context)

        self.history.append(HumanMessage(content=augmented))
        response = self.llm.invoke(self.history)
        answer = response.content

        self.history[-1] = HumanMessage(content=question)
        self.history.append(AIMessage(content=answer))

        return answer, sources

    def reset(self):
        """Clear conversation history (start new session)."""
        self.history = [SystemMessage(
            content=_SYSTEM_PROMPT_TEMPLATE.format(
                subject=SUBJECT,
                subject_description=SUBJECT_DESCRIPTION,
                language_instruction=_LANGUAGE_INSTRUCTIONS.get(LANGUAGE, ""),
            )
        )]

    # ── Retrieval ─────────────────────────────────────────
    def _retrieve(self, query: str, verbose: bool = False) -> Tuple[str, List[Dict]]:
        if not self._kb_ready:
            return _NO_KB_WARNING, []
        self.retriever.k_documents = self.k_documents
        self.retriever.k_learnings = self.k_learnings
        return self.retriever.retrieve(query, verbose=verbose)

    @staticmethod
    def _build_augmented_message(question: str, context: str) -> str:
        return (
            f"RETRIEVED DOCUMENTS FROM KNOWLEDGE BASE:\n\n"
            f"{context}\n\n"
            f"USER QUESTION:\n{question}"
        )

    # ── Initialization helpers ────────────────────────────
    @staticmethod
    def _init_llm(provider: str, model_name: Optional[str]):
        if provider == "auto":
            provider = active_llm_provider()

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model_name or PRIMARY_LLM,
                temperature=0.2,
                max_tokens=3000,
            )
        elif provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=model_name or "gemini-2.0-flash",
                temperature=0.2,
            )
        elif provider == "groq":
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=model_name or "llama-3.3-70b-versatile",
                temperature=0.2,
                api_key=GROQ_API_KEY,
            )
        else:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model_name or "gpt-4o",
                temperature=0.2,
            )

    @staticmethod
    def _init_retriever(k_documents: int, k_learnings: int) -> Retriever:
        logger.info("Initializing retriever and loading knowledge base...")
        return Retriever(k_documents=k_documents, k_learnings=k_learnings)

    # ── Convenience ───────────────────────────────────────
    @property
    def kb_stats(self) -> Dict[str, int]:
        """Knowledge base collection sizes."""
        return self.retriever.knowledge_base_stats()

    def explore_topic(self, topic: str, n: int = 10) -> List[Dict]:
        """Browse top N documents for a topic (no LLM call)."""
        return self.retriever.retrieve_for_topic(topic, n_results=n)


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print(f"\n{'='*60}")
    print(f"  AI Subject Matter Expert — {SUBJECT.title()}")
    print(f"{'='*60}\n")

    agent = SMEAgent()

    stats = agent.kb_stats
    total = sum(stats.values())
    if total > 0:
        print(f"Knowledge base ready: {stats}")
    else:
        print("WARNING: Knowledge base is empty.")
        print("Run: python scripts/build_rag.py\n")

    print("Type your question (or 'quit' to exit, 'reset' to clear history).\n")

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"Question: {question}\n")
        print(agent.chat(question, verbose=True))
    else:
        while True:
            try:
                q = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.")
                break

            if not q:
                continue
            if q.lower() in ("quit", "exit", "q"):
                print("Goodbye.")
                break
            if q.lower() == "reset":
                agent.reset()
                print("Conversation reset.\n")
                continue

            print("\nAssistant:")
            print(agent.chat(q))
            print()
