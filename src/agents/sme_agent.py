"""
SME Agent — Subject Matter Expert conversational agent.

Grounded RAG agent that:
  1. Retrieves relevant case law and legal principles from ChromaDB
  2. Injects them into the LLM context
  3. Generates well-cited, grounded answers
  4. Maintains multi-turn conversation history
  5. Cites source cases and principles in every response

Supports Claude (Anthropic), GPT-4 (OpenAI), and Gemini (Google).
Auto-detects available API keys.

Usage:
    agent = SMEAgent()
    answer = agent.chat("What factors do courts consider for child custody?")
    answer = agent.chat("How does relocation affect custody arrangements?")  # follow-up
    agent.reset()
"""

import logging
import sys
from typing import Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.config import (
    DOMAIN, DOMAIN_DESCRIPTION, PRIMARY_LLM, active_llm_provider
)
from src.rag.retriever import Retriever

logger = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM_PROMPT_TEMPLATE = """You are an expert AI assistant specializing in {domain_description}.

You have access to a knowledge base of {domain} court cases, legal opinions, and distilled legal principles gathered from public sources.

STRICT RULES:
1. Always ground your answers in the retrieved documents provided below.
2. Cite specific cases by their title and source (e.g., "In Smith v. Jones [CourtListener]...").
3. If the retrieved documents don't cover the question, clearly state what's not in the knowledge base and provide general legal context with appropriate caveats.
4. Do NOT fabricate case names, citations, or legal rules.
5. Structure your answers clearly: state the general rule, then cite supporting cases.
6. Maintain context from earlier in the conversation for follow-up questions.
7. Always include a disclaimer that this is not legal advice and users should consult a qualified attorney.

Your responses should be:
- Professional and precise
- Well-organized (use headers when appropriate)
- Grounded in real case law from the knowledge base
- Appropriately cautious about legal advice"""


_NO_KB_WARNING = """
NOTE: The knowledge base appears to be empty or not yet built.
Running in general knowledge mode — answers are based on my training data only.
To build the knowledge base, run: python scripts/build_rag.py

"""


class SMEAgent:
    """
    Conversational Subject Matter Expert agent for {domain} law.

    Maintains conversation history and retrieves relevant cases
    from the vector store for every user turn.

    Args:
        llm_provider:   "anthropic" | "openai" | "google" | "auto"
        model_name:     Override model name
        k_opinions:     Number of opinion chunks to retrieve per turn
        k_learnings:    Number of learning/principle chunks to retrieve
        retriever:      Optional pre-built Retriever instance
    """

    def __init__(
        self,
        llm_provider: str = "auto",
        model_name: Optional[str] = None,
        k_opinions: int = 4,
        k_learnings: int = 3,
        retriever: Optional[Retriever] = None,
    ):
        self.llm = self._init_llm(llm_provider, model_name)
        self.retriever = retriever or self._init_retriever(k_opinions, k_learnings)
        self.k_opinions = k_opinions
        self.k_learnings = k_learnings

        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            domain=DOMAIN,
            domain_description=DOMAIN_DESCRIPTION,
        )
        self.history: List = [SystemMessage(content=system_prompt)]
        self._kb_ready = self.retriever.is_ready()

        if not self._kb_ready:
            logger.warning("Knowledge base is empty. Run 'python scripts/build_rag.py' to populate it.")

    # ── Main interface ────────────────────────────────────
    def chat(self, question: str, verbose: bool = False) -> str:
        """
        Ask a question and receive a grounded expert answer.

        Args:
            question:  User's question (English).
            verbose:   Log retrieved sources.

        Returns:
            Agent's answer as a string.
        """
        # 1. Retrieve relevant context
        context, sources = self._retrieve(question, verbose=verbose)

        # 2. Build augmented user message with retrieved context
        augmented = self._build_augmented_message(question, context)

        # 3. Invoke LLM with full history
        self.history.append(HumanMessage(content=augmented))
        response = self.llm.invoke(self.history)
        answer = response.content

        # 4. Store clean question (not augmented) in history for follow-ups
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
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            domain=DOMAIN,
            domain_description=DOMAIN_DESCRIPTION,
        )
        self.history = [SystemMessage(content=system_prompt)]

    # ── Retrieval ─────────────────────────────────────────
    def _retrieve(self, query: str, verbose: bool = False) -> Tuple[str, List[Dict]]:
        """Retrieve context from the vector store."""
        if not self._kb_ready:
            return _NO_KB_WARNING, []

        # Update k values in case they were changed externally
        self.retriever.k_opinions = self.k_opinions
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
                model=model_name or "gemini-2.5-flash",
                temperature=0.2,
            )
        else:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model_name or "gpt-4o",
                temperature=0.2,
            )

    @staticmethod
    def _init_retriever(k_opinions: int, k_learnings: int) -> Retriever:
        logger.info("Initializing retriever and loading knowledge base...")
        return Retriever(k_opinions=k_opinions, k_learnings=k_learnings)

    # ── Convenience ───────────────────────────────────────
    @property
    def kb_stats(self) -> Dict[str, int]:
        """Knowledge base collection sizes."""
        return self.retriever.knowledge_base_stats()

    def explore_topic(self, topic: str, n: int = 10) -> List[Dict]:
        """Browse top N cases for a topic (no LLM call)."""
        return self.retriever.retrieve_for_topic(topic, n_results=n)


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print(f"\n{'='*60}")
    print(f"  AI Subject Matter Expert — {DOMAIN.title()} Law")
    print(f"{'='*60}\n")

    agent = SMEAgent()

    stats = agent.kb_stats
    total = sum(stats.values())
    if total > 0:
        print(f"Knowledge base ready: {stats}")
    else:
        print("WARNING: Knowledge base is empty.")
        print("Run: python scripts/build_rag.py\n")

    print("Type your question (or 'quit' to exit).\n")

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
