"""
Tests for SMEAgent — offline tests using mocked LLM and retriever.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


class TestSMEAgentOffline:
    """Tests that don't require LLM API calls."""

    def _make_agent_with_mocks(self, kb_ready: bool = True):
        """Create an SMEAgent with mocked LLM and retriever."""
        from src.agents.sme_agent import SMEAgent

        agent = SMEAgent.__new__(SMEAgent)

        # Mock LLM
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = (
            "Based on the retrieved documents, neural networks are computational systems "
            "inspired by biological brains. According to [Wikipedia — Neural Networks], "
            "they consist of layers of interconnected nodes."
        )
        mock_llm.invoke.return_value = mock_response
        agent.llm = mock_llm

        # Mock retriever
        mock_retriever = MagicMock()
        mock_retriever.is_ready.return_value = kb_ready
        mock_retriever.retrieve.return_value = (
            "## KEY INSIGHTS\n\nNeural networks learn from data.\n",
            [{"title": "Neural Networks", "source_name": "Wikipedia",
              "score": 0.95, "url": "https://en.wikipedia.org/wiki/Neural_network",
              "date": "2024-01-01", "author": "Wikipedia contributors",
              "topics": "neural networks,deep learning", "collection": "learnings"}]
        )
        mock_retriever.knowledge_base_stats.return_value = {"documents": 10, "learnings": 5, "summaries": 3}
        mock_retriever.k_documents = 4
        mock_retriever.k_learnings = 3
        agent.retriever = mock_retriever

        from langchain_core.messages import SystemMessage
        from src.config import SUBJECT, SUBJECT_DESCRIPTION
        from src.agents.sme_agent import _SYSTEM_PROMPT_TEMPLATE, _LANGUAGE_INSTRUCTIONS
        agent.history = [SystemMessage(content=_SYSTEM_PROMPT_TEMPLATE.format(
            subject=SUBJECT, subject_description=SUBJECT_DESCRIPTION,
            language_instruction=_LANGUAGE_INSTRUCTIONS.get("en", ""),
        ))]
        agent._kb_ready = kb_ready
        agent.k_documents = 4
        agent.k_learnings = 3

        return agent

    def test_basic_chat(self):
        agent = self._make_agent_with_mocks()
        answer = agent.chat("What is a neural network?")
        assert isinstance(answer, str)
        assert len(answer) > 20
        assert "neural network" in answer.lower()

    def test_chat_with_sources(self):
        agent = self._make_agent_with_mocks()
        answer, sources = agent.chat_with_sources("How do neural networks learn?")
        assert isinstance(answer, str)
        assert isinstance(sources, list)
        assert len(sources) > 0
        assert sources[0]["title"] == "Neural Networks"

    def test_reset_clears_history(self):
        agent = self._make_agent_with_mocks()
        agent.chat("First question")
        agent.chat("Follow-up question")
        assert len(agent.history) > 1
        agent.reset()
        assert len(agent.history) == 1

    def test_multi_turn_conversation(self):
        agent = self._make_agent_with_mocks()
        answer1 = agent.chat("What is deep learning?")
        assert answer1
        answer2 = agent.chat("How does it differ from machine learning?")
        assert answer2
        # system + 2 human + 2 ai = 5
        assert len(agent.history) == 5

    def test_empty_kb_mode(self):
        """Agent should still work even if KB is empty."""
        agent = self._make_agent_with_mocks(kb_ready=False)
        answer = agent.chat("Tell me about transformers")
        assert isinstance(answer, str)

    def test_kb_stats(self):
        agent = self._make_agent_with_mocks()
        stats = agent.kb_stats
        assert isinstance(stats, dict)
        assert "documents" in stats

    def test_build_augmented_message(self):
        from src.agents.sme_agent import SMEAgent
        question = "What is backpropagation?"
        context = "## KEY INSIGHTS\n\nBackpropagation uses the chain rule."
        message = SMEAgent._build_augmented_message(question, context)
        assert "What is backpropagation?" in message
        assert "RETRIEVED DOCUMENTS" in message
        assert "chain rule" in message
