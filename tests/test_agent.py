"""
Tests for SMEAgent — offline tests using mocked LLM and retriever.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


class TestSMEAgentOffline:
    """Tests that don't require LLM API calls."""

    def _make_agent_with_mocks(self, kb_ready: bool = True):
        """Create an SMEAgent with mocked LLM and retriever."""
        from src.agents.sme_agent import SMEAgent

        # Create agent without calling __init__
        agent = SMEAgent.__new__(SMEAgent)

        # Mock LLM
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = (
            "Based on the retrieved cases, child custody is determined by the "
            "best interests of the child standard. In Anderson v. Anderson [TestSource], "
            "the court held that the primary caregiver's role is a key factor."
        )
        mock_llm.invoke.return_value = mock_response
        agent.llm = mock_llm

        # Mock retriever
        mock_retriever = MagicMock()
        mock_retriever.is_ready.return_value = kb_ready
        mock_retriever.retrieve.return_value = (
            "## LEGAL PRINCIPLES\n\nBest interests of child is paramount.\n",
            [{"title": "Anderson v. Anderson", "source_name": "TestSource",
              "score": 0.95, "url": "", "date": "2023-01-01", "court": "CA",
              "section": "learnings", "practice_areas": "custody"}]
        )
        mock_retriever.knowledge_base_stats.return_value = {"opinions": 5, "learnings": 3}
        mock_retriever.k_opinions = 4
        mock_retriever.k_learnings = 3
        agent.retriever = mock_retriever

        # Initialize conversation history
        from langchain_core.messages import SystemMessage
        from src.config import DOMAIN, DOMAIN_DESCRIPTION
        from src.agents.sme_agent import _SYSTEM_PROMPT_TEMPLATE
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            domain=DOMAIN, domain_description=DOMAIN_DESCRIPTION
        )
        agent.history = [SystemMessage(content=system_prompt)]
        agent._kb_ready = kb_ready
        agent.k_opinions = 4
        agent.k_learnings = 3

        return agent

    def test_basic_chat(self):
        agent = self._make_agent_with_mocks()
        answer = agent.chat("What factors determine child custody?")
        assert isinstance(answer, str)
        assert len(answer) > 20
        assert "best interests" in answer.lower()

    def test_chat_with_sources(self):
        agent = self._make_agent_with_mocks()
        answer, sources = agent.chat_with_sources("How is custody determined?")
        assert isinstance(answer, str)
        assert isinstance(sources, list)
        assert len(sources) > 0
        assert sources[0]["title"] == "Anderson v. Anderson"

    def test_reset_clears_history(self):
        agent = self._make_agent_with_mocks()
        agent.chat("First question about custody")
        agent.chat("Follow-up about relocation")

        initial_length = len(agent.history)
        assert initial_length > 1  # System + messages

        agent.reset()
        # After reset, only system message should remain
        assert len(agent.history) == 1

    def test_multi_turn_conversation(self):
        agent = self._make_agent_with_mocks()

        # First turn
        answer1 = agent.chat("What is joint custody?")
        assert answer1

        # Second turn (follow-up)
        answer2 = agent.chat("How does relocation affect that?")
        assert answer2

        # History should have: system + 2 human + 2 ai = 5 messages
        assert len(agent.history) == 5

    def test_empty_kb_mode(self):
        """Agent should still work even if KB is empty."""
        agent = self._make_agent_with_mocks(kb_ready=False)
        # Should not raise even with empty KB
        answer = agent.chat("Tell me about divorce law")
        assert isinstance(answer, str)

    def test_kb_stats(self):
        agent = self._make_agent_with_mocks()
        stats = agent.kb_stats
        assert isinstance(stats, dict)
        assert "opinions" in stats

    def test_build_augmented_message(self):
        from src.agents.sme_agent import SMEAgent
        question = "How is property divided?"
        context = "## CASE LAW\n\nIn Brown v. Brown, the court applied equitable distribution."
        message = SMEAgent._build_augmented_message(question, context)
        assert "How is property divided?" in message
        assert "RETRIEVED DOCUMENTS" in message
        assert "Brown v. Brown" in message
