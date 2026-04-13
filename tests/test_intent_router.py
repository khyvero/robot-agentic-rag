"""
Unit tests for Intent Router Agent
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from core.agents.intent_router import IntentRouterAgent
from core.agents.types import RouterInput, RouterDecision


class TestIntentRouterAgent:
    """Test suite for Intent Router Agent"""

    @pytest.fixture
    def mock_kb(self):
        # create a mock knowledgebase
        kb = Mock()
        return kb

    @pytest.fixture
    def router_agent(self, mock_kb):
        # create an intentrouteragent with mock kb
        return IntentRouterAgent(mock_kb)

    def test_exact_match_route(self, router_agent, mock_kb):
        # test routing for exact match (distance < 1.1)
        # Mock KB to return exact match
        mock_kb.query_declarative.return_value = (
            '{"mission_name": "Pick up blood", "steps": ["Pick blood"]}',
            0.85  # Distance < 1.1
        )

        input_data = RouterInput(
            user_prompt="Pick up blood",
            valid_objects=["test_tube_blood", "beaker_water"]
        )

        decision = router_agent.route(input_data)

        assert decision.route == "EXACT_MATCH"
        assert decision.intent_text is not None
        assert decision.distance == 0.85
        assert "Pick up blood" in decision.intent_text

    def test_ambiguous_route(self, router_agent, mock_kb):
        # test routing for ambiguous input (1.1 <= distance < 1.6)
        # Mock KB to return ambiguous match
        mock_kb.query_declarative.return_value = (
            '{"mission_name": "Mix samples"}',
            1.3  # 1.1 <= distance < 1.6
        )

        # Mock candidates
        mock_kb.get_candidates.return_value = [
            ('{"mission_name": "Mix blood and DNA"}', 1.3),
            ('{"mission_name": "Mix phenol"}', 1.4),
            ('{"mission_name": "Shake samples"}', 1.5)
        ]

        input_data = RouterInput(
            user_prompt="Mix them",
            valid_objects=["test_tube_blood", "test_tube_DNA"]
        )

        decision = router_agent.route(input_data)

        assert decision.route == "AMBIGUOUS"
        assert decision.distance == 1.3
        assert decision.candidates is not None
        assert len(decision.candidates) == 3

    @patch('core.agents.intent_router.ollama.chat')
    def test_novel_task_route(self, mock_chat, router_agent, mock_kb):
        # test routing for novel task (distance >= 1.6, llm says task)
        # Mock KB to return high distance
        mock_kb.query_declarative.return_value = (
            '{"mission_name": "Unrelated"}',
            2.5  # Distance >= 1.6
        )

        # Mock LLM to classify as TASK
        mock_chat.return_value = {
            'message': {'content': 'TASK'}
        }

        input_data = RouterInput(
            user_prompt="Do a complex new procedure",
            valid_objects=["test_tube_blood", "beaker_water"]
        )

        decision = router_agent.route(input_data)

        assert decision.route == "NOVEL_TASK"
        assert decision.distance == 2.5
        mock_chat.assert_called_once()

    @patch('core.agents.intent_router.ollama.chat')
    def test_not_task_route(self, mock_chat, router_agent, mock_kb):
        # test routing for non-task input (distance >= 1.6, llm says not_task)
        # Mock KB to return high distance
        mock_kb.query_declarative.return_value = (
            '{"mission_name": "Unrelated"}',
            3.0  # Distance >= 1.6
        )

        # Mock LLM to classify as NOT_TASK
        mock_chat.return_value = {
            'message': {'content': 'NOT_TASK'}
        }

        input_data = RouterInput(
            user_prompt="Hello, how are you?",
            valid_objects=["test_tube_blood"]
        )

        decision = router_agent.route(input_data)

        assert decision.route == "NOT_TASK"
        assert decision.distance == 3.0
        mock_chat.assert_called_once()

    def test_threshold_exact_boundary(self, router_agent, mock_kb):
        # test exact threshold boundary (distance = 1.1)
        # Mock KB to return distance exactly at threshold
        mock_kb.query_declarative.return_value = (
            '{"mission_name": "Test"}',
            1.1  # Exactly at threshold
        )

        # Mock candidates for ambiguous case
        mock_kb.get_candidates.return_value = [
            ('{"mission_name": "Test 1"}', 1.1),
            ('{"mission_name": "Test 2"}', 1.2)
        ]

        input_data = RouterInput(
            user_prompt="Test prompt",
            valid_objects=["test_tube_blood"]
        )

        decision = router_agent.route(input_data)

        # At exactly 1.1, should be AMBIGUOUS (not EXACT_MATCH)
        assert decision.route == "AMBIGUOUS"

    def test_threshold_ambiguous_boundary(self, router_agent, mock_kb):
        # test ambiguous threshold boundary (distance = 1.6)
        # Mock KB to return distance exactly at threshold
        mock_kb.query_declarative.return_value = (
            '{"mission_name": "Test"}',
            1.6  # Exactly at threshold
        )

        # Mock LLM for classification
        with patch('core.agents.intent_router.ollama.chat') as mock_chat:
            mock_chat.return_value = {
                'message': {'content': 'TASK'}
            }

            input_data = RouterInput(
                user_prompt="Test prompt",
                valid_objects=["test_tube_blood"]
            )

            decision = router_agent.route(input_data)

            # At exactly 1.6, should go to LLM classification
            assert decision.route == "NOVEL_TASK"

    def test_pydantic_validation(self):
        # test pydantic validation for routerinput
        # Valid input
        input_data = RouterInput(
            user_prompt="Test",
            valid_objects=["obj1", "obj2"]
        )
        assert input_data.user_prompt == "Test"

        # Invalid input (missing required field)
        with pytest.raises(Exception):  # Pydantic ValidationError
            RouterInput(user_prompt="Test")  # Missing valid_objects


if __name__ == "__main__":
    pytest.main([__file__, "-v"])