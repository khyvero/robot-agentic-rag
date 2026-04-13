"""
Unit tests for Action Extractor Agent
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock, mock_open
from core.agents.action_extractor import ActionExtractorAgent
from core.agents.types import ActionExtractionInput, ActionExtractionResult


class TestActionExtractorAgent:
    """Test suite for Action Extractor Agent"""

    @pytest.fixture
    def mock_kb(self):
        # create a mock knowledgebase
        kb = Mock()
        return kb

    @pytest.fixture
    def sample_actions(self):
        # sample action metadata
        return [
            {
                "action_name": "pick",
                "action_keywords": ["pick", "pick up", "grab", "take"],
                "description": "Picks up an object",
                "function_signature": "pick(obj)"
            },
            {
                "action_name": "place",
                "action_keywords": ["place", "put", "drop"],
                "description": "Places an object",
                "function_signature": "place(obj)"
            },
            {
                "action_name": "pour",
                "action_keywords": ["pour", "empty", "transfer"],
                "description": "Pours liquid",
                "function_signature": "pour(obj)"
            }
        ]

    @pytest.fixture
    def action_extractor(self, mock_kb, sample_actions):
        # create an actionextractoragent with mock kb
        # Mock the file reading for action loading
        mock_file_content = json.dumps(sample_actions)

        with patch('builtins.open', mock_open(read_data=mock_file_content)):
            agent = ActionExtractorAgent(mock_kb)
            # Manually set actions for testing
            agent.available_actions = sample_actions
            return agent

    @patch('core.agents.action_extractor.ollama.chat')
    def test_extract_actions_simple(self, mock_chat, action_extractor):
        # test basic action extraction
        # Mock LLM response
        mock_chat.return_value = {
            'message': {'content': '["pick", "place"]'}
        }

        input_data = ActionExtractionInput(
            user_prompt="Pick up blood and place it",
            intent_text=None,
            available_actions=action_extractor.available_actions
        )

        result = action_extractor.extract_actions(input_data)

        assert result.actions == ["pick", "place"]
        assert mock_chat.called

    @patch('core.agents.action_extractor.ollama.chat')
    def test_extract_actions_with_intent(self, mock_chat, action_extractor):
        # test action extraction with intent context
        # Mock LLM response
        mock_chat.return_value = {
            'message': {'content': '["pick", "pour"]'}
        }

        input_data = ActionExtractionInput(
            user_prompt="Pour the blood",
            intent_text="Recipe: 1. Pick blood, 2. Pour into beaker",
            available_actions=action_extractor.available_actions
        )

        result = action_extractor.extract_actions(input_data)

        assert result.actions == ["pick", "pour"]
        # Check that intent was included in the prompt
        call_args = mock_chat.call_args[1]['messages'][0]['content']
        assert "Recipe:" in call_args

    @patch('core.agents.action_extractor.ollama.chat')
    def test_extract_actions_with_markdown(self, mock_chat, action_extractor):
        # test handling of markdown code blocks in llm response
        # Mock LLM response with markdown
        mock_chat.return_value = {
            'message': {'content': '```json\n["pick", "place", "pour"]\n```'}
        }

        input_data = ActionExtractionInput(
            user_prompt="Pick, place, and pour",
            intent_text=None,
            available_actions=action_extractor.available_actions
        )

        result = action_extractor.extract_actions(input_data)

        assert result.actions == ["pick", "place", "pour"]

    @patch('core.agents.action_extractor.ollama.chat')
    def test_extract_actions_fallback(self, mock_chat, action_extractor):
        # test fallback extraction when json parsing fails
        # Mock LLM response with invalid JSON
        mock_chat.return_value = {
            'message': {'content': 'You should pick and pour the sample'}
        }

        input_data = ActionExtractionInput(
            user_prompt="Pick and pour",
            intent_text=None,
            available_actions=action_extractor.available_actions
        )

        result = action_extractor.extract_actions(input_data)

        # Fallback should extract actions mentioned in text
        assert "pick" in result.actions
        assert "pour" in result.actions

    @patch('core.agents.action_extractor.ollama.chat')
    def test_extract_actions_empty(self, mock_chat, action_extractor):
        # test action extraction with empty result
        # Mock LLM response with empty array
        mock_chat.return_value = {
            'message': {'content': '[]'}
        }

        input_data = ActionExtractionInput(
            user_prompt="Reset robot",
            intent_text=None,
            available_actions=action_extractor.available_actions
        )

        result = action_extractor.extract_actions(input_data)

        assert result.actions == []

    def test_format_actions_context(self, action_extractor):
        # test formatting of actions context for llm
        context = action_extractor._format_actions_context(
            action_extractor.available_actions
        )

        assert "pick" in context
        assert "place" in context
        assert "pour" in context
        assert "Keywords:" in context
        assert "Description:" in context

    def test_fallback_extraction(self, action_extractor):
        # test fallback extraction logic
        text = "You should pick up the object and then pour it"

        actions = action_extractor._fallback_extraction(text)

        assert "pick" in actions
        assert "pour" in actions

    @patch('core.agents.action_extractor.Config.ACTION_EXTRACTION_ENABLED', False)
    def test_disabled_extraction(self, action_extractor):
        # test when action extraction is disabled
        input_data = ActionExtractionInput(
            user_prompt="Pick up blood",
            intent_text=None,
            available_actions=action_extractor.available_actions
        )

        result = action_extractor.extract_actions(input_data)

        assert result.actions == []
        assert "disabled" in result.reasoning.lower()

    def test_load_all_actions(self, mock_kb, sample_actions):
        # test loading all actions from file
        mock_file_content = json.dumps(sample_actions)

        with patch('builtins.open', mock_open(read_data=mock_file_content)):
            agent = ActionExtractorAgent(mock_kb)

            # Should have loaded actions
            assert len(agent.available_actions) > 0

    @patch('core.agents.action_extractor.ollama.chat')
    def test_complex_extraction(self, mock_chat, action_extractor):
        # test extraction with complex multi-step prompt
        # Mock LLM response
        mock_chat.return_value = {
            'message': {'content': '["pick", "place", "pick", "pour"]'}
        }

        input_data = ActionExtractionInput(
            user_prompt="Pick up blood, place it aside, then pick up DNA and pour it",
            intent_text=None,
            available_actions=action_extractor.available_actions
        )

        result = action_extractor.extract_actions(input_data)

        # Should preserve duplicates for multi-step tasks
        assert "pick" in result.actions
        assert "place" in result.actions
        assert "pour" in result.actions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])