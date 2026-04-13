"""
Unit tests for Plan Generator Agent
"""

import pytest
import json
from unittest.mock import Mock, patch
from core.agents.plan_generator import PlanGeneratorAgent
from core.agents.types import PlanGenerationInput, PlanGenerationResult


class TestPlanGeneratorAgent:
    """Test suite for Plan Generator Agent"""

    @pytest.fixture
    def plan_generator(self):
        # create a plangeneratoragent
        return PlanGeneratorAgent()

    def test_tier_2_generation(self, plan_generator):
        # test tier 2 plan generation with real ollama
        input_data = PlanGenerationInput(
            user_prompt="Pick up test_tube_blood",
            intent_context="Recipe: 1. Pick up blood sample, 2. Place in holder",
            procedural_context="pick(target_obj_name: str) - Picks up object\nplace(destination) - Places object",
            mode="TIER_2_REFINEMENT"
        )

        result = plan_generator.generate_plan(input_data)

        # Should succeed with valid JSON
        assert result.success is True
        assert result.error is None

        # Validate JSON structure
        data = json.loads(result.plan_json)
        assert "tasks" in data
        assert "settings" in data
        assert isinstance(data["tasks"], list)

    def test_tier_3_generation(self, plan_generator):
        # test tier 3 plan generation with real ollama
        input_data = PlanGenerationInput(
            user_prompt="Mix blood and DNA samples",
            intent_context=None,  # No intent for Tier 3
            procedural_context="pick(obj) - Picks object\npour(obj) - Pours liquid\nshake(obj) - Shakes object",
            mode="TIER_3_GENERATION"
        )

        result = plan_generator.generate_plan(input_data)

        # Should succeed
        assert result.success is True

        # Validate JSON
        data = json.loads(result.plan_json)
        assert "tasks" in data
        assert "settings" in data

    def test_clean_json_markdown(self, plan_generator):
        # test json cleaning with markdown code blocks
        raw_text = """```json
{
  "tasks": [{"type": "pick", "target_obj_name": "test"}],
  "settings": {"simulation_speed": 1}
}
```"""

        clean = plan_generator._clean_json(raw_text)
        data = json.loads(clean)

        assert "tasks" in data
        assert "settings" in data

    def test_clean_json_with_comments(self, plan_generator):
        # test json cleaning with javascript comments
        raw_text = """{
  "tasks": [
    {"type": "pick"} // This picks an object
  ],
  "settings": {"simulation_speed": 1}
}"""

        clean = plan_generator._clean_json(raw_text)
        data = json.loads(clean)

        assert "tasks" in data
        assert "settings" in data

    def test_clean_json_with_multiline_comments(self, plan_generator):
        # test json cleaning with multiline comments
        raw_text = """{
  /* This is a comment */
  "tasks": [{"type": "pick"}],
  "settings": {"simulation_speed": 1}
}"""

        clean = plan_generator._clean_json(raw_text)
        data = json.loads(clean)

        assert "tasks" in data

    def test_invalid_mode(self, plan_generator):
        # test with invalid mode
        input_data = PlanGenerationInput(
            user_prompt="Test",
            intent_context=None,
            procedural_context="test",
            mode="INVALID_MODE"
        )

        result = plan_generator.generate_plan(input_data)

        assert result.success is False
        assert "Invalid mode" in result.error

    @patch('core.agents.plan_generator.ollama.chat')
    def test_json_validation_missing_keys(self, mock_chat, plan_generator):
        # test validation catches missing required keys
        # Mock LLM to return JSON without required keys
        mock_chat.return_value = {
            'message': {'content': '{"tasks": []}'}  # Missing 'settings'
        }

        input_data = PlanGenerationInput(
            user_prompt="Test",
            intent_context="context",
            procedural_context="api",
            mode="TIER_2_REFINEMENT"
        )

        result = plan_generator.generate_plan(input_data)

        assert result.success is False
        assert "Missing required keys" in result.error

    @patch('core.agents.plan_generator.ollama.chat')
    def test_malformed_json(self, mock_chat, plan_generator):
        # test handling of malformed json
        # Mock LLM to return invalid JSON
        mock_chat.return_value = {
            'message': {'content': '{invalid json'}
        }

        input_data = PlanGenerationInput(
            user_prompt="Test",
            intent_context="context",
            procedural_context="api",
            mode="TIER_2_REFINEMENT"
        )

        result = plan_generator.generate_plan(input_data)

        assert result.success is False
        assert "JSON decode error" in result.error

    def test_clean_json_extraction(self, plan_generator):
        # test json extraction from mixed text
        raw_text = """Here is the plan:
{
  "tasks": [{"type": "pick"}],
  "settings": {"simulation_speed": 1}
}
Some trailing text"""

        clean = plan_generator._clean_json(raw_text)
        data = json.loads(clean)

        assert "tasks" in data
        assert "settings" in data

    def test_pydantic_validation(self):
        # test pydantic validation for input
        # Valid input
        input_data = PlanGenerationInput(
            user_prompt="Test",
            intent_context="context",
            procedural_context="api",
            mode="TIER_2_REFINEMENT"
        )
        assert input_data.mode == "TIER_2_REFINEMENT"

        # Invalid mode (not in Literal)
        with pytest.raises(Exception):  # Pydantic ValidationError
            PlanGenerationInput(
                user_prompt="Test",
                intent_context="context",
                procedural_context="api",
                mode="INVALID"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])