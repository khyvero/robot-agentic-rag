"""
Unit tests for Conversation Agent
"""

import pytest
import json
from unittest.mock import Mock, patch
from core.agents.conversation_agent import ConversationAgent
from core.domain import RobotMission


class TestConversationAgent:
    """Test suite for Conversation Agent"""

    @pytest.fixture
    def conversation_agent(self):
        # create a conversationagent
        return ConversationAgent()

    @pytest.fixture
    def sample_candidates(self):
        # sample candidates for ambiguity resolution
        return [
            ('{"mission_name": "Mix blood", "steps": ["pick blood", "shake"]}', 1.2),
            ('{"mission_name": "Mix DNA", "steps": ["pick DNA", "shake"]}', 1.3),
            ('{"mission_name": "Mix phenol", "steps": ["pick phenol", "shake"]}', 1.4)
        ]

    @pytest.fixture
    def sample_plan_json(self):
        # sample plan json
        return json.dumps({
            "tasks": [
                {"type": "pick", "target_obj_name": "test_tube_blood"},
                {"type": "place", "destination_obj_name": "biohazard_bin", "destination_coords": {"x": 0, "y": 0}}
            ],
            "settings": {"simulation_speed": 1}
        })

    def test_generate_clarification(self, conversation_agent, sample_candidates):
        # test generating clarification question
        mission = conversation_agent.generate_clarification(
            user_prompt="Mix them",
            candidates=sample_candidates
        )

        assert mission.name == "Clarification Question"
        assert len(mission.steps) == 1
        assert mission.steps[0]["type"] == "ask_user"
        assert "Option 1" in mission.steps[0]["question"]
        assert "Option 2" in mission.steps[0]["question"]
        assert "Option 3" in mission.steps[0]["question"]

    def test_interpret_selection_new_plan(self, conversation_agent, sample_candidates):
        # test interpreting user selection for new plan
        result = conversation_agent.interpret_selection(
            user_reply="None of these, I want something different",
            candidates=sample_candidates
        )

        assert result.next_action == "GENERATE_PLAN"
        assert result.state_transition == "RESET_TO_IDLE"

    def test_interpret_selection_continue(self, conversation_agent, sample_candidates):
        # test interpreting user selection with more info
        result = conversation_agent.interpret_selection(
            user_reply="Use the red tube instead",
            candidates=sample_candidates
        )

        # Could be CONTINUE or EXECUTE depending on LLM interpretation
        assert result.state_transition == "RESET_TO_IDLE"

    def test_interpret_selection_execute(self, conversation_agent, sample_candidates):
        # test interpreting user selection of an option
        result = conversation_agent.interpret_selection(
            user_reply="Option 1",
            candidates=sample_candidates
        )

        assert result.state_transition == "RESET_TO_IDLE"
        # Should have identified selection
        assert result.next_action is not None

    def test_review_plan_initial(self, conversation_agent, sample_plan_json):
        # test generating initial plan review
        mission = conversation_agent.review_plan(
            plan_json=sample_plan_json,
            updated=False
        )

        assert mission.name == "Plan Review"
        assert len(mission.steps) == 1
        assert mission.steps[0]["type"] == "ask_user"

        question = mission.steps[0]["question"]
        assert "Pick up test_tube_blood" in question
        assert "Place it on biohazard_bin" in question
        assert "yes" in question.lower()

    def test_review_plan_updated(self, conversation_agent, sample_plan_json):
        # test generating updated plan review
        mission = conversation_agent.review_plan(
            plan_json=sample_plan_json,
            updated=True
        )

        assert mission.name == "Plan Review"
        question = mission.steps[0]["question"]
        assert "Updated Plan:" in question

    def test_review_plan_complex_tasks(self, conversation_agent):
        # test review with various task types
        complex_plan = json.dumps({
            "tasks": [
                {"type": "pick", "target_obj_name": "test_tube_blood"},
                {"type": "pour", "target_obj_name": "beaker_water"},
                {"type": "shake", "target_obj_name": "test_tube_blood"},
                {"type": "swirl", "target_obj_name": "test_tube_blood"},
                {"type": "wait", "seconds": 5},
                {"type": "move_home"},
                {"type": "ensure_gripper_empty"}
            ],
            "settings": {"simulation_speed": 1}
        })

        mission = conversation_agent.review_plan(
            plan_json=complex_plan,
            updated=False
        )

        question = mission.steps[0]["question"]
        assert "Pour into beaker_water" in question
        assert "Shake test_tube_blood" in question
        assert "Wait 5 seconds" in question
        assert "Move to home position" in question
        assert "Ensure gripper is empty" in question

    def test_review_plan_invalid_json(self, conversation_agent):
        # test review with invalid json
        mission = conversation_agent.review_plan(
            plan_json="{invalid json",
            updated=False
        )

        assert mission.name == "Plan Review"
        question = mission.steps[0]["question"]
        assert "failed" in question.lower()

    def test_modify_plan(self, conversation_agent, sample_plan_json):
        # test modifying a plan based on feedback
        modified = conversation_agent.modify_plan(
            current_plan=sample_plan_json,
            user_feedback="Change the destination to beaker_water"
        )

        # Should return valid JSON
        data = json.loads(modified)
        assert "tasks" in data
        assert "settings" in data

    def test_handle_plan_confirmation_yes(self, conversation_agent, sample_plan_json):
        # test handling plan confirmation with 'yes'
        result = conversation_agent.handle_plan_confirmation(
            user_input="yes",
            pending_plan_json=sample_plan_json
        )

        assert result.state_transition == "RESET_TO_IDLE"
        assert result.next_action == "EXECUTE"
        assert result.mission.name == "Dual RAG (Generated)"
        assert len(result.mission.steps) > 0

    def test_handle_plan_confirmation_modify(self, conversation_agent, sample_plan_json):
        # test handling plan modification request
        result = conversation_agent.handle_plan_confirmation(
            user_input="Change to DNA instead",
            pending_plan_json=sample_plan_json
        )

        assert result.state_transition == "STAY"
        assert result.next_action is None
        assert result.mission.name == "Plan Review"

    def test_handle_plan_confirmation_variations(self, conversation_agent, sample_plan_json):
        # test various confirmation keywords
        confirmation_words = ["ok", "confirm", "execute", "go", "y"]

        for word in confirmation_words:
            result = conversation_agent.handle_plan_confirmation(
                user_input=word,
                pending_plan_json=sample_plan_json
            )

            assert result.state_transition == "RESET_TO_IDLE"
            assert result.next_action == "EXECUTE"

    def test_handle_plan_confirmation_invalid_json(self, conversation_agent):
        # test handling confirmation with invalid pending plan
        result = conversation_agent.handle_plan_confirmation(
            user_input="yes",
            pending_plan_json="{invalid"
        )

        # Should return error mission
        assert result.mission.name == "Error"
        assert result.state_transition == "RESET_TO_IDLE"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])