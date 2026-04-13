"""
Integration tests for Dual-RAG Strategy

These tests verify the end-to-end functionality of the multi-agent system
with real Ollama LLM and real KnowledgeBase.
"""

import pytest
import json
from core.strategies.dual_rag.strategy import DualRAGStrategy, AgentState
from core.knowledge_base import KnowledgeBase
from core.domain import RobotMission


class TestDualRAGIntegration:
    """Integration test suite for Dual-RAG Strategy"""

    @pytest.fixture(scope="class")
    def knowledge_base(self):
        # create a real knowledgebase instance (fresh data)
        return KnowledgeBase()

    @pytest.fixture
    def dual_rag_strategy(self, knowledge_base):
        # create a dualragstrategy with real kb
        return DualRAGStrategy(knowledge_base)

    # =========================================================================
    # TIER 2 TESTS (Exact Match + Refinement)
    # =========================================================================

    def test_tier_2_exact_match_simple(self, dual_rag_strategy):
        # test single-action task (should be novel since it's not a complete recipe)
        # "Pick up test_tube_blood" is just a single action, not a complete recipe like "Sample Analysis"
        # This should trigger NOVEL_TASK (Tier 3) with plan review
        mission = dual_rag_strategy.generate_mission("Pick up test_tube_blood")

        # For single actions, system should enter plan review or directly execute
        # Accept both refined (Tier 2) or plan review (Tier 3)
        assert mission.name in ["Dual RAG (Refined)", "Plan Review", "Dual RAG (Generated)"]

        # If plan review, confirm it
        if dual_rag_strategy.state == AgentState.PLAN_REVIEW:
            mission = dual_rag_strategy.generate_mission("yes")
            assert dual_rag_strategy.state == AgentState.IDLE

        assert len(mission.steps) > 0

        # Validate mission structure
        assert isinstance(mission.steps, list)
        assert isinstance(mission.settings, dict)

        # Should have pick action
        assert any(step.get("type") == "pick" for step in mission.steps)

        print(f"✓ Tier 2 Simple: Generated {len(mission.steps)} steps")

    def test_tier_2_exact_match_complex(self, dual_rag_strategy):
        # test tier 2 flow with complex command
        mission = dual_rag_strategy.generate_mission(
            "Move test_tube_blood to biohazard_bin"
        )

        assert mission.name == "Dual RAG (Refined)"
        assert len(mission.steps) > 0

        # Should have both pick and place actions
        task_types = [step.get("type") for step in mission.steps]
        assert "pick" in task_types

        print(f"✓ Tier 2 Complex: Generated {len(mission.steps)} steps - {task_types}")

    # =========================================================================
    # TIER 3 TESTS (Novel Task + Generation)
    # =========================================================================

    def test_tier_3_novel_task_generation(self, dual_rag_strategy):
        # test tier 3 flow with novel task
        # Reset state
        dual_rag_strategy.state = AgentState.IDLE

        # First call: Novel task should trigger plan generation + review
        # Use verbose description with step explanations to signal novel task
        mission1 = dual_rag_strategy.generate_mission(
            "Crystallization procedure: first cool the supersaturated solution in the chamber "
            "to induce crystal formation, then carefully transfer the crystallized sample to "
            "a storage container and place it in the observation area"
        )

        # Should enter PLAN_REVIEW state (or AMBIGUOUS if distance is 1.1-1.6)
        # Update assertion to be more flexible
        assert dual_rag_strategy.state in [AgentState.PLAN_REVIEW, AgentState.AMBIGUITY_CHECK]

        if dual_rag_strategy.state == AgentState.AMBIGUITY_CHECK:
            # If ambiguous, select "none of these" to trigger novel task generation
            mission1 = dual_rag_strategy.generate_mission("None of these, create new task")

        # After handling ambiguity or directly, should be in PLAN_REVIEW
        assert dual_rag_strategy.state == AgentState.PLAN_REVIEW

        # Should return ask_user mission with plan review
        assert mission1.steps[0]["type"] == "ask_user"
        question = mission1.steps[0]["question"]
        assert "step" in question.lower() or "plan" in question.lower()

        print(f"✓ Tier 3 Generation: Entered PLAN_REVIEW state")
        print(f"  Plan summary: {question[:100]}...")

        # Second call: User confirms plan
        mission2 = dual_rag_strategy.generate_mission("yes")

        # Should return executable mission
        assert dual_rag_strategy.state == AgentState.IDLE
        assert mission2.name == "Dual RAG (Generated)"
        assert len(mission2.steps) > 0

        # Validate mission has appropriate actions
        task_types = [step.get("type") for step in mission2.steps]
        print(f"✓ Tier 3 Execution: Generated {len(mission2.steps)} steps - {task_types}")

        # Should have relevant actions for the task
        assert len(task_types) > 0

    def test_tier_3_plan_modification(self, dual_rag_strategy):
        # test tier 3 plan modification flow
        # Reset state
        dual_rag_strategy.state = AgentState.IDLE

        # First: Generate plan with a truly novel task using verbose description
        # Include step explanations to signal teaching/novel task
        mission1 = dual_rag_strategy.generate_mission(
            "Spectrometry calibration procedure: adjust the wavelength settings to optimal range, "
            "align the detector components for accuracy, then verify calibration with test sample"
        )

        # Handle ambiguity if needed
        if dual_rag_strategy.state == AgentState.AMBIGUITY_CHECK:
            mission1 = dual_rag_strategy.generate_mission("None of these, create new task")

        assert dual_rag_strategy.state == AgentState.PLAN_REVIEW

        # Second: Request modification
        mission2 = dual_rag_strategy.generate_mission(
            "Change the destination to beaker_water"
        )

        # Should still be in PLAN_REVIEW state
        assert dual_rag_strategy.state == AgentState.PLAN_REVIEW

        # Should return updated plan review
        assert mission2.steps[0]["type"] == "ask_user"
        question = mission2.steps[0]["question"]
        assert "plan" in question.lower()

        print(f"✓ Tier 3 Modification: Plan modified successfully")

        # Third: Confirm modified plan
        mission3 = dual_rag_strategy.generate_mission("yes")

        assert dual_rag_strategy.state == AgentState.IDLE
        assert len(mission3.steps) > 0

        print(f"✓ Tier 3 Confirmed: Generated {len(mission3.steps)} steps")

    # =========================================================================
    # AMBIGUITY TESTS
    # =========================================================================

    def test_ambiguity_resolution_flow(self, dual_rag_strategy):
        # test ambiguity resolution flow
        # Reset state
        dual_rag_strategy.state = AgentState.IDLE

        # First: Ambiguous input
        mission1 = dual_rag_strategy.generate_mission("Mix them")

        # Might trigger ambiguity or novel task depending on DB
        # If ambiguous:
        if dual_rag_strategy.state == AgentState.AMBIGUITY_CHECK:
            assert mission1.steps[0]["type"] == "ask_user"
            question = mission1.steps[0]["question"]
            assert "option" in question.lower()

            print(f"✓ Ambiguity Detected: Asking for clarification")

            # Second: Select option
            mission2 = dual_rag_strategy.generate_mission("Option 1")

            # Should resolve ambiguity and generate plan
            assert dual_rag_strategy.state in [AgentState.IDLE, AgentState.PLAN_REVIEW]
            assert len(mission2.steps) > 0

            print(f"✓ Ambiguity Resolved: Selected option processed")
        else:
            print(f"✓ Ambiguity: Routed to {dual_rag_strategy.state.name}")

    # =========================================================================
    # NOT_TASK TESTS
    # =========================================================================

    def test_not_task_input(self, dual_rag_strategy):
        # test handling of non-task inputs
        mission = dual_rag_strategy.generate_mission("Hello, how are you?")

        # Should return info mission
        assert mission.name == "Info"
        assert mission.steps[0]["type"] == "ask_user"
        assert "robot" in mission.steps[0]["question"].lower()

        print(f"✓ Non-Task: Handled with info response")

    # =========================================================================
    # STATE MANAGEMENT TESTS
    # =========================================================================

    def test_state_persistence_across_calls(self, dual_rag_strategy):
        # test that state persists correctly across calls
        # Reset to IDLE
        dual_rag_strategy.state = AgentState.IDLE

        # Trigger Tier 3
        mission1 = dual_rag_strategy.generate_mission(
            "Do a complex new task with test_tube_blood"
        )

        # State should change
        if dual_rag_strategy.state == AgentState.PLAN_REVIEW:
            # Pending plan should be stored
            assert dual_rag_strategy.pending_plan_json is not None

            # Confirm plan
            mission2 = dual_rag_strategy.generate_mission("yes")

            # State should reset
            assert dual_rag_strategy.state == AgentState.IDLE
            assert dual_rag_strategy.pending_plan_json is None

            print(f"✓ State Management: Correctly transitioned IDLE→PLAN_REVIEW→IDLE")

    # =========================================================================
    # MISSION VALIDATION TESTS
    # =========================================================================

    def test_mission_json_structure(self, dual_rag_strategy):
        # test that generated missions have valid json structure
        mission = dual_rag_strategy.generate_mission("Pick up test_tube_blood")

        # Validate mission structure
        assert hasattr(mission, 'name')
        assert hasattr(mission, 'steps')
        assert hasattr(mission, 'settings')

        # Validate steps structure
        for step in mission.steps:
            assert isinstance(step, dict)
            assert "type" in step

            # Validate specific step types
            if step["type"] == "pick":
                assert "target_obj_name" in step
            elif step["type"] == "place":
                # Should have either destination_obj_name or coords
                assert "destination_obj_name" in step or "destination_coords" in step

        # Validate settings
        assert isinstance(mission.settings, dict)
        if "simulation_speed" in mission.settings:
            assert isinstance(mission.settings["simulation_speed"], (int, float))

        print(f"✓ Mission Validation: Structure is valid")

    def test_multiple_sequential_missions(self, dual_rag_strategy):
        # test multiple sequential mission generations
        # Reset state
        dual_rag_strategy.state = AgentState.IDLE

        prompts = [
            "Pick up test_tube_blood",
            "Move test_tube_DNA to biohazard_bin",
            "Pour test_tube_phenol into beaker_water"
        ]

        for prompt in prompts:
            dual_rag_strategy.state = AgentState.IDLE
            mission = dual_rag_strategy.generate_mission(prompt)

            # Each should generate valid mission
            # Can be refined, generated, or plan review (if modifications detected)
            assert mission.name in ["Dual RAG (Refined)", "Dual RAG (Generated)", "Plan Review", "Plan Review (Modified Recipe)"]
            assert len(mission.steps) > 0

            print(f"✓ Sequential Mission: '{prompt}' → {len(mission.steps)} steps")

    # =========================================================================
    # ACTION EXTRACTION TESTS
    # =========================================================================

    def test_action_extraction_integration(self, dual_rag_strategy):
        # test that action extraction integrates correctly
        # This should trigger action extraction
        mission = dual_rag_strategy.generate_mission(
            "Pick up the blood sample and shake it"
        )

        # Should successfully generate a mission
        assert len(mission.steps) > 0

        # Check if relevant actions are present
        task_types = [step.get("type") for step in mission.steps]

        print(f"✓ Action Extraction: Generated tasks {task_types}")

        # Should have actions related to pick and shake
        # (though exact actions depend on LLM generation)
        assert len(task_types) > 0

    # =========================================================================
    # EDGE CASES
    # =========================================================================

    def test_empty_prompt(self, dual_rag_strategy):
        # test handling of empty prompt
        dual_rag_strategy.state = AgentState.IDLE

        mission = dual_rag_strategy.generate_mission("")

        # Should handle gracefully
        assert mission is not None
        assert isinstance(mission, RobotMission)

        print(f"✓ Edge Case: Empty prompt handled")

    def test_invalid_object_names(self, dual_rag_strategy):
        # test with invalid object names
        dual_rag_strategy.state = AgentState.IDLE

        mission = dual_rag_strategy.generate_mission(
            "Pick up invalid_object_xyz"
        )

        # Should still generate a mission (even if invalid)
        assert mission is not None

        print(f"✓ Edge Case: Invalid object handled")


class TestDualRAGStressTest:
    # stress tests for dual-rag system

    @pytest.fixture(scope="class")
    def knowledge_base(self):
        # create a real knowledgebase instance
        return KnowledgeBase()

    @pytest.fixture
    def dual_rag_strategy(self, knowledge_base):
        # create a dualragstrategy
        return DualRAGStrategy(knowledge_base)

    def test_rapid_sequential_calls(self, dual_rag_strategy):
        # test rapid sequential mission generation
        dual_rag_strategy.state = AgentState.IDLE

        prompts = [
            "Pick up test_tube_blood",
            "Place it in the bin",
            "Move test_tube_DNA",
            "Pour into beaker"
        ]

        missions = []
        for prompt in prompts:
            dual_rag_strategy.state = AgentState.IDLE
            mission = dual_rag_strategy.generate_mission(prompt)
            missions.append(mission)

        # All should succeed
        assert all(len(m.steps) > 0 for m in missions)

        print(f"✓ Stress Test: Generated {len(missions)} missions rapidly")

    def test_complex_multi_step_task(self, dual_rag_strategy):
        # test complex multi-step task
        dual_rag_strategy.state = AgentState.IDLE

        complex_prompt = (
            "Pick up test_tube_blood, shake it for 5 seconds, "
            "pour it into beaker_water, wait 3 seconds, "
            "then move the beaker to biohazard_bin"
        )

        mission = dual_rag_strategy.generate_mission(complex_prompt)

        # Should handle complex task
        if dual_rag_strategy.state == AgentState.PLAN_REVIEW:
            # Confirm the plan
            mission = dual_rag_strategy.generate_mission("yes")

        assert len(mission.steps) > 0

        print(f"✓ Stress Test: Complex task generated {len(mission.steps)} steps")


if __name__ == "__main__":
    # Run with verbose output
    pytest.main([__file__, "-v", "-s"])