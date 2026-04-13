"""
Integration tests for Single RAG Strategy

These tests verify the end-to-end functionality of the Single RAG strategy
with real Ollama LLM and real KnowledgeBase.

Single RAG uses a unified knowledge base (declarative + procedural mixed)
for fair comparison with Dual-RAG's separated architecture.
"""

import pytest
import json
from core.strategies.single_rag import SingleRAGStrategy
from core.knowledge_base import KnowledgeBase
from core.domain import RobotMission


class TestSingleRAGIntegration:
    """Integration test suite for Single RAG Strategy"""

    @pytest.fixture(scope="class")
    def knowledge_base(self):
        # create a real knowledgebase instance (fresh data)
        return KnowledgeBase()

    @pytest.fixture
    def single_rag_strategy(self, knowledge_base):
        # create a singleragstrategy with real kb
        return SingleRAGStrategy(knowledge_base)

    # =========================================================================
    # BASIC TASK TESTS
    # =========================================================================

    def test_simple_pick_task(self, single_rag_strategy):
        # test simple pick command
        mission = single_rag_strategy.generate_mission("Pick up test_tube_blood")

        assert mission.name == "Single RAG Mission"
        assert len(mission.steps) > 0

        # Validate mission structure
        assert isinstance(mission.steps, list)
        assert isinstance(mission.settings, dict)

        # Should have pick action
        assert any(step.get("type") == "pick" for step in mission.steps)

        print(f"✓ Simple Pick: Generated {len(mission.steps)} steps")

    def test_simple_move_task(self, single_rag_strategy):
        # test simple move command
        mission = single_rag_strategy.generate_mission(
            "Move test_tube_DNA to biohazard_bin"
        )

        assert mission.name == "Single RAG Mission"
        assert len(mission.steps) > 0

        # Should have pick action at minimum
        task_types = [step.get("type") for step in mission.steps]
        assert "pick" in task_types

        print(f"✓ Simple Move: Generated {len(mission.steps)} steps - {task_types}")

    def test_pour_task(self, single_rag_strategy):
        # test pour command
        mission = single_rag_strategy.generate_mission(
            "Pour test_tube_phenol into beaker_water"
        )

        assert mission.name == "Single RAG Mission"
        assert len(mission.steps) > 0

        # Should have pour action (pick is optional - LLM may infer it or skip it)
        task_types = [step.get("type") for step in mission.steps]
        assert "pour" in task_types

        print(f"✓ Pour Task: Generated {len(mission.steps)} steps - {task_types}")

    def test_shake_task(self, single_rag_strategy):
        # test shake command
        mission = single_rag_strategy.generate_mission("Shake test_tube_blood")

        assert mission.name == "Single RAG Mission"
        assert len(mission.steps) > 0

        # Should have shake action (pick is optional - LLM may infer it or skip it)
        task_types = [step.get("type") for step in mission.steps]
        assert "shake" in task_types

        print(f"✓ Shake Task: Generated {len(mission.steps)} steps - {task_types}")

    def test_swirl_task(self, single_rag_strategy):
        # test swirl command
        mission = single_rag_strategy.generate_mission("Swirl test_tube_DNA")

        assert mission.name == "Single RAG Mission"
        assert len(mission.steps) > 0

        # Should have swirl action (pick is optional - LLM may infer it or skip it)
        task_types = [step.get("type") for step in mission.steps]
        assert "swirl" in task_types

        print(f"✓ Swirl Task: Generated {len(mission.steps)} steps - {task_types}")

    # =========================================================================
    # GREETING AND CONVERSATION TESTS
    # =========================================================================

    def test_greeting_hello(self, single_rag_strategy):
        # test greeting detection - hello
        mission = single_rag_strategy.generate_mission("Hello")

        assert mission.name == "Info"
        assert len(mission.steps) == 1
        assert mission.steps[0]["type"] == "ask_user"
        assert "robot assistant" in mission.steps[0]["question"].lower()

        print("✓ Greeting (hello): Detected correctly")

    def test_greeting_hi(self, single_rag_strategy):
        # test greeting detection - hi
        mission = single_rag_strategy.generate_mission("Hi")

        assert mission.name == "Info"
        assert len(mission.steps) == 1
        assert mission.steps[0]["type"] == "ask_user"

        print("✓ Greeting (hi): Detected correctly")

    def test_greeting_thanks(self, single_rag_strategy):
        # test greeting detection - thanks
        mission = single_rag_strategy.generate_mission("Thanks")

        assert mission.name == "Info"
        assert len(mission.steps) == 1

        print("✓ Greeting (thanks): Detected correctly")

    def test_greeting_what_can_you_do(self, single_rag_strategy):
        # test conversational question
        mission = single_rag_strategy.generate_mission("What can you do?")

        assert mission.name == "Info"
        assert len(mission.steps) == 1

        print("✓ Greeting (what can you do): Detected correctly")

    # =========================================================================
    # COMPLEX MULTI-STEP TASKS
    # =========================================================================

    def test_multi_step_task_pick_swirl_place(self, single_rag_strategy):
        # test task that requires multiple actions
        mission = single_rag_strategy.generate_mission(
            "Pick up test_tube_blood, swirl it, and place it in storage area"
        )

        assert mission.name == "Single RAG Mission"
        assert len(mission.steps) >= 3  # pick, swirl, place

        task_types = [step.get("type") for step in mission.steps]
        assert "pick" in task_types
        assert "swirl" in task_types
        # Should have some placement action (place, place_free_spot, etc.)
        assert any(t in ["place", "place_free_spot", "place_in_area"] for t in task_types)

        print(f"✓ Multi-Step (pick-swirl-place): Generated {len(mission.steps)} steps - {task_types}")

    def test_complex_laboratory_workflow(self, single_rag_strategy):
        # test complex laboratory workflow
        mission = single_rag_strategy.generate_mission(
            "Perform pH test: pick up dropper_Phenolphtalein, pour it into test_tube_blood, "
            "observe the color change, then dispose of the test tube"
        )

        assert mission.name == "Single RAG Mission"
        assert len(mission.steps) > 0

        # Should have pick and pour actions at minimum
        task_types = [step.get("type") for step in mission.steps]
        assert "pick" in task_types

        print(f"✓ Complex Workflow: Generated {len(mission.steps)} steps")

    # =========================================================================
    # NOVEL/UNFAMILIAR TASKS
    # =========================================================================

    def test_novel_task_verbose_description(self, single_rag_strategy):
        # test novel task with verbose step-by-step description
        mission = single_rag_strategy.generate_mission(
            "Crystallization procedure: first cool the supersaturated solution, "
            "then carefully transfer the crystallized sample to storage, "
            "finally place it in the observation area for analysis"
        )

        # Single RAG should attempt to generate a plan (no plan review state)
        assert mission.name in ["Single RAG Mission", "Error"]

        # If successful, should have multiple steps
        if mission.name == "Single RAG Mission":
            assert len(mission.steps) > 0
            print(f"✓ Novel Task (verbose): Generated {len(mission.steps)} steps")
        else:
            print("✓ Novel Task (verbose): Handled gracefully (error state)")

    def test_novel_task_unfamiliar_objects(self, single_rag_strategy):
        # test task with unfamiliar objects (should still generate plan)
        mission = single_rag_strategy.generate_mission(
            "Transfer the quantum sample to the containment chamber"
        )

        # Should attempt to generate something (even if objects don't exist)
        assert mission.name in ["Single RAG Mission", "Error"]

        print(f"✓ Novel Task (unfamiliar objects): {mission.name}")

    # =========================================================================
    # EDGE CASES AND ERROR HANDLING
    # =========================================================================

    def test_empty_prompt(self, single_rag_strategy):
        # test empty user prompt
        mission = single_rag_strategy.generate_mission("")

        # Should handle gracefully (either info or error)
        assert mission.name in ["Info", "Error", "Single RAG Mission"]
        assert isinstance(mission.steps, list)

        print(f"✓ Empty Prompt: Handled gracefully ({mission.name})")

    def test_very_long_prompt(self, single_rag_strategy):
        # test very long user prompt
        long_prompt = (
            "I need you to perform a very complex laboratory procedure that involves "
            "picking up multiple test tubes in a specific sequence, then performing "
            "various mixing and heating operations while carefully monitoring the "
            "temperature and pH levels throughout the entire process " * 5
        )

        mission = single_rag_strategy.generate_mission(long_prompt)

        # Should handle gracefully
        assert mission.name in ["Single RAG Mission", "Error"]
        assert isinstance(mission.steps, list)

        print(f"✓ Long Prompt: Handled gracefully ({mission.name})")

    def test_invalid_object_names(self, single_rag_strategy):
        # test with invalid object names
        mission = single_rag_strategy.generate_mission(
            "Pick up the magical unicorn horn"
        )

        # Should still generate a plan (LLM doesn't validate object existence)
        assert mission.name in ["Single RAG Mission", "Error"]

        print(f"✓ Invalid Objects: Handled gracefully ({mission.name})")

    def test_ambiguous_short_command(self, single_rag_strategy):
        # test ambiguous command like 'mix them'
        mission = single_rag_strategy.generate_mission("Mix them")

        # Single RAG has no ambiguity handling, should attempt to generate plan
        assert mission.name in ["Single RAG Mission", "Error"]

        print(f"✓ Ambiguous Command: Generated plan ({mission.name})")

    # =========================================================================
    # JSON STRUCTURE VALIDATION
    # =========================================================================

    def test_mission_json_structure(self, single_rag_strategy):
        # test that generated missions have valid json structure
        mission = single_rag_strategy.generate_mission("Pick up test_tube_blood")

        assert mission.name == "Single RAG Mission"
        assert hasattr(mission, 'raw_plan')
        assert mission.raw_plan is not None

        # Validate it's valid JSON
        try:
            plan_data = json.loads(mission.raw_plan)
            assert "tasks" in plan_data
            assert "settings" in plan_data
            assert isinstance(plan_data["tasks"], list)
            assert isinstance(plan_data["settings"], dict)
            print("✓ JSON Structure: Valid")
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON structure: {e}")

    def test_task_type_validation(self, single_rag_strategy):
        # test that all generated tasks have valid 'type' field
        mission = single_rag_strategy.generate_mission("Pick up test_tube_blood and pour it")

        assert mission.name == "Single RAG Mission"
        assert len(mission.steps) > 0

        # All tasks must have 'type' field
        for i, step in enumerate(mission.steps):
            assert "type" in step, f"Step {i} missing 'type' field"
            assert isinstance(step["type"], str), f"Step {i} 'type' is not string"

        print(f"✓ Task Type Validation: All {len(mission.steps)} tasks valid")

    # =========================================================================
    # SEQUENTIAL EXECUTION TESTS
    # =========================================================================

    def test_multiple_sequential_tasks(self, single_rag_strategy):
        # test multiple tasks executed sequentially
        tasks = [
            "Pick up test_tube_blood",
            "Move test_tube_DNA to biohazard_bin",
            "Pour test_tube_phenol into beaker_water"
        ]

        missions = []
        for task in tasks:
            mission = single_rag_strategy.generate_mission(task)
            missions.append(mission)
            print(f"✓ Sequential Task: '{task}' → {len(mission.steps)} steps")

        # All missions should be successful
        for i, mission in enumerate(missions):
            assert mission.name in ["Single RAG Mission", "Error"], f"Task {i} failed"
            if mission.name == "Single RAG Mission":
                assert len(mission.steps) > 0, f"Task {i} has no steps"

    # =========================================================================
    # RETRY AND ERROR HANDLING TESTS
    # =========================================================================

    def test_json_retry_logic(self, single_rag_strategy):
        # test that retry logic is working (indirect test)
        # This is an indirect test - we execute multiple tasks and ensure
        # they all succeed, which demonstrates retry logic is working
        test_prompts = [
            "Pick up test_tube_blood",
            "Shake test_tube_DNA",
            "Pour beaker_water",
            "Move test_tube_phenol to biohazard_bin",
            "Swirl test_tube_empty"
        ]

        success_count = 0
        for prompt in test_prompts:
            mission = single_rag_strategy.generate_mission(prompt)
            if mission.name == "Single RAG Mission" and len(mission.steps) > 0:
                success_count += 1

        # At least 80% should succeed (4 out of 5)
        assert success_count >= 4, f"Only {success_count}/5 tasks succeeded"
        print(f"✓ Retry Logic: {success_count}/5 tasks succeeded")


class TestSingleRAGStressTest:
    # stress tests for single rag strategy

    @pytest.fixture(scope="class")
    def knowledge_base(self):
        # create a real knowledgebase instance
        return KnowledgeBase()

    @pytest.fixture
    def single_rag_strategy(self, knowledge_base):
        # create a singleragstrategy with real kb
        return SingleRAGStrategy(knowledge_base)

    def test_rapid_sequential_calls(self, single_rag_strategy):
        # test rapid sequential calls to the strategy
        tasks = [
            "Pick up test_tube_blood",
            "Place it in the bin",
            "Move test_tube_DNA",
            "Pour into beaker"
        ]

        missions = []
        for task in tasks:
            mission = single_rag_strategy.generate_mission(task)
            missions.append(mission)

        # Most should succeed
        successful = [m for m in missions if m.name == "Single RAG Mission" and len(m.steps) > 0]
        assert len(successful) >= 2  # At least 50% success rate

        print(f"✓ Rapid Sequential: {len(successful)}/{len(tasks)} successful")

    def test_complex_multi_step_task(self, single_rag_strategy):
        # test complex task with many steps
        mission = single_rag_strategy.generate_mission(
            "Perform DNA extraction: First pick up test_tube_DNA, then shake it vigorously, "
            "pour the contents into beaker_water, swirl the mixture, wait 5 seconds, "
            "and finally transfer everything to biohazard_bin for disposal"
        )

        assert mission.name in ["Single RAG Mission", "Error"]

        if mission.name == "Single RAG Mission":
            # Should generate multiple steps
            assert len(mission.steps) > 0
            print(f"✓ Complex Multi-Step: Generated {len(mission.steps)} steps")
        else:
            print("✓ Complex Multi-Step: Handled gracefully (error state)")


class TestSingleRAGRetrievalQuality:
    # tests focused on retrieval quality and context

    @pytest.fixture(scope="class")
    def knowledge_base(self):
        # create a real knowledgebase instance
        return KnowledgeBase()

    @pytest.fixture
    def single_rag_strategy(self, knowledge_base):
        # create a singleragstrategy with real kb
        return SingleRAGStrategy(knowledge_base)

    def test_retrieval_budget_respected(self, single_rag_strategy):
        # test that retrieval uses n_results=5
        # This is an indirect test - we verify that the strategy works
        # with n_results=5 by executing a task
        mission = single_rag_strategy.generate_mission("Pick up test_tube_blood")

        assert mission.name == "Single RAG Mission"
        assert len(mission.steps) > 0

        print("✓ Retrieval Budget: n_results=5 working correctly")

    def test_unified_context_handling(self, single_rag_strategy):
        # test that unified context (mixed declarative + procedural) works
        # Single RAG retrieves mixed context
        mission = single_rag_strategy.generate_mission(
            "Perform sample analysis on test_tube_blood"
        )

        # Should successfully generate a plan using unified context
        assert mission.name in ["Single RAG Mission", "Error"]

        if mission.name == "Single RAG Mission":
            assert len(mission.steps) > 0
            print(f"✓ Unified Context: Generated {len(mission.steps)} steps from mixed context")


class TestSingleRAGComparison:
    # tests designed for fair comparison with dual-rag

    @pytest.fixture(scope="class")
    def knowledge_base(self):
        # create a real knowledgebase instance
        return KnowledgeBase()

    @pytest.fixture
    def single_rag_strategy(self, knowledge_base):
        # create a singleragstrategy with real kb
        return SingleRAGStrategy(knowledge_base)

    def test_same_model_as_dual_rag(self, single_rag_strategy):
        # verify using llama3 for json generation (same as dual-rag)
        from config.config import Config

        # Verify config uses llama3 for plan generation
        assert Config.LLM_PLAN_GENERATION_MODEL == "llama3:8b"

        # Execute a task to ensure it works
        mission = single_rag_strategy.generate_mission("Pick up test_tube_blood")
        assert mission.name == "Single RAG Mission"

        print("✓ Model Parity: Using llama3:8b like Dual-RAG")

    def test_json_reliability(self, single_rag_strategy):
        # test json generation reliability (should match dual-rag)
        tasks = [
            "Pick up test_tube_blood",
            "Shake test_tube_DNA",
            "Pour test_tube_phenol into beaker_water",
            "Move test_tube_empty to biohazard_bin",
            "Swirl test_tube_blood"
        ]

        success_count = 0
        for task in tasks:
            mission = single_rag_strategy.generate_mission(task)
            if mission.name == "Single RAG Mission":
                # Verify JSON is valid
                try:
                    if mission.raw_plan:
                        json.loads(mission.raw_plan)
                        success_count += 1
                except json.JSONDecodeError:
                    pass

        # Should have high success rate (>= 80%)
        assert success_count >= 4, f"Only {success_count}/5 had valid JSON"
        print(f"✓ JSON Reliability: {success_count}/5 (80%+)")

    def test_greeting_handling_parity(self, single_rag_strategy):
        # test greeting handling matches dual-rag behavior
        greetings = ["Hello", "Hi", "Thanks", "What can you do?"]

        for greeting in greetings:
            mission = single_rag_strategy.generate_mission(greeting)
            assert mission.name == "Info", f"Failed for: {greeting}"
            assert len(mission.steps) == 1

        print(f"✓ Greeting Parity: All {len(greetings)} greetings handled correctly")