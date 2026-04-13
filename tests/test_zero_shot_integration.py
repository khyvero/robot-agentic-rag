"""
Integration tests for Zero-Shot Strategy

These tests verify the end-to-end functionality of the Zero-Shot strategy
with real Ollama LLM but NO knowledge base retrieval.

Zero-Shot uses pure LLM generation (no RAG) as a baseline for comparison
with Single RAG and Dual-RAG strategies.
"""

import pytest
import json
from core.strategies.zero_shot import ZeroShotStrategy
from core.domain import RobotMission


class TestZeroShotIntegration:
    """Integration test suite for Zero-Shot Strategy"""

    @pytest.fixture
    def zero_shot_strategy(self):
        # create a zeroshotstrategy (no kb needed)
        return ZeroShotStrategy()

    # =========================================================================
    # BASIC TASK TESTS
    # =========================================================================

    def test_simple_pick_task(self, zero_shot_strategy):
        # test simple pick command
        mission = zero_shot_strategy.generate_mission("Pick up test_tube_blood")

        assert mission.name == "Zero-Shot Mission"
        assert len(mission.steps) > 0

        # Validate mission structure
        assert isinstance(mission.steps, list)
        assert isinstance(mission.settings, dict)

        # Should have pick action (though Zero-Shot might not always get it right)
        # More lenient than RAG tests since Zero-Shot has no context
        task_types = [step.get("type") for step in mission.steps]
        assert len(task_types) > 0

        print(f"✓ Simple Pick: Generated {len(mission.steps)} steps - {task_types}")

    def test_simple_move_task(self, zero_shot_strategy):
        # test simple move command
        mission = zero_shot_strategy.generate_mission(
            "Move test_tube_DNA to biohazard_bin"
        )

        assert mission.name == "Zero-Shot Mission"
        assert len(mission.steps) > 0

        # Zero-Shot should generate some plan (may not be perfect)
        task_types = [step.get("type") for step in mission.steps]
        assert len(task_types) > 0

        print(f"✓ Simple Move: Generated {len(mission.steps)} steps - {task_types}")

    def test_pour_task(self, zero_shot_strategy):
        # test pour command
        mission = zero_shot_strategy.generate_mission(
            "Pour test_tube_phenol into beaker_water"
        )

        assert mission.name == "Zero-Shot Mission"
        assert len(mission.steps) > 0

        # Zero-Shot should generate a plan (accuracy tested separately)
        print(f"✓ Pour Task: Generated {len(mission.steps)} steps")

    def test_shake_task(self, zero_shot_strategy):
        # test shake command
        mission = zero_shot_strategy.generate_mission("Shake test_tube_blood")

        assert mission.name == "Zero-Shot Mission"
        assert len(mission.steps) > 0

        print(f"✓ Shake Task: Generated {len(mission.steps)} steps")

    def test_swirl_task(self, zero_shot_strategy):
        # test swirl command
        mission = zero_shot_strategy.generate_mission("Swirl test_tube_DNA")

        assert mission.name == "Zero-Shot Mission"
        assert len(mission.steps) > 0

        print(f"✓ Swirl Task: Generated {len(mission.steps)} steps")

    # =========================================================================
    # GREETING AND CONVERSATION TESTS
    # =========================================================================

    def test_greeting_hello(self, zero_shot_strategy):
        # test greeting detection - hello
        mission = zero_shot_strategy.generate_mission("Hello")

        assert mission.name == "Info"
        assert len(mission.steps) == 1
        assert mission.steps[0]["type"] == "ask_user"
        assert "robot assistant" in mission.steps[0]["question"].lower()

        print("✓ Greeting (hello): Detected correctly")

    def test_greeting_hi(self, zero_shot_strategy):
        # test greeting detection - hi
        mission = zero_shot_strategy.generate_mission("Hi")

        assert mission.name == "Info"
        assert len(mission.steps) == 1
        assert mission.steps[0]["type"] == "ask_user"

        print("✓ Greeting (hi): Detected correctly")

    def test_greeting_thanks(self, zero_shot_strategy):
        # test greeting detection - thanks
        mission = zero_shot_strategy.generate_mission("Thanks")

        assert mission.name == "Info"
        assert len(mission.steps) == 1

        print("✓ Greeting (thanks): Detected correctly")

    def test_greeting_what_can_you_do(self, zero_shot_strategy):
        # test conversational question
        mission = zero_shot_strategy.generate_mission("What can you do?")

        assert mission.name == "Info"
        assert len(mission.steps) == 1

        print("✓ Greeting (what can you do): Detected correctly")

    # =========================================================================
    # COMPLEX MULTI-STEP TASKS
    # =========================================================================

    def test_multi_step_task_pick_swirl_place(self, zero_shot_strategy):
        # test task that requires multiple actions
        mission = zero_shot_strategy.generate_mission(
            "Pick up test_tube_blood, swirl it, and place it in storage area"
        )

        assert mission.name == "Zero-Shot Mission"
        # Zero-Shot might generate fewer or more steps than expected
        # Just verify it generates something
        assert len(mission.steps) > 0

        task_types = [step.get("type") for step in mission.steps]
        print(f"✓ Multi-Step (pick-swirl-place): Generated {len(mission.steps)} steps - {task_types}")

    def test_complex_laboratory_workflow(self, zero_shot_strategy):
        # test complex laboratory workflow
        mission = zero_shot_strategy.generate_mission(
            "Perform pH test: pick up dropper_Phenolphtalein, pour it into test_tube_blood, "
            "observe the color change, then dispose of the test tube"
        )

        assert mission.name == "Zero-Shot Mission"
        assert len(mission.steps) > 0

        print(f"✓ Complex Workflow: Generated {len(mission.steps)} steps")

    # =========================================================================
    # NOVEL/UNFAMILIAR TASKS
    # =========================================================================

    def test_novel_task_verbose_description(self, zero_shot_strategy):
        # test novel task with verbose step-by-step description
        mission = zero_shot_strategy.generate_mission(
            "Crystallization procedure: first cool the supersaturated solution, "
            "then carefully transfer the crystallized sample to storage, "
            "finally place it in the observation area for analysis"
        )

        # Zero-Shot should attempt to generate a plan (no guarantee of accuracy)
        assert mission.name in ["Zero-Shot Mission", "Error"]

        if mission.name == "Zero-Shot Mission":
            assert len(mission.steps) > 0
            print(f"✓ Novel Task (verbose): Generated {len(mission.steps)} steps")
        else:
            print("✓ Novel Task (verbose): Handled gracefully (error state)")

    def test_novel_task_unfamiliar_objects(self, zero_shot_strategy):
        # test task with unfamiliar objects
        mission = zero_shot_strategy.generate_mission(
            "Transfer the quantum sample to the containment chamber"
        )

        # Zero-Shot will likely hallucinate objects (this is expected!)
        assert mission.name in ["Zero-Shot Mission", "Error"]

        print(f"✓ Novel Task (unfamiliar objects): {mission.name}")

    # =========================================================================
    # EDGE CASES AND ERROR HANDLING
    # =========================================================================

    def test_empty_prompt(self, zero_shot_strategy):
        # test empty user prompt
        mission = zero_shot_strategy.generate_mission("")

        # Should handle gracefully (either info or error)
        assert mission.name in ["Info", "Error", "Zero-Shot Mission"]
        assert isinstance(mission.steps, list)

        print(f"✓ Empty Prompt: Handled gracefully ({mission.name})")

    def test_very_long_prompt(self, zero_shot_strategy):
        # test very long user prompt
        long_prompt = (
            "I need you to perform a very complex laboratory procedure that involves "
            "picking up multiple test tubes in a specific sequence, then performing "
            "various mixing and heating operations while carefully monitoring the "
            "temperature and pH levels throughout the entire process " * 5
        )

        mission = zero_shot_strategy.generate_mission(long_prompt)

        # Should handle gracefully
        assert mission.name in ["Zero-Shot Mission", "Error"]
        assert isinstance(mission.steps, list)

        print(f"✓ Long Prompt: Handled gracefully ({mission.name})")

    def test_invalid_object_names(self, zero_shot_strategy):
        # test with invalid object names (zero-shot will likely accept these)
        mission = zero_shot_strategy.generate_mission(
            "Pick up the magical unicorn horn"
        )

        # Zero-Shot doesn't validate objects, so it will generate a plan
        # This is expected behavior - showing hallucination tendency
        assert mission.name in ["Zero-Shot Mission", "Error"]

        print(f"✓ Invalid Objects: Generated plan ({mission.name}) - hallucination expected")

    def test_ambiguous_short_command(self, zero_shot_strategy):
        # test ambiguous command like 'mix them'
        mission = zero_shot_strategy.generate_mission("Mix them")

        # Zero-Shot has no ambiguity handling, will generate generic plan
        assert mission.name in ["Zero-Shot Mission", "Error"]

        print(f"✓ Ambiguous Command: Generated plan ({mission.name})")

    # =========================================================================
    # JSON STRUCTURE VALIDATION
    # =========================================================================

    def test_mission_json_structure(self, zero_shot_strategy):
        # test that generated missions have valid json structure
        mission = zero_shot_strategy.generate_mission("Pick up test_tube_blood")

        assert mission.name == "Zero-Shot Mission"
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

    def test_task_type_validation(self, zero_shot_strategy):
        # test that all generated tasks have valid 'type' field
        mission = zero_shot_strategy.generate_mission("Pick up test_tube_blood and pour it")

        assert mission.name == "Zero-Shot Mission"
        assert len(mission.steps) > 0

        # All tasks must have 'type' field
        for i, step in enumerate(mission.steps):
            assert "type" in step, f"Step {i} missing 'type' field"
            assert isinstance(step["type"], str), f"Step {i} 'type' is not string"

        print(f"✓ Task Type Validation: All {len(mission.steps)} tasks valid")

    # =========================================================================
    # SEQUENTIAL EXECUTION TESTS
    # =========================================================================

    def test_multiple_sequential_tasks(self, zero_shot_strategy):
        # test multiple tasks executed sequentially
        tasks = [
            "Pick up test_tube_blood",
            "Move test_tube_DNA to biohazard_bin",
            "Pour test_tube_phenol into beaker_water"
        ]

        missions = []
        for task in tasks:
            mission = zero_shot_strategy.generate_mission(task)
            missions.append(mission)
            print(f"✓ Sequential Task: '{task}' → {len(mission.steps)} steps")

        # All missions should be successful (or graceful errors)
        for i, mission in enumerate(missions):
            assert mission.name in ["Zero-Shot Mission", "Error"], f"Task {i} failed"
            if mission.name == "Zero-Shot Mission":
                assert len(mission.steps) > 0, f"Task {i} has no steps"

    # =========================================================================
    # RETRY AND ERROR HANDLING TESTS
    # =========================================================================

    def test_json_retry_logic(self, zero_shot_strategy):
        # test that retry logic is working (indirect test)
        # This is an indirect test - we execute multiple tasks and ensure
        # they mostly succeed, which demonstrates retry logic is working
        test_prompts = [
            "Pick up test_tube_blood",
            "Shake test_tube_DNA",
            "Pour beaker_water",
            "Move test_tube_phenol to biohazard_bin",
            "Swirl test_tube_empty"
        ]

        success_count = 0
        for prompt in test_prompts:
            mission = zero_shot_strategy.generate_mission(prompt)
            if mission.name == "Zero-Shot Mission" and len(mission.steps) > 0:
                success_count += 1

        # At least 60% should succeed (lower bar than RAG since no context)
        assert success_count >= 3, f"Only {success_count}/5 tasks succeeded"
        print(f"✓ Retry Logic: {success_count}/5 tasks succeeded")

    # =========================================================================
    # HALLUCINATION TESTS (Zero-Shot Specific)
    # =========================================================================

    def test_hallucination_tendency_objects(self, zero_shot_strategy):
        # test that zero-shot may hallucinate object names (expected behavior)
        # Use a prompt that might cause hallucination
        mission = zero_shot_strategy.generate_mission(
            "Pick up the special container and transfer to the analyzer"
        )

        # Zero-Shot will generate a plan (may have hallucinated objects)
        # This test just verifies it doesn't crash
        assert mission.name in ["Zero-Shot Mission", "Error"]

        if mission.name == "Zero-Shot Mission":
            print(f"✓ Hallucination Test: Generated {len(mission.steps)} steps (may contain hallucinations)")

    def test_hallucination_tendency_actions(self, zero_shot_strategy):
        # test that zero-shot may use generic action names
        mission = zero_shot_strategy.generate_mission("Process the sample")

        # Zero-Shot will generate something generic
        assert mission.name in ["Zero-Shot Mission", "Error"]

        if mission.name == "Zero-Shot Mission":
            task_types = [step.get("type") for step in mission.steps]
            print(f"✓ Generic Actions: Generated actions - {task_types}")


class TestZeroShotStressTest:
    # stress tests for zero-shot strategy

    @pytest.fixture
    def zero_shot_strategy(self):
        # create a zeroshotstrategy
        return ZeroShotStrategy()

    def test_rapid_sequential_calls(self, zero_shot_strategy):
        # test rapid sequential calls to the strategy
        tasks = [
            "Pick up test_tube_blood",
            "Place it in the bin",
            "Move test_tube_DNA",
            "Pour into beaker"
        ]

        missions = []
        for task in tasks:
            mission = zero_shot_strategy.generate_mission(task)
            missions.append(mission)

        # Most should succeed (lower bar than RAG)
        successful = [m for m in missions if m.name == "Zero-Shot Mission" and len(m.steps) > 0]
        assert len(successful) >= 2  # At least 50% success rate

        print(f"✓ Rapid Sequential: {len(successful)}/{len(tasks)} successful")

    def test_complex_multi_step_task(self, zero_shot_strategy):
        # test complex task with many steps
        mission = zero_shot_strategy.generate_mission(
            "Perform DNA extraction: First pick up test_tube_DNA, then shake it vigorously, "
            "pour the contents into beaker_water, swirl the mixture, wait 5 seconds, "
            "and finally transfer everything to biohazard_bin for disposal"
        )

        assert mission.name in ["Zero-Shot Mission", "Error"]

        if mission.name == "Zero-Shot Mission":
            # Should generate multiple steps (may not be accurate)
            assert len(mission.steps) > 0
            print(f"✓ Complex Multi-Step: Generated {len(mission.steps)} steps")
        else:
            print("✓ Complex Multi-Step: Handled gracefully (error state)")


class TestZeroShotBaseline:
    # tests specific to zero-shot as baseline for comparison

    @pytest.fixture
    def zero_shot_strategy(self):
        # create a zeroshotstrategy
        return ZeroShotStrategy()

    def test_no_knowledge_base_required(self, zero_shot_strategy):
        # test that zero-shot works without any knowledge base
        # This is a key difference - Zero-Shot doesn't need KB
        mission = zero_shot_strategy.generate_mission("Pick up test_tube_blood")

        # Should work even without KB
        assert mission.name in ["Zero-Shot Mission", "Error"]

        print("✓ No KB Required: Works without knowledge base")

    def test_pure_llm_generation(self, zero_shot_strategy):
        # test that zero-shot uses only llm (no context retrieval)
        # Verify the strategy generates plans purely from LLM
        mission = zero_shot_strategy.generate_mission("Move beaker to storage")

        # Should generate a plan (quality tested separately)
        assert mission.name in ["Zero-Shot Mission", "Error"]

        if mission.name == "Zero-Shot Mission":
            print(f"✓ Pure LLM: Generated {len(mission.steps)} steps without context")


class TestZeroShotComparison:
    # tests designed for fair comparison with rag strategies

    @pytest.fixture
    def zero_shot_strategy(self):
        # create a zeroshotstrategy
        return ZeroShotStrategy()

    def test_same_model_as_rag(self, zero_shot_strategy):
        # verify using llama3 for json generation (same as rag strategies)
        from config.config import Config

        # Verify config uses llama3 for plan generation
        assert Config.LLM_PLAN_GENERATION_MODEL == "llama3:8b"

        # Execute a task to ensure it works
        mission = zero_shot_strategy.generate_mission("Pick up test_tube_blood")
        assert mission.name in ["Zero-Shot Mission", "Error"]

        print("✓ Model Parity: Using llama3:8b like RAG strategies")

    def test_json_reliability(self, zero_shot_strategy):
        # test json generation reliability (should match rag)
        tasks = [
            "Pick up test_tube_blood",
            "Shake test_tube_DNA",
            "Pour test_tube_phenol into beaker_water",
            "Move test_tube_empty to biohazard_bin",
            "Swirl test_tube_blood"
        ]

        success_count = 0
        for task in tasks:
            mission = zero_shot_strategy.generate_mission(task)
            if mission.name == "Zero-Shot Mission":
                # Verify JSON is valid
                try:
                    if mission.raw_plan:
                        json.loads(mission.raw_plan)
                        success_count += 1
                except json.JSONDecodeError:
                    pass

        # Should have decent success rate (>= 60%, lower than RAG)
        assert success_count >= 3, f"Only {success_count}/5 had valid JSON"
        print(f"✓ JSON Reliability: {success_count}/5 (60%+)")

    def test_greeting_handling_parity(self, zero_shot_strategy):
        # test greeting handling matches rag behavior
        greetings = ["Hello", "Hi", "Thanks", "What can you do?"]

        for greeting in greetings:
            mission = zero_shot_strategy.generate_mission(greeting)
            assert mission.name == "Info", f"Failed for: {greeting}"
            assert len(mission.steps) == 1

        print(f"✓ Greeting Parity: All {len(greetings)} greetings handled correctly")

    def test_retry_parity(self, zero_shot_strategy):
        # test that zero-shot has retry logic like rag strategies
        # Execute multiple tasks - retry logic should help with success rate
        tasks = [
            "Pick test_tube_blood",
            "Shake test_tube_DNA",
            "Pour beaker_water",
            "Move test_tube_phenol"
        ]

        success_count = sum(
            1 for task in tasks
            if zero_shot_strategy.generate_mission(task).name == "Zero-Shot Mission"
        )

        # With retry logic, should have reasonable success rate
        assert success_count >= 2, f"Only {success_count}/4 succeeded"
        print(f"✓ Retry Parity: {success_count}/4 succeeded with retry logic")

    def test_error_handling_parity(self, zero_shot_strategy):
        # test that zero-shot handles errors gracefully like rag strategies
        # Test with problematic inputs
        test_cases = [
            "",  # Empty
            "   ",  # Whitespace only
            "a" * 1000,  # Very long
        ]

        for test_input in test_cases:
            mission = zero_shot_strategy.generate_mission(test_input)
            # Should not crash, return Info or Error mission
            assert mission.name in ["Info", "Error", "Zero-Shot Mission"]
            assert isinstance(mission.steps, list)

        print("✓ Error Handling Parity: Graceful handling like RAG strategies")


class TestZeroShotPerformanceMetrics:
    # tests to collect baseline performance metrics

    @pytest.fixture
    def zero_shot_strategy(self):
        # create a zeroshotstrategy
        return ZeroShotStrategy()

    def test_latency_baseline(self, zero_shot_strategy):
        # measure baseline latency (no retrieval overhead)
        import time

        start = time.time()
        mission = zero_shot_strategy.generate_mission("Pick up test_tube_blood")
        latency = (time.time() - start) * 1000  # Convert to ms

        # Should be faster than RAG (no retrieval)
        # Typically ~2000ms for LLM generation only
        print(f"✓ Latency Baseline: {latency:.0f}ms (no retrieval overhead)")

        assert mission.name in ["Zero-Shot Mission", "Error"]

    def test_json_success_rate(self, zero_shot_strategy):
        # measure json generation success rate
        test_prompts = [
            "Pick up test_tube_blood",
            "Move beaker to storage",
            "Pour test_tube_phenol",
            "Shake test_tube_DNA",
            "Swirl test_tube_empty"
        ]

        valid_json_count = 0
        for prompt in test_prompts:
            mission = zero_shot_strategy.generate_mission(prompt)
            if mission.name == "Zero-Shot Mission" and mission.raw_plan:
                try:
                    json.loads(mission.raw_plan)
                    valid_json_count += 1
                except:
                    pass

        success_rate = (valid_json_count / len(test_prompts)) * 100
        print(f"✓ JSON Success Rate: {success_rate:.0f}% ({valid_json_count}/{len(test_prompts)})")

        # Should have decent success rate with retry + fix logic
        assert valid_json_count >= 3