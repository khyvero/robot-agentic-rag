# generates and validates json mission plans using llm
# supports tier 2 refinement and tier 3 generation modes

import json
import re
import ollama
from core.agents.types import PlanGenerationInput, PlanGenerationResult
from config.config import Config
from core.strategies.dual_rag.prompts import LLM_SYSTEM_PROMPT, DUAL_RAG_USER_TEMPLATE, TEACH_MODE_PROMPT


class PlanGeneratorAgent:
    # generates json mission plans using llm with tier 2 refinement or tier 3 generation modes

    def __init__(self):
        pass

    def generate_plan(self, input_data: PlanGenerationInput, max_retries: int = 1) -> PlanGenerationResult:
        # generates mission plan with retry logic for malformed json and validation failures
        print(f" [Plan Generator] Generating plan in {input_data.mode} mode...")

        # select appropriate prompt template based on mode
        if input_data.mode == "TIER_2_REFINEMENT":
            user_message = DUAL_RAG_USER_TEMPLATE.format(
                intent_context=input_data.intent_context,
                procedural_context=input_data.procedural_context,
                user_prompt=input_data.user_prompt
            )
        elif input_data.mode == "TIER_3_GENERATION":
            user_message = TEACH_MODE_PROMPT.format(
                api_context=input_data.procedural_context,
                user_prompt=input_data.user_prompt,
                valid_objects=str(Config.VALID_OBJECTS)
            )
        else:
            return PlanGenerationResult(
                plan_json="",
                success=False,
                error=f"Invalid mode: {input_data.mode}"
            )

        # retry logic with dynamic prompt injection for validation errors
        validation_errors_from_previous_attempt = []

        for attempt in range(max_retries + 1):
            if attempt > 0:
                print(f" [Plan Generator] Retry attempt {attempt}/{max_retries}...")

            try:
                # build user message with error hints if retrying
                current_user_message = user_message
                if validation_errors_from_previous_attempt:
                    error_hints = self._build_error_hints(validation_errors_from_previous_attempt)
                    if error_hints:
                        current_user_message += f"\n\n⚠️ CRITICAL CORRECTIONS NEEDED:\n{error_hints}"
                        print(f" [Plan Generator] Injecting error hints into prompt")

                # use higher temperature on retries to generate different outputs
                temperature = Config.LLM_TEMPERATURE if attempt == 0 else min(0.3 + (attempt * 0.2), 0.9)
                if attempt > 0:
                    print(f" [Plan Generator] Using temperature={temperature:.1f} for retry")

                # use specialized json generation model
                response = ollama.chat(
                    model=Config.LLM_PLAN_GENERATION_MODEL,
                    messages=[
                        {'role': 'system', 'content': LLM_SYSTEM_PROMPT},
                        {'role': 'user', 'content': current_user_message}
                    ],
                    options={'temperature': temperature}
                )

                raw_content = response['message']['content']
                print(f" [Plan Generator] LLM response received ({len(raw_content)} chars)")

                # clean and fix json
                clean_json = self._clean_json(raw_content)
                fixed_json = self._fix_json(clean_json)

                # validate json structure
                try:
                    data = json.loads(fixed_json)

                    # validate required keys
                    if "tasks" not in data or "settings" not in data:
                        if attempt < max_retries:
                            print(f" [Plan Generator] Missing required keys, retrying...")
                            continue
                        return PlanGenerationResult(
                            plan_json=fixed_json,
                            success=False,
                            error="Missing required keys: 'tasks' or 'settings'"
                        )

                    # validate safety constraints with retry
                    validation_errors = self._validate_plan_safety(data)
                    if validation_errors:
                        if attempt < max_retries:
                            print(f" [Plan Generator] Validation failed: {validation_errors}, retrying...")
                            validation_errors_from_previous_attempt = validation_errors
                            continue
                        print(f" [Plan Generator] Validation failed after retries: {validation_errors}")

                    print(f" [Plan Generator] Plan generated successfully ({len(data.get('tasks', []))} tasks)")

                    return PlanGenerationResult(
                        plan_json=fixed_json,
                        success=True,
                        error=None
                    )

                except json.JSONDecodeError as e:
                    if attempt < max_retries:
                        print(f" [Plan Generator] JSON validation failed: {e}, retrying...")
                        continue

                    print(f" [Plan Generator] JSON validation failed after {max_retries} retries: {e}")
                    print(f" [Plan Generator] Raw output:\n{raw_content}")
                    print(f" [Plan Generator] Fixed output:\n{fixed_json}")
                    return PlanGenerationResult(
                        plan_json=fixed_json,
                        success=False,
                        error=f"JSON decode error: {e}"
                    )

            except Exception as e:
                if attempt < max_retries:
                    print(f" [Plan Generator] Error during generation: {e}, retrying...")
                    continue

                print(f" [Plan Generator] Error during generation after {max_retries} retries: {e}")
                return PlanGenerationResult(
                    plan_json="",
                    success=False,
                    error=f"Generation error: {e}"
                )

    def _clean_json(self, text: str) -> str:
        # extract and clean json from llm response
        # removes markdown code blocks, comments, and python literals

        # extract json using regex (find first {...})
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            text = match.group(0)
        else:
            text = text.strip()

        # remove markdown code blocks
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]

        # remove comments
        text = re.sub(r'//.*', '', text)
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

        # replace python literals with json equivalents
        text = text.replace(': None', ': null')
        text = text.replace(':None', ':null')
        text = text.replace(': True', ': true')
        text = text.replace(':True', ':true')
        text = text.replace(': False', ': false')
        text = text.replace(':False', ':false')

        return text.strip()

    def _fix_json(self, text: str) -> str:
        # fix common json formatting issues: missing braces/brackets, trailing commas

        # count braces and brackets
        open_braces = text.count('{')
        close_braces = text.count('}')
        open_brackets = text.count('[')
        close_brackets = text.count(']')

        # fix missing closing brackets (arrays)
        if open_brackets > close_brackets:
            missing_brackets = open_brackets - close_brackets
            print(f" [Plan Generator] Adding {missing_brackets} missing closing bracket(s)")
            text += ']' * missing_brackets

        # fix missing closing braces (objects)
        if open_braces > close_braces:
            missing_braces = open_braces - close_braces
            print(f" [Plan Generator] Adding {missing_braces} missing closing brace(s)")
            text += '}' * missing_braces

        # remove trailing commas
        text = re.sub(r',(\s*[}\]])', r'\1', text)

        return text.strip()

    def _validate_plan_safety(self, data: dict) -> list:
        # validate safety constraints on generated plan
        # returns list of validation errors, empty if valid
        errors = []

        # validate simulation speed
        speed = data.get("settings", {}).get("simulation_speed", 1)
        if not isinstance(speed, int) or not (1 <= speed <= 5):
            errors.append(f"Invalid simulation_speed: {speed}")

        # validate tasks
        tasks = data.get("tasks", [])
        if not tasks:
            errors.append("No tasks in plan")
            return errors

        for i, task in enumerate(tasks):
            task_type = task.get("type", "")

            # validate action types and required parameters
            if task_type == "pick":
                if "target_obj_name" not in task:
                    errors.append(f"Task {i}: 'pick' missing 'target_obj_name'")

            elif task_type == "place":
                if "destination_obj_name" not in task and "destination_coords" not in task:
                    errors.append(f"Task {i}: 'place' missing 'destination_obj_name' or 'destination_coords'")

            elif task_type == "pour":
                if "target_container_name" not in task:
                    errors.append(f"Task {i}: 'pour' missing 'target_container_name'")

            elif task_type in ["shake", "swirl"]:
                if "target_obj_name" not in task:
                    errors.append(f"Task {i}: '{task_type}' missing 'target_obj_name'")

            elif task_type == "wait":
                if "duration_seconds" not in task:
                    errors.append(f"Task {i}: 'wait' missing 'duration_seconds'")

            elif task_type == "place_in_area":
                if "area_bounds" not in task:
                    errors.append(f"Task {i}: 'place_in_area' missing 'area_bounds'")

        return errors

    def _build_error_hints(self, errors: list) -> str:
        # build targeted error hints for llm to fix plan on retry
        hints = []

        for error in errors:
            # wait action missing duration
            if "'wait' missing 'duration_seconds'" in error:
                hints.append(
                    "- The 'wait' action MUST include 'duration_seconds' parameter.\n"
                    "  Correct format: {\"type\": \"wait\", \"duration_seconds\": 3}"
                )

            # pour action missing target container
            elif "'pour' missing 'target_container_name'" in error:
                hints.append(
                    "- The 'pour' action MUST include 'target_container_name' parameter.\n"
                    "  Correct format: {\"type\": \"pour\", \"target_container_name\": \"beaker_water\"}"
                )

            # pick action missing target object
            elif "'pick' missing 'target_obj_name'" in error:
                hints.append(
                    "- The 'pick' action MUST include 'target_obj_name' parameter.\n"
                    "  Correct format: {\"type\": \"pick\", \"target_obj_name\": \"test_tube_blood\"}"
                )

            # place action missing destination
            elif "'place' missing 'destination_obj_name' or 'destination_coords'" in error:
                hints.append(
                    "- The 'place' action MUST include either 'destination_obj_name' OR 'destination_coords'.\n"
                    "  Correct format: {\"type\": \"place\", \"destination_obj_name\": \"storage_rack\"} OR\n"
                    "  {\"type\": \"place\", \"destination_coords\": {\"x\": 200, \"y\": 300}}"
                )

            # shake/swirl action missing target
            elif "'shake' missing 'target_obj_name'" in error or "'swirl' missing 'target_obj_name'" in error:
                action = "shake" if "shake" in error else "swirl"
                hints.append(
                    f"- The '{action}' action MUST include 'target_obj_name' parameter.\n"
                    f"  Correct format: {{\"type\": \"{action}\", \"target_obj_name\": \"test_tube_reagent\"}}"
                )

            # place_in_area missing area_bounds
            elif "'place_in_area' missing 'area_bounds'" in error:
                hints.append(
                    "- The 'place_in_area' action MUST include 'area_bounds' parameter.\n"
                    "  Correct format: {\"type\": \"place_in_area\", \"area_bounds\": {\"x_min\": 100, \"x_max\": 300, \"y_min\": 200, \"y_max\": 400}}"
                )

            # invalid simulation speed
            elif "Invalid simulation_speed" in error:
                hints.append(
                    "- The 'simulation_speed' in settings MUST be an integer between 1 and 5.\n"
                    "  Correct format: \"settings\": {\"simulation_speed\": 3}"
                )

        if not hints:
            return ""

        # deduplicate hints
        unique_hints = []
        seen = set()
        for hint in hints:
            hint_type = hint.split('\n')[0]
            if hint_type not in seen:
                unique_hints.append(hint)
                seen.add(hint_type)

        return "\n".join(unique_hints)