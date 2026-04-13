# handles multi-turn dialogue for ambiguity resolution and plan review

import json
import ollama
from typing import List, Tuple
from core.agents.types import ConversationInput, ConversationResult
from core.domain import RobotMission
from config.config import Config
from core.strategies.dual_rag.prompts import (
    AMBIGUITY_RESOLUTION_PROMPT,
    SELECTION_PROMPT,
    PLAN_MODIFICATION_PROMPT,
    LLM_SYSTEM_PROMPT
)


class ConversationAgent:
    # manages multi-turn dialogue for ambiguity, plan review, and plan modification

    def __init__(self):
        pass

    def generate_clarification(
        self,
        user_prompt: str,
        candidates: List[Tuple[str, float]]
    ) -> RobotMission:
        # generate clarification question for ambiguous user input
        print(" [Conversation] Generating clarification question...")

        # format candidates list
        candidate_str = ""
        for i, (doc, dist) in enumerate(candidates):
            try:
                data = json.loads(doc)
                name = data.get("mission_name", "Unknown Task")
                candidate_str += f"Option {i + 1}: {name}\n"
            except json.JSONDecodeError:
                candidate_str += f"Option {i + 1}: [Unable to parse]\n"

        # build prompt and call llm
        prompt = AMBIGUITY_RESOLUTION_PROMPT.format(
            user_prompt=user_prompt,
            candidate_list=candidate_str,
            num_matches=len(candidates)
        )
        response = ollama.chat(
            model=Config.LLM_MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )

        question = response['message']['content']

        return RobotMission(
            name="Clarification Question",
            steps=[{"type": "ask_user", "question": question}],
            settings={"simulation_speed": 1}
        )

    def interpret_selection(
        self,
        user_reply: str,
        candidates: List[Tuple[str, float]]
    ) -> ConversationResult:
        # interpret user's response to clarification question
        print(" [Conversation] Interpreting user selection...")

        # build candidates string and prompt for llm
        candidate_str = ""
        for i, (doc, _) in enumerate(candidates):
            candidate_str += f"Option {i + 1}: {doc}\n"
        prompt = SELECTION_PROMPT.format(
            candidate_list=candidate_str,
            user_reply=user_reply
        )

        response = ollama.chat(
            model=Config.LLM_MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )

        content = response['message']['content'].strip()

        # clean markdown code blocks
        if "```" in content:
            content = content.replace("```json", "").replace("```", "").strip()

        # interpret llm response
        if "NEW_PLAN" in content:
            print("   > User requested new plan")
            return ConversationResult(
                mission=RobotMission(name="Info", steps=[], settings={}),
                state_transition="RESET_TO_IDLE",
                next_action="GENERATE_PLAN"
            )

        elif "CONTINUE" in content:
            print("   > User provided more context")
            return ConversationResult(
                mission=RobotMission(name="Info", steps=[], settings={}),
                state_transition="RESET_TO_IDLE",
                next_action="RETRY_PIPELINE"
            )

        else:
            print("   > User selected an existing option")
            # Content should be the JSON of the selected mission
            return ConversationResult(
                mission=RobotMission(name="Info", steps=[], settings={}),
                state_transition="RESET_TO_IDLE",
                next_action="EXECUTE"
            )

    def review_plan(
        self,
        plan_json: str,
        updated: bool = False
    ) -> RobotMission:
        # generate human-readable plan review question
        print(f" [Conversation] Generating plan review ({'updated' if updated else 'initial'})...")

        try:
            plan_data = json.loads(plan_json)

            # generate human-readable summary
            summary_lines = []
            for i, task in enumerate(plan_data.get("tasks", []), 1):
                task_type = task.get("type")
                target = task.get("target_obj_name", "Unknown")
                dest = task.get("destination_obj_name")

                if task_type == "pick":
                    desc = f"Pick up {target}"
                elif task_type == "place":
                    location = dest if dest else f"Coords {task.get('destination_coords')}"
                    desc = f"Place it on {location}"
                elif task_type == "place_free_spot":
                    location = dest if dest else f"Coords {task.get('destination_coords')}"
                    desc = f"Place it near {location}"
                elif task_type == "pour":
                    desc = f"Pour into {target}"
                elif task_type == "shake":
                    desc = f"Shake {target}"
                elif task_type == "swirl":
                    desc = f"Swirl {target}"
                elif task_type == "wait":
                    seconds = task.get("seconds", 0)
                    desc = f"Wait {seconds} seconds"
                elif task_type == "move_home":
                    desc = "Move to home position"
                elif task_type == "ensure_gripper_empty":
                    desc = "Ensure gripper is empty"
                else:
                    desc = f"{task_type} {target}"

                summary_lines.append(f"{i}. {desc}")

            steps_summary = "\n".join(summary_lines)
            prefix = "Updated Plan:" if updated else f"I have generated a custom plan with {len(plan_data.get('tasks', []))} steps:"
            question = f"{prefix}\n{steps_summary}\n\nType 'yes' to execute, or tell me what to change."

        except json.JSONDecodeError as e:
            print(f" [Conversation] Error parsing plan JSON: {e}")
            question = f"Plan generation failed (JSON error: {e}). Please try rephrasing your request."

        except Exception as e:
            print(f" [Conversation] Error creating review: {e}")
            question = f"Plan review failed: {e}. Please try again."

        return RobotMission(
            name="Plan Review",
            steps=[{"type": "ask_user", "question": question}],
            settings={"simulation_speed": 1}
        )

    def modify_plan(
        self,
        current_plan: str,
        user_feedback: str
    ) -> str:
        # modify existing plan based on user feedback
        print(" [Conversation] Modifying plan based on user feedback...")

        # build prompt and call specialized json generation model
        prompt = PLAN_MODIFICATION_PROMPT.format(
            current_plan=current_plan,
            user_feedback=user_feedback
        )
        response = ollama.chat(
            model=Config.LLM_PLAN_GENERATION_MODEL,
            messages=[
                {'role': 'system', 'content': LLM_SYSTEM_PROMPT},
                {'role': 'user', 'content': prompt}
            ]
        )

        raw_content = response['message']['content']

        # clean json
        import re
        match = re.search(r'\{.*\}', raw_content, re.DOTALL)
        if match:
            clean_json = match.group(0)
        else:
            clean_json = raw_content.strip()

        # remove markdown and comments
        if clean_json.startswith("```"):
            clean_json = clean_json.strip("`")
            if clean_json.startswith("json"):
                clean_json = clean_json[4:]
        clean_json = re.sub(r'//.*', '', clean_json)

        return clean_json.strip()

    def handle_plan_confirmation(
        self,
        user_input: str,
        pending_plan_json: str
    ) -> ConversationResult:
        # handle user's response to plan review (yes/no/modify)
        print(" [Conversation] Processing plan confirmation...")

        # check for confirmation
        if user_input.lower() in ["yes", "ok", "confirm", "execute", "go", "y"]:
            print("   > User confirmed plan")

            try:
                data = json.loads(pending_plan_json)
                mission = RobotMission(
                    name="Dual RAG (Generated)",
                    steps=data.get("tasks", []),
                    settings=data.get("settings", {"simulation_speed": 1})
                )

                return ConversationResult(
                    mission=mission,
                    state_transition="RESET_TO_IDLE",
                    next_action="EXECUTE"
                )

            except json.JSONDecodeError as e:
                print(f" [Conversation] Error parsing plan: {e}")
                error_mission = RobotMission(
                    name="Error",
                    steps=[{"type": "ask_user", "question": f"Error parsing plan: {e}"}],
                    settings={"simulation_speed": 1}
                )

                return ConversationResult(
                    mission=error_mission,
                    state_transition="RESET_TO_IDLE",
                    next_action=None
                )

        else:
            print("   > User requested modifications")

            # modify plan and generate review
            modified_plan = self.modify_plan(pending_plan_json, user_input)
            review_mission = self.review_plan(modified_plan, updated=True)

            return ConversationResult(
                mission=review_mission,
                state_transition="STAY",
                next_action=None
            )