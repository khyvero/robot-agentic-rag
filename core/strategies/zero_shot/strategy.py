# pure llm generation without rag context retrieval
# no agentic features - baseline for measuring rag impact

import json
import ollama
import re
from core.interfaces import PlanningStrategy
from core.domain import RobotMission
from config.config import Config
from .prompts import LLM_SYSTEM_PROMPT_ZERO_SHOT


class ZeroShotStrategy(PlanningStrategy):
    # pure llm generation without rag - baseline for research

    def __init__(self):
        pass

    def generate_mission(self, user_prompt: str) -> RobotMission:
        # generate robot mission using pure llm without rag context
        print(" [Zero-Shot] Processing user request (no RAG)...")

        # basic greeting detection
        if self._is_greeting(user_prompt):
            print(" [Zero-Shot] Detected greeting/conversation")
            return self._create_info_mission()

        # call llm once with no context retrieval (no retry logic)
        try:
            response = ollama.chat(
                model=Config.LLM_PLAN_GENERATION_MODEL,
                messages=[
                    {'role': 'system', 'content': LLM_SYSTEM_PROMPT_ZERO_SHOT},
                    {'role': 'user', 'content': user_prompt}
                ],
                options={'temperature': Config.LLM_TEMPERATURE}
            )

            raw_content = response['message']['content']
            print(f" [Zero-Shot] LLM response received ({len(raw_content)} chars)")

            # clean and fix json
            clean_json = self._clean_json(raw_content)
            fixed_json = self._fix_json(clean_json)

            # validate json
            try:
                data = json.loads(fixed_json)

                # validate required keys
                if "tasks" not in data or "settings" not in data:
                    print(f" [Zero-Shot] Missing required keys: 'tasks' or 'settings'")
                    return self._create_error_mission("Missing required keys: 'tasks' or 'settings'")

                print(f" [Zero-Shot] Plan generated successfully ({len(data.get('tasks', []))} tasks)")

                return RobotMission(
                    name="Zero-Shot Mission",
                    steps=data.get("tasks", []),
                    settings=data.get("settings", {"simulation_speed": 1}),
                    raw_plan=fixed_json
                )

            except json.JSONDecodeError as e:
                print(f" [Zero-Shot] JSON validation failed: {e}")
                return self._create_error_mission(f"JSON decode error: {e}")

        except Exception as e:
            print(f" [Zero-Shot] Error during generation: {e}")
            return self._create_error_mission(f"Generation error: {e}")

    def _is_greeting(self, user_prompt: str) -> bool:
        # detect if user input is greeting/conversation rather than task command
        prompt_lower = user_prompt.lower().strip()

        greetings = [
            "hello", "hi", "hey", "greetings",
            "good morning", "good afternoon", "good evening",
            "how are you", "what's up", "sup",
            "thanks", "thank you", "bye", "goodbye",
            "what can you do", "help", "who are you"
        ]
        for greeting in greetings:
            if prompt_lower.startswith(greeting) or prompt_lower == greeting:
                return True

        return False

    def _create_info_mission(self) -> RobotMission:
        # create informational mission for greetings/conversations
        info_text = (
            "I'm a laboratory robot assistant. I can help you with tasks like:\n"
            "- Pick up objects\n"
            "- Place or move items\n"
            "- Pour, shake, or swirl contents\n\n"
            "Please provide a robot task command, such as 'Pick up test_tube_blood' or 'Move beaker_water to bin'."
        )

        return RobotMission(
            name="Info",
            steps=[{"type": "ask_user", "question": info_text}],
            settings={"simulation_speed": 1}
        )

    def _create_error_mission(self, error_msg: str) -> RobotMission:
        # create error mission when plan generation fails
        return RobotMission(
            name="Error",
            steps=[],
            settings={"simulation_speed": 1}
        )

    def _clean_json(self, text: str) -> str:
        # extract and clean json from llm response

        # extract json using regex
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

        # fix missing closing brackets
        if open_brackets > close_brackets:
            missing_brackets = open_brackets - close_brackets
            print(f" [Zero-Shot] Adding {missing_brackets} missing closing bracket(s)")
            text += ']' * missing_brackets

        # fix missing closing braces
        if open_braces > close_braces:
            missing_braces = open_braces - close_braces
            print(f" [Zero-Shot] Adding {missing_braces} missing closing brace(s)")
            text += '}' * missing_braces

        # remove trailing commas
        text = re.sub(r',(\s*[}\]])', r'\1', text)

        return text.strip()
