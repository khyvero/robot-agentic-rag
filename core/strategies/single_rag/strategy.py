# uses one unified knowledge base for retrieval (declarative + procedural mixed)
# simple monolithic architecture - research baseline for comparing with dual-rag

import json
import ollama
import re
from core.interfaces import PlanningStrategy
from core.domain import RobotMission
from core.knowledge_base import KnowledgeBase
from config.config import Config
from .prompts import LLM_SYSTEM_PROMPT, SINGLE_RAG_USER_TEMPLATE


class SingleRAGStrategy(PlanningStrategy):
    # unified knowledge base retrieval
    # compares 1 unified source vs dual-rag's 2 separated sources

    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base

    def generate_mission(self, user_prompt: str) -> RobotMission:
        # generate robot mission using unified rag retrieval
        print(" [Single RAG] Processing user request...")

        # basic greeting detection
        if self._is_greeting(user_prompt):
            print(" [Single RAG] Detected greeting/conversation")
            return self._create_info_mission()

        # retrieve from unified knowledge base (key difference from dual-rag)
        context = self.kb.query_unified(user_prompt, n_results=3)
        print(f" [Single RAG] Retrieved {len(context.split('---'))} context items")

        # construct prompt with context
        full_user_message = SINGLE_RAG_USER_TEMPLATE.format(
            context=context,
            user_prompt=user_prompt
        )

        # call llm with no retry logic (max 1 attempt only)
        for attempt in range(1):
            if attempt > 0:
                print(f" [Single RAG] Retry attempt {attempt}/0...")

            try:
                response = ollama.chat(
                    model=Config.LLM_PLAN_GENERATION_MODEL,
                    messages=[
                        {'role': 'system', 'content': LLM_SYSTEM_PROMPT},
                        {'role': 'user', 'content': full_user_message}
                    ],
                    options={'temperature': Config.LLM_TEMPERATURE}
                )

                raw_content = response['message']['content']
                print(f" [Single RAG] LLM response received ({len(raw_content)} chars)")

                # clean and fix json
                clean_json = self._clean_json(raw_content)
                fixed_json = self._fix_json(clean_json)

                # validate json
                try:
                    data = json.loads(fixed_json)

                    # validate required keys
                    if "tasks" not in data or "settings" not in data:
                        if attempt < 1:
                            print(f" [Single RAG] Missing required keys, retrying...")
                            continue
                        return self._create_error_mission("Missing required keys: 'tasks' or 'settings'")

                    print(f" [Single RAG] Plan generated successfully ({len(data.get('tasks', []))} tasks)")

                    return RobotMission(
                        name="Single RAG Mission",
                        steps=data.get("tasks", []),
                        settings=data.get("settings", {"simulation_speed": 1}),
                        raw_plan=fixed_json
                    )

                except json.JSONDecodeError as e:
                    if attempt < 1:
                        print(f" [Single RAG] JSON validation failed: {e}, retrying...")
                        continue

                    print(f" [Single RAG] JSON validation failed after 1 retry: {e}")
                    return self._create_error_mission(f"JSON decode error: {e}")

            except Exception as e:
                if attempt < 1:
                    print(f" [Single RAG] Error during generation: {e}, retrying...")
                    continue

                print(f" [Single RAG] Error during generation after 1 retry: {e}")
                return self._create_error_mission(f"Generation error: {e}")

        return self._create_error_mission("Unknown error")

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
            print(f" [Single RAG] Adding {missing_brackets} missing closing bracket(s)")
            text += ']' * missing_brackets

        # fix missing closing braces
        if open_braces > close_braces:
            missing_braces = open_braces - close_braces
            print(f" [Single RAG] Adding {missing_braces} missing closing brace(s)")
            text += '}' * missing_braces

        # remove trailing commas
        text = re.sub(r',(\s*[}\]])', r'\1', text)

        return text.strip()
