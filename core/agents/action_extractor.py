# extracts required robot actions from user prompts using llm
# loads action metadata dynamically from procedural knowledge base

import json
import ollama
from typing import List, Dict, Any
from core.agents.types import ActionExtractionInput, ActionExtractionResult
from core.knowledge_base import KnowledgeBase
from config.config import Config
from core.strategies.dual_rag.prompts import ACTION_EXTRACTION_PROMPT


class ActionExtractorAgent:
    # extracts required robot actions using llm for targeted procedural retrieval

    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        self.available_actions = self._load_all_actions()
        print(f" [Action Extractor] Loaded {len(self.available_actions)} available actions")

    def _load_all_actions(self) -> List[Dict[str, Any]]:
        # load all procedural apis from knowledge base and extract action metadata
        procedural_text = self.kb.query_procedural("robot actions", n_results=50)

        actions = []
        try:
            # read from json file directly for cleaner parsing
            import os
            json_path = os.path.join(os.path.dirname(__file__), "../../data/knowledge/procedural_api.json")
            with open(json_path, 'r') as f:
                api_data = json.load(f)

            for api in api_data:
                action_dict = {
                    "name": api.get("action_name", ""),
                    "keywords": api.get("action_keywords", []),
                    "description": api.get("description", ""),
                    "function_signature": api.get("function_signature", "")
                }
                actions.append(action_dict)

        except Exception as e:
            print(f" [Action Extractor] Warning: Failed to load actions: {e}")
            return []

        return actions

    def extract_actions(self, input_data: ActionExtractionInput) -> ActionExtractionResult:
        # extract required actions from user prompt and optional intent text
        if not Config.ACTION_EXTRACTION_ENABLED:
            print(" [Action Extractor] Feature disabled, returning empty actions")
            return ActionExtractionResult(actions=[], reasoning="Action extraction disabled")

        print(" [Action Extractor] Extracting actions from prompt...")

        # build actions context and intent section for prompt
        actions_context = self._format_actions_context(self.available_actions)

        intent_section = ""
        if input_data.intent_text:
            intent_section = f"\nRECIPE STEPS (from similar past task):\n{input_data.intent_text}\n"
        prompt = ACTION_EXTRACTION_PROMPT.format(
            actions_context=actions_context,
            user_prompt=input_data.user_prompt,
            intent_section=intent_section
        )

        # call llm to extract actions
        try:
            response = ollama.chat(
                model=Config.LLM_MODEL,
                messages=[{'role': 'user', 'content': prompt}]
            )

            raw_content = response['message']['content'].strip()
            print(f" [Action Extractor] LLM Response: {raw_content}")

            # parse json array from response
            clean_content = raw_content
            if "```" in clean_content:
                clean_content = clean_content.replace("```json", "").replace("```", "").strip()

            # extract json array using regex
            import re
            json_match = re.search(r'\[.*?\]', clean_content, re.DOTALL)
            if json_match:
                clean_content = json_match.group(0)

            actions = json.loads(clean_content)

            if not isinstance(actions, list):
                print(f" [Action Extractor] Warning: Expected list, got {type(actions)}")
                actions = []

            print(f" [Action Extractor] Extracted Actions: {actions}")

            return ActionExtractionResult(
                actions=actions,
                reasoning=raw_content
            )

        except json.JSONDecodeError as e:
            print(f" [Action Extractor] Error: Failed to parse JSON: {e}")
            print(f" [Action Extractor] Raw content: {raw_content}")
            # fallback extraction from text
            fallback_actions = self._fallback_extraction(raw_content)
            return ActionExtractionResult(
                actions=fallback_actions,
                reasoning=f"Fallback extraction due to JSON error: {e}"
            )

        except Exception as e:
            print(f" [Action Extractor] Error: {e}")
            return ActionExtractionResult(
                actions=[],
                reasoning=f"Extraction failed: {e}"
            )

    def _format_actions_context(self, actions: List[Dict[str, Any]]) -> str:
        # format actions into readable context string for llm
        context_lines = []
        for i, action in enumerate(actions, 1):
            name = action.get("name", "unknown")
            keywords = ", ".join(action.get("keywords", []))
            description = action.get("description", "")

            context_lines.append(
                f"{i}. {name}\n"
                f"   Keywords: {keywords}\n"
                f"   Description: {description}"
            )

        return "\n\n".join(context_lines)

    def _fallback_extraction(self, text: str) -> List[str]:
        # fallback method to extract action names if json parsing fails
        action_names = [action["name"] for action in self.available_actions]
        found_actions = []

        for action_name in action_names:
            if action_name.lower() in text.lower():
                found_actions.append(action_name)

        return found_actions