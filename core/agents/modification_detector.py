# modification detector agent
import ollama
import json
import re
from core.agents.types import ModificationDetectionInput, ModificationDetectionResult
from config.config import Config


class ModificationDetectorAgent:
    def __init__(self):
        self.modification_prompt_template = self._build_prompt_template()

    def detect_modification(self, input_data: ModificationDetectionInput) -> ModificationDetectionResult:
        print(" [Modification Detector] Analyzing for modifications...")

        # Quick pre-check: Only call LLM if modification keywords are present
        modification_keywords = ["but", "except", "without", "skip", "don't", "dont", "add", "also", "extra", "instead of", "faster", "slower", "however", "although"]
        prompt_lower = input_data.user_prompt.lower()

        has_keyword = any(keyword in prompt_lower for keyword in modification_keywords)

        if not has_keyword:
            print(" [Modification Detector] No modification keywords found - executing recipe as-is")
            return ModificationDetectionResult(
                has_modification=False,
                modification_description=None,
                modification_type=None
            )

        # Build prompt for LLM
        # Use replace instead of format to avoid KeyError from curly braces in matched_recipe JSON
        prompt = self.modification_prompt_template
        prompt = prompt.replace("{user_prompt}", input_data.user_prompt)
        prompt = prompt.replace("{matched_recipe}", input_data.matched_recipe)

        # Call LLM
        try:
            response = ollama.chat(
                model=Config.LLM_MODEL,
                messages=[{'role': 'user', 'content': prompt}]
            )

            raw_content = response['message']['content'].strip()
            print(f" [Modification Detector] LLM Response: {raw_content[:100]}...")

            # Parse JSON response
            clean_content = self._clean_json(raw_content)
            result_data = json.loads(clean_content)

            has_modification = result_data.get("has_modification", False)
            modification_description = result_data.get("modification_description", None)
            modification_type = result_data.get("modification_type", None)

            if has_modification:
                print(f" [Modification Detector] Modification detected: {modification_type}")
                print(f"   > Description: {modification_description}")
            else:
                print(" [Modification Detector] No modifications detected")

            return ModificationDetectionResult(
                has_modification=has_modification,
                modification_description=modification_description,
                modification_type=modification_type
            )

        except json.JSONDecodeError as e:
            print(f" [Modification Detector] JSON parsing error: {e}")
            print(f" [Modification Detector] Raw content: {raw_content}")
            # Fallback: assume no modification if parsing fails
            return ModificationDetectionResult(
                has_modification=False,
                modification_description=None,
                modification_type=None
            )

        except Exception as e:
            print(f" [Modification Detector] Error: {e}")
            return ModificationDetectionResult(
                has_modification=False,
                modification_description=None,
                modification_type=None
            )

    def _build_prompt_template(self) -> str:
        return """You are a modification detector. Detect ONLY if user wants to CHANGE a matched recipe.

USER: "{user_prompt}"
RECIPE: {matched_recipe}

RULES:
1. If user request is just executing the recipe with specific objects → has_modification=FALSE
2. ONLY return has_modification=TRUE if user explicitly uses words like: "but", "except", "without", "skip", "add", "also", "instead of", "faster", "slower"

EXAMPLES:

User: "Pick up test_tube_blood"
→ {{"has_modification": false, "modification_description": null, "modification_type": null}}

User: "Move DNA to bin"
→ {{"has_modification": false, "modification_description": null, "modification_type": null}}

User: "Do DNA extraction but skip heating"
→ {{"has_modification": true, "modification_description": "Skip heating", "modification_type": "remove"}}

User: "Pick blood but shake it first"
→ {{"has_modification": true, "modification_description": "Add shake step", "modification_type": "add"}}

NOW ANALYZE. Return JSON only:
"""

    def _clean_json(self, text: str) -> str:
        # Extract JSON using regex (find first {...})
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            text = match.group(0)

        # Remove markdown code blocks
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]

        return text.strip()