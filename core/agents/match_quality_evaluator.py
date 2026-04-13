# uses llm to intelligently evaluate match quality and make routing decisions
# replaces threshold-based routing with context-aware semantic analysis

import ollama
import json
import re
from core.agents.types import MatchEvaluationInput, MatchEvaluationResult, CandidateMatch
from config.config import Config
from core.strategies.dual_rag.prompts import MATCH_QUALITY_EVALUATION_PROMPT


class MatchQualityEvaluatorAgent:
    # uses llm to analyze candidate matches and decide routing based on semantic alignment
    # returns routing decision: exact_match, ambiguous, novel_task, or not_task

    def __init__(self):
        pass

    def evaluate_matches(self, input_data: MatchEvaluationInput) -> MatchEvaluationResult:
        # evaluate match quality using llm and return routing decision
        print(f" [Match Evaluator] Analyzing {len(input_data.candidates)} candidates...")

        # format candidates for llm prompt
        candidates_formatted = self._format_candidates(input_data.candidates)

        # build prompt with scene objects for context
        prompt = MATCH_QUALITY_EVALUATION_PROMPT.format(
            user_prompt=input_data.user_prompt,
            candidates_formatted=candidates_formatted,
            scene_objects=", ".join(input_data.scene_objects)
        )

        # call llm with deterministic temperature for routing
        try:
            response = ollama.chat(
                model=Config.LLM_MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.0}
            )

            raw_content = response['message']['content'].strip()
            print(f" [Match Evaluator] LLM Response: {raw_content[:150]}...")

            # parse json response
            clean_json = self._clean_json(raw_content)
            result_data = json.loads(clean_json)

            # validate required fields
            decision = result_data.get("decision")
            confidence = result_data.get("confidence", 0.5)
            reasoning = result_data.get("reasoning", "No reasoning provided")
            selected_recipe_rank = result_data.get("selected_recipe_rank", None)
            ambiguous_recipe_ranks = result_data.get("ambiguous_recipe_ranks", None)

            print(f" [Match Evaluator] Decision: {decision} (confidence: {confidence:.2f})")
            print(f" [Match Evaluator] Reasoning: {reasoning}")

            return MatchEvaluationResult(
                decision=decision,
                selected_recipe_rank=selected_recipe_rank,
                confidence=confidence,
                reasoning=reasoning,
                ambiguous_recipe_ranks=ambiguous_recipe_ranks
            )

        except json.JSONDecodeError as e:
            print(f" [Match Evaluator] JSON parsing error: {e}")
            print(f" [Match Evaluator] Raw content: {raw_content}")
            # fallback to novel_task if parsing fails
            return MatchEvaluationResult(
                decision="NOVEL_TASK",
                selected_recipe_rank=None,
                confidence=0.3,
                reasoning=f"Failed to parse LLM response: {e}",
                ambiguous_recipe_ranks=None
            )

        except Exception as e:
            print(f" [Match Evaluator] Error: {e}")
            # fallback to novel_task on error
            return MatchEvaluationResult(
                decision="NOVEL_TASK",
                selected_recipe_rank=None,
                confidence=0.3,
                reasoning=f"Error during evaluation: {e}",
                ambiguous_recipe_ranks=None
            )

    def _format_candidates(self, candidates: list[CandidateMatch]) -> str:
        # format candidates for llm prompt with enhanced context
        formatted_lines = []

        for candidate in candidates:
            # parse recipe json to extract key info
            try:
                recipe = json.loads(candidate.recipe_json)
                mission_name = recipe.get("mission_name", "Unknown")
                keywords = recipe.get("intent_keywords", [])
                steps = recipe.get("logic_steps", [])

                # classify distance strength
                distance = candidate.distance
                if distance < 1.0:
                    strength = "STRONG match"
                elif distance < 1.3:
                    strength = "MODERATE match"
                else:
                    strength = "WEAK match"

                # format all keywords and steps
                keywords_str = ", ".join(keywords) if keywords else "No keywords"

                # format all steps with numbering
                if steps:
                    steps_formatted = "\n".join([f"      {i}. {step}" for i, step in enumerate(steps, 1)])
                else:
                    steps_formatted = "      No steps"

                formatted_lines.append(
                    f"Candidate {candidate.rank}: {mission_name} (distance: {distance:.4f} = {strength})\n"
                    f"  Intent Keywords: {keywords_str}\n"
                    f"  Recipe Steps ({len(steps)} total):\n{steps_formatted}\n"
                )

            except (json.JSONDecodeError, KeyError) as e:
                # if parsing fails, show raw distance
                formatted_lines.append(
                    f"Candidate {candidate.rank}: Unable to parse (distance: {candidate.distance:.4f})\n"
                )

        return "\n".join(formatted_lines)

    def _clean_json(self, text: str) -> str:
        # extract and clean json from llm response
        # extract json using regex (find first {...})
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            text = match.group(0)

        # remove markdown code blocks
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]

        # replace python literals with json equivalents
        text = text.replace(': None', ': null')
        text = text.replace(':None', ':null')
        text = text.replace(': True', ': true')
        text = text.replace(':True', ':true')
        text = text.replace(': False', ': false')
        text = text.replace(':False', ':false')

        return text.strip()
