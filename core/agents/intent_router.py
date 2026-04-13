# routes user prompts to appropriate processing tiers using llm-based match quality evaluation

from core.agents.types import RouterInput, RouterDecision, MatchEvaluationInput, CandidateMatch
from core.agents.match_quality_evaluator import MatchQualityEvaluatorAgent
from core.knowledge_base import KnowledgeBase


class IntentRouterAgent:
    # routes user prompts using llm-based match quality evaluation
    # returns exact_match, ambiguous, or novel_task decisions

    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        self.match_evaluator = MatchQualityEvaluatorAgent()

    def route(self, input_data: RouterInput) -> RouterDecision:
        # route prompt to appropriate tier using llm-based evaluation
        print(" [Intent Router] Analyzing user prompt...")

        # get top 3 candidates from knowledge base
        intent_text, distance = self.kb.query_declarative(input_data.user_prompt, n_results=1)
        print(f" [Intent Router] Best Match: '{intent_text[:50]}...' (Distance: {distance:.4f})")

        candidates_raw = self.kb.get_candidates(input_data.user_prompt, n_results=3)

        # convert to candidatematch objects
        candidates = []
        for rank, (recipe_json, dist) in enumerate(candidates_raw, start=1):
            candidates.append(CandidateMatch(
                recipe_json=recipe_json,
                distance=dist,
                rank=rank
            ))

        # use llm to evaluate matches and make routing decision
        print(f" [Intent Router] Using LLM for routing decision (best distance: {candidates[0].distance:.4f})...")

        eval_input = MatchEvaluationInput(
            user_prompt=input_data.user_prompt,
            candidates=candidates,
            scene_objects=input_data.valid_objects
        )
        eval_result = self.match_evaluator.evaluate_matches(eval_input)

        # convert evaluation result to routerdecision
        print(f" [Intent Router] Final Route: {eval_result.decision}")

        if eval_result.decision == "EXACT_MATCH":
            # get the selected recipe by rank
            selected_rank = eval_result.selected_recipe_rank
            if selected_rank and 1 <= selected_rank <= len(candidates):
                selected_recipe = candidates[selected_rank - 1].recipe_json
                selected_distance = candidates[selected_rank - 1].distance
            else:
                # fallback to best match if rank is invalid
                selected_recipe = intent_text
                selected_distance = distance

            return RouterDecision(
                route="EXACT_MATCH",
                intent_text=selected_recipe,
                distance=selected_distance,
                reasoning=eval_result.reasoning,
                confidence=eval_result.confidence
            )

        elif eval_result.decision == "AMBIGUOUS":
            # get ambiguous candidates
            ambiguous_ranks = eval_result.ambiguous_recipe_ranks or [1, 2, 3]
            ambiguous_candidates = []

            for rank in ambiguous_ranks:
                if 1 <= rank <= len(candidates_raw):
                    ambiguous_candidates.append(candidates_raw[rank - 1])

            return RouterDecision(
                route="AMBIGUOUS",
                distance=distance,
                candidates=ambiguous_candidates if ambiguous_candidates else candidates_raw[:3],
                reasoning=eval_result.reasoning,
                confidence=eval_result.confidence
            )

        elif eval_result.decision == "NOVEL_TASK":
            return RouterDecision(
                route="NOVEL_TASK",
                distance=distance,
                reasoning=eval_result.reasoning,
                confidence=eval_result.confidence
            )

        else:
            # if not_task or unknown decision, default to novel_task
            print(f" [Intent Router] Decision '{eval_result.decision}' treated as NOVEL_TASK (thesis mode)")
            return RouterDecision(
                route="NOVEL_TASK",
                distance=distance,
                reasoning=eval_result.reasoning or "Treated as novel task for thesis system",
                confidence=eval_result.confidence
            )
