# multi-agent architecture orchestrating specialized agents for robot mission planning
# routes prompts, extracts actions, retrieves apis, generates plans, handles dialogue

import json
from enum import Enum, auto

from core.interfaces import PlanningStrategy
from core.domain import RobotMission
from core.knowledge_base import KnowledgeBase

# Import agents
from core.agents.intent_router import IntentRouterAgent
from core.agents.modification_detector import ModificationDetectorAgent
from core.agents.action_extractor import ActionExtractorAgent
from core.agents.plan_generator import PlanGeneratorAgent
from core.agents.conversation_agent import ConversationAgent

# Import services
from core.services.procedural_retrieval import ProceduralRetrievalService

# Import types
from core.agents.types import (
    RouterInput,
    ModificationDetectionInput,
    ActionExtractionInput,
    PlanGenerationInput,
    RetrievalInput,
)

from config.config import Config


# finite state machine definition
class AgentState(Enum):
    IDLE = auto()              # standard listening mode
    AMBIGUITY_CHECK = auto()   # waiting for user to select option
    PLAN_REVIEW = auto()       # waiting for user to confirm/edit plan


class DualRAGStrategy(PlanningStrategy):
    # multi-agent architecture orchestrating intent routing, action extraction, retrieval, generation, conversation

    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base

        # initialize agents
        print(" [DualRAG] Initializing agents...")
        self.intent_router = IntentRouterAgent(knowledge_base)
        self.modification_detector = ModificationDetectorAgent()
        self.action_extractor = ActionExtractorAgent(knowledge_base)
        self.plan_generator = PlanGeneratorAgent()
        self.conversation_agent = ConversationAgent()
        self.procedural_retrieval = ProceduralRetrievalService(knowledge_base)

        # state management
        self.state = AgentState.IDLE

        # memory for ambiguity and plan review states
        self.last_ambiguous_prompt = None
        self.last_candidates = []
        self.pending_plan_json = None

        print(" [DualRAG] Initialization complete")

    def generate_mission(self, user_prompt: str) -> RobotMission:
        # generate robot mission from user prompt using multi-agent architecture
        print(f"\n [DualRAG] State: {self.state.name} | Input: '{user_prompt}'")

        # state routing
        if self.state == AgentState.PLAN_REVIEW:
            return self._handle_plan_review_state(user_prompt)

        elif self.state == AgentState.AMBIGUITY_CHECK:
            return self._handle_ambiguity_state(user_prompt)
        else:  # AgentState.IDLE
            return self._handle_idle_state(user_prompt)

    # state handlers

    def _handle_idle_state(self, user_prompt: str) -> RobotMission:
        # handle idle state: standard pipeline with intent routing
        # intent routing
        router_input = RouterInput(
            user_prompt=user_prompt,
            valid_objects=Config.VALID_OBJECTS
        )

        decision = self.intent_router.route(router_input)

        # route to appropriate tier
        if decision.route == "NOT_TASK":
            return self._create_info_mission()

        elif decision.route == "AMBIGUOUS":
            return self._trigger_ambiguity_dialog(user_prompt, decision.candidates)

        elif decision.route == "EXACT_MATCH":
            return self._execute_tier_2_refinement(user_prompt, decision.intent_text)

        else:  # NOVEL_TASK
            return self._execute_tier_3_generation(user_prompt)

    def _handle_ambiguity_state(self, user_reply: str) -> RobotMission:
        # handle ambiguity_check state: interpret user selection
        result = self.conversation_agent.interpret_selection(
            user_reply=user_reply,
            candidates=self.last_candidates
        )

        # process result
        if result.next_action == "GENERATE_PLAN":
            self.state = AgentState.IDLE
            return self._execute_tier_3_generation(self.last_ambiguous_prompt)

        elif result.next_action == "RETRY_PIPELINE":
            self.state = AgentState.IDLE
            merged_prompt = f"{self.last_ambiguous_prompt} {user_reply}"
            return self.generate_mission(merged_prompt)

        else:  # EXECUTE - user selected existing mission
            self.state = AgentState.IDLE
            # Extract selected mission JSON from candidates
            selected_json = self._extract_selected_json(user_reply, self.last_candidates)
            return self._execute_tier_2_refinement(self.last_ambiguous_prompt, selected_json)

    def _handle_plan_review_state(self, user_input: str) -> RobotMission:
        # handle plan_review state: confirm or modify plan
        # check for confirmation first
        if user_input.lower() in ["yes", "ok", "confirm", "execute", "go", "y"]:
            print("   > User confirmed plan")
            self.state = AgentState.IDLE

            try:
                data = json.loads(self.pending_plan_json)
                raw_json = self.pending_plan_json
                self.pending_plan_json = None
                return RobotMission(
                    name="Dual RAG (Generated)",
                    steps=data.get("tasks", []),
                    settings=data.get("settings", {"simulation_speed": 1}),
                    raw_plan=raw_json
                )
            except json.JSONDecodeError as e:
                print(f" [Plan Review] Error parsing plan: {e}")
                self.pending_plan_json = None
                return RobotMission(
                    name="Error",
                    steps=[],
                    settings={"simulation_speed": 1}
                )

        else:
            # user requested modifications
            print("   > User requested modifications")
            modified_plan = self.conversation_agent.modify_plan(
                current_plan=self.pending_plan_json,
                user_feedback=user_input
            )

            # update pending plan and generate review (stay in plan_review state)
            self.pending_plan_json = modified_plan
            return self.conversation_agent.review_plan(
                plan_json=modified_plan,
                updated=True
            )

    # tier execution methods

    def _execute_tier_2_refinement(
        self,
        user_prompt: str,
        intent_text: str
    ) -> RobotMission:
        # execute tier 2: recipe refinement pipeline
        # checks for modifications, extracts actions, retrieves apis, generates plan
        print(" [Tier 2] Recipe Refinement Pipeline")

        # check for modifications
        modification_input = ModificationDetectionInput(
            user_prompt=user_prompt,
            matched_recipe=intent_text
        )

        modification_result = self.modification_detector.detect_modification(modification_input)

        # extract actions
        action_input = ActionExtractionInput(
            user_prompt=user_prompt,
            intent_text=intent_text,
            available_actions=self.action_extractor.available_actions
        )

        action_result = self.action_extractor.extract_actions(action_input)

        # retrieve procedural apis
        retrieval_input = RetrievalInput(
            user_prompt=user_prompt,
            intent_text=intent_text,
            extracted_actions=action_result.actions,
            min_results=Config.PROCEDURAL_MIN_RESULTS
        )

        retrieval_result = self.procedural_retrieval.retrieve(retrieval_input)

        # generate plan
        plan_input = PlanGenerationInput(
            user_prompt=user_prompt,
            intent_context=intent_text,
            procedural_context=retrieval_result.procedural_context,
            mode="TIER_2_REFINEMENT"
        )

        plan_result = self.plan_generator.generate_plan(plan_input)

        # handle modifications and create mission
        if plan_result.success:
            try:
                data = json.loads(plan_result.plan_json)

                # if modifications detected, enter plan_review for confirmation
                if modification_result.has_modification:
                    print(f" [Tier 2] Modifications detected, entering PLAN_REVIEW")
                    self.pending_plan_json = plan_result.plan_json
                    self.state = AgentState.PLAN_REVIEW

                    # return plan review mission with modification context
                    return RobotMission(
                        name="Plan Review (Modified Recipe)",
                        steps=[{
                            "type": "ask_user",
                            "question": (
                                f"I found a matching recipe but detected you want to make changes:\n\n"
                                f"**Modification**: {modification_result.modification_description}\n\n"
                                f"**Generated Plan** (with modifications):\n"
                                + self._format_plan_for_review(data) +
                                f"\n\nDo you want to proceed with this modified plan? "
                                f"(Say 'yes' to confirm, or describe further changes)"
                            )
                        }],
                        settings={"simulation_speed": 1}
                    )

                # no modifications, execute directly
                return RobotMission(
                    name="Dual RAG (Refined)",
                    steps=data.get("tasks", []),
                    settings=data.get("settings", {"simulation_speed": 1}),
                    raw_plan=plan_result.plan_json
                )
            except json.JSONDecodeError as e:
                print(f" [Tier 2] Error parsing plan: {e}")
                return RobotMission(
                    name="Error",
                    steps=[],
                    settings={"simulation_speed": 1}
                )
        else:
            print(f" [Tier 2] Plan generation failed: {plan_result.error}")
            return RobotMission(
                name="Failed",
                steps=[],
                settings={"simulation_speed": 1}
            )

    def _format_plan_for_review(self, plan_data: dict) -> str:
        # format plan data for human-readable review
        steps = plan_data.get("tasks", [])
        formatted_steps = []

        for i, step in enumerate(steps, 1):
            step_type = step.get("type", "unknown")
            formatted_steps.append(f"  {i}. {step_type.upper()}")

            if step_type == "pick":
                formatted_steps.append(f"     - Object: {step.get('target_obj_name', 'N/A')}")
            elif step_type == "place":
                if "destination_obj_name" in step:
                    formatted_steps.append(f"     - Destination: {step['destination_obj_name']}")
                elif "destination_coords" in step:
                    formatted_steps.append(f"     - Coordinates: {step['destination_coords']}")
            elif step_type == "pour":
                formatted_steps.append(f"     - Target: {step.get('target_container_name', 'N/A')}")
            elif step_type == "shake":
                formatted_steps.append(f"     - Duration: {step.get('duration_seconds', 'N/A')} seconds")
            elif step_type == "wait":
                formatted_steps.append(f"     - Duration: {step.get('duration_seconds', 'N/A')} seconds")

        return "\n".join(formatted_steps)

    def _execute_tier_3_generation(self, user_prompt: str) -> RobotMission:
        # execute tier 3: novel task generation pipeline
        # extracts actions, retrieves apis, generates plan, enters plan_review for human confirmation
        print(" [Tier 3] Novel Task Generation Pipeline")

        # extract actions
        action_input = ActionExtractionInput(
            user_prompt=user_prompt,
            intent_text=None,  # No intent for novel tasks
            available_actions=self.action_extractor.available_actions
        )

        action_result = self.action_extractor.extract_actions(action_input)

        # retrieve procedural apis
        retrieval_input = RetrievalInput(
            user_prompt=user_prompt,
            intent_text=None,
            extracted_actions=action_result.actions,
            min_results=Config.PROCEDURAL_MIN_RESULTS
        )

        retrieval_result = self.procedural_retrieval.retrieve(retrieval_input)

        # generate plan
        plan_input = PlanGenerationInput(
            user_prompt=user_prompt,
            intent_context=None,
            procedural_context=retrieval_result.procedural_context,
            mode="TIER_3_GENERATION"
        )

        plan_result = self.plan_generator.generate_plan(plan_input)

        # store plan and enter review state
        if plan_result.success:
            self.pending_plan_json = plan_result.plan_json
            self.state = AgentState.PLAN_REVIEW

            # generate review mission
            return self.conversation_agent.review_plan(
                plan_json=self.pending_plan_json,
                updated=False
            )
        else:
            print(f" [Tier 3] Plan generation failed: {plan_result.error}")
            return RobotMission(
                name="Error",
                steps=[{"type": "ask_user", "question": f"Plan generation failed: {plan_result.error}"}],
                settings={"simulation_speed": 1}
            )

    # helper methods

    def _trigger_ambiguity_dialog(self, user_prompt: str, candidates) -> RobotMission:
        # transition to ambiguity state and ask clarification question
        print(" [Transition] Entering AMBIGUITY_CHECK state")

        self.state = AgentState.AMBIGUITY_CHECK
        self.last_ambiguous_prompt = user_prompt
        self.last_candidates = candidates

        return self.conversation_agent.generate_clarification(
            user_prompt=user_prompt,
            candidates=candidates
        )

    def _create_info_mission(self) -> RobotMission:
        # create informational mission for non-task inputs
        return RobotMission(
            name="Info",
            steps=[{
                "type": "ask_user",
                "question": (
                    "I'm a laboratory robot assistant. I can help you with tasks like:\n"
                    "- Pick up objects\n"
                    "- Place or move items\n"
                    "- Pour, shake, or swirl contents\n\n"
                    "Please provide a robot task command, such as 'Pick up test_tube_blood' "
                    "or 'Move beaker_water to bin'."
                )
            }],
            settings={"simulation_speed": 1}
        )

    def _extract_selected_json(self, user_reply: str, candidates) -> str:
        # extract json of selected candidate from user reply
        # try to parse option number from user reply
        import re
        match = re.search(r'option\s*(\d+)', user_reply.lower())

        if match:
            option_num = int(match.group(1)) - 1

            if 0 <= option_num < len(candidates):
                return candidates[option_num][0]

        # if no match, return first candidate
        if candidates:
            return candidates[0][0]

        return "{}"