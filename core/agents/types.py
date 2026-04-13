# data classes for multi-agent architecture
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple, Literal


class RouterInput(BaseModel):
    user_prompt: str = Field(..., description="User's natural language prompt")
    valid_objects: List[str] = Field(..., description="List of valid object names in the scene")


class RouterDecision(BaseModel):
    route: Literal["EXACT_MATCH", "AMBIGUOUS", "NOVEL_TASK", "NOT_TASK"] = Field(
        ..., description="Routing decision for the user prompt"
    )
    intent_text: Optional[str] = Field(None, description="Recipe text for EXACT_MATCH")
    distance: Optional[float] = Field(None, description="Semantic distance score")
    candidates: Optional[List[Tuple[str, float]]] = Field(None, description="Candidate matches for AMBIGUOUS route")
    reasoning: Optional[str] = Field(None, description="LLM reasoning for the routing decision")
    confidence: Optional[float] = Field(None, description="Confidence score (0-1) for the decision")

class CandidateMatch(BaseModel):
    recipe_json: str = Field(..., description="Recipe JSON string")
    distance: float = Field(..., description="Semantic distance score")
    rank: int = Field(..., description="Rank in retrieval (1=best, 2=second, etc.)")


class MatchEvaluationInput(BaseModel):
    user_prompt: str = Field(..., description="User's natural language prompt")
    candidates: List[CandidateMatch] = Field(..., description="Top candidate matches from knowledge base")
    scene_objects: List[str] = Field(..., description="Available objects in the scene")


class MatchEvaluationResult(BaseModel):
    decision: Literal["EXACT_MATCH", "AMBIGUOUS", "NOVEL_TASK", "NOT_TASK"] = Field(
        ..., description="Routing decision based on match quality analysis"
    )
    selected_recipe_rank: Optional[int] = Field(None, description="Rank of selected recipe (1-3) for EXACT_MATCH")
    confidence: float = Field(..., description="Confidence score (0-1) for the decision")
    reasoning: str = Field(..., description="Explanation for the routing decision")
    ambiguous_recipe_ranks: Optional[List[int]] = Field(None, description="List of recipe ranks for AMBIGUOUS route")

class ModificationDetectionInput(BaseModel):
    user_prompt: str = Field(..., description="User's natural language prompt")
    matched_recipe: str = Field(..., description="Matched recipe/intent text")


class ModificationDetectionResult(BaseModel):
    has_modification: bool = Field(..., description="Whether user wants to modify the recipe")
    modification_description: Optional[str] = Field(None, description="Description of requested modifications")
    modification_type: Optional[Literal["add", "remove", "replace", "constraint", "parameter"]] = Field(
        None, description="Type of modification detected"
    )

class ActionExtractionInput(BaseModel):
    user_prompt: str = Field(..., description="User's natural language prompt")
    intent_text: Optional[str] = Field(None, description="Recipe steps if Tier 2")
    available_actions: List[Dict[str, Any]] = Field(..., description="Loaded from procedural DB")


class ActionExtractionResult(BaseModel):
    actions: List[str] = Field(..., description="Extracted action names, e.g., ['pick', 'place', 'pour']")
    reasoning: str = Field(..., description="LLM explanation for debugging")

class PlanGenerationInput(BaseModel):
    user_prompt: str = Field(..., description="User's natural language prompt")
    intent_context: Optional[str] = Field(None, description="Recipe steps (Tier 2)")
    procedural_context: str = Field(..., description="Retrieved API documentation")
    mode: Literal["TIER_2_REFINEMENT", "TIER_3_GENERATION"] = Field(
        ..., description="Generation mode"
    )


class PlanGenerationResult(BaseModel):
    plan_json: str = Field(..., description="Clean JSON string")
    success: bool = Field(..., description="Whether plan generation succeeded")
    error: Optional[str] = Field(None, description="Error message if failed")

class ConversationInput(BaseModel):
    conversation_type: Literal["AMBIGUITY", "PLAN_REVIEW", "PLAN_MODIFICATION"] = Field(
        ..., description="Type of conversation"
    )
    user_reply: str = Field(..., description="User's response")
    context: Dict[str, Any] = Field(..., description="Type-specific context")


class ConversationResult(BaseModel):
    mission: Any = Field(..., description="RobotMission object")
    state_transition: Literal["STAY", "RESET_TO_IDLE"] = Field(
        ..., description="FSM state transition"
    )
    next_action: Optional[Literal["EXECUTE", "RETRY_PIPELINE", "GENERATE_PLAN"]] = Field(
        None, description="Next action to take"
    )

    class Config:
        arbitrary_types_allowed = True

class RetrievalInput(BaseModel):
    user_prompt: str = Field(..., description="User's natural language prompt")
    intent_text: Optional[str] = Field(None, description="Recipe text if available")
    extracted_actions: List[str] = Field(..., description="Actions extracted from Action Extractor")
    min_results: int = Field(3, description="Minimum number of APIs to retrieve")
    max_results: int = Field(10, description="Maximum number of APIs to retrieve")


class RetrievalResult(BaseModel):
    procedural_context: str = Field(..., description="Formatted API documentation")
    apis_retrieved: int = Field(..., description="Number of APIs retrieved")