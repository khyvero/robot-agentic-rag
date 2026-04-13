"""
Multi-Agent Architecture for Dual-RAG Strategy

This package contains specialized agents for the robot mission planning system:
- IntentRouter: Routes user prompts to appropriate processing tiers
- ModificationDetector: Detects if user wants to modify a matched recipe
- ActionExtractor: Extracts required robot actions from prompts using LLM
- PlanGenerator: Generates and validates JSON mission plans
- ConversationAgent: Handles multi-turn dialogue (ambiguity, plan review)
"""

from core.agents.types import (
    RouterInput,
    RouterDecision,
    ModificationDetectionInput,
    ModificationDetectionResult,
    ActionExtractionInput,
    ActionExtractionResult,
    PlanGenerationInput,
    PlanGenerationResult,
    ConversationInput,
    ConversationResult,
)

__all__ = [
    "RouterInput",
    "RouterDecision",
    "ModificationDetectionInput",
    "ModificationDetectionResult",
    "ActionExtractionInput",
    "ActionExtractionResult",
    "PlanGenerationInput",
    "PlanGenerationResult",
    "ConversationInput",
    "ConversationResult",
]