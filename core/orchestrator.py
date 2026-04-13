# orchestrator coordinates planning strategies
from typing import Dict
from core.interfaces import PlanningStrategy
from core.knowledge_base import KnowledgeBase
from core.strategies.zero_shot import ZeroShotStrategy
from core.strategies.single_rag import SingleRAGStrategy
from core.strategies.dual_rag.strategy import DualRAGStrategy
from core.domain import RobotMission

class Orchestrator:
    def __init__(self):
        self.knowledge_base = KnowledgeBase()

        self.strategies: Dict[str, PlanningStrategy] = {
            "zero_shot": ZeroShotStrategy(),
            "single_rag": SingleRAGStrategy(self.knowledge_base),
            "dual_rag": DualRAGStrategy(self.knowledge_base)
        }

    def plan_mission(self, user_prompt: str, mode: str = "zero_shot") -> RobotMission:
        if mode not in self.strategies:
            raise ValueError(f"Unknown mode: {mode}. Available modes: {list(self.strategies.keys())}")

        strategy = self.strategies[mode]
        return strategy.generate_mission(user_prompt)
