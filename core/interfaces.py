# planning strategy interface
from abc import ABC, abstractmethod
from core.domain import RobotMission

class PlanningStrategy(ABC):
    @abstractmethod
    def generate_mission(self, user_prompt: str) -> RobotMission:
        pass
