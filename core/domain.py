# robot mission domain model
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from core.mission_executor import MissionExecutor

class RobotMission(BaseModel):
    name: str
    steps: List[Dict[str, Any]]
    settings: Dict[str, Any]
    raw_plan: Optional[str] = None

    def execute(self):
        print(f"Starting Mission: {self.name}")

        if self.steps and all(task.get("type") == "ask_user" for task in self.steps):
            print(" [Info] This is an informational mission. Skipping RoboDK initialization.")
            for task in self.steps:
                question = task.get("question", "Please clarify.")
                print(f"\n[ROBOT]: {question}")
            return

        config_data = {
            "settings": self.settings,
            "tasks": self.steps
        }

        executor = MissionExecutor(config_data=config_data)
        executor.execute()
