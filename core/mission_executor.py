# executes robot tasks from config
from typing import Dict, Any, List
from robodk.robolink import *
from config.loader import load_config
from config.config import Config
from core.robot_control import (
    setup_robodk, setup_robot, check_and_clear_gripper,
    get_blocking_objects, clear_path, pick_up,
    put_down, put_down_on_free_spot, put_down_in_area,
    get_object_size, get_position, get_aligned_rotation,
    move_home, pour, shake, swirl, get_held_obj, wait_task
)

class MissionExecutor:
    def __init__(self, config_data: Dict[str, Any] = None, config_path: str = "config/config.json"):
        if config_data:
            self.data = config_data
        else:
            self.data = load_config(config_path)

        self.settings = self.data.get("settings", {})
        self.tasks = self.data.get("tasks", [])

        if not self.tasks:
            print("Error: No 'tasks' found in configuration.")

        speed = self.settings.get("simulation_speed", 1)
        self.rdk = setup_robodk(speed=speed, collision_active=Config.COLLISION_ACTIVE)
        self.robot, self.tool_item, self.gripper_mech = setup_robot(self.rdk)

        self.task_handlers = {
            "pick": self.execute_pick,
            "place": self.execute_place,
            "place_free_spot": self.execute_place_free_spot,
            "place_in_area": self.execute_place_in_area,
            "move_home": self.execute_move_home,
            "ensure_gripper_empty": self.execute_ensure_gripper_empty,
            "pour": self.execute_pour,
            "shake": self.execute_shake,
            "swirl": self.execute_swirl,
            "wait": self.execute_wait,
            "ask_user": self.execute_ask_user
        }

    def execute(self):
        if not self.tasks:
            print("No tasks to execute.")
            return

        total_tasks = len(self.tasks)
        for i, task in enumerate(self.tasks):
            task_type = task.get("type")
            print(f"\n *** Executing Task {i+1}/{total_tasks}: {task_type} ***")

            handler = self.task_handlers.get(task_type)
            if handler:
                try:
                    handler(task)
                except Exception as e:
                    print(f"Error executing task '{task_type}': {e}")
            else:
                print(f"Unknown Task Type: {task_type}")

        is_interactive = (
                self.tasks and self.tasks[-1].get("type") == "ask_user"
        )

        if not is_interactive:
            print("\nAll missions accomplished.\n")

    def execute_pick(self, task: Dict[str, Any]):
        target_obj_name = task.get("target_obj_name")
        if not target_obj_name:
            print("Error: 'pick' task missing 'target_obj_name'")
            return

        target_obj = self.rdk.Item(target_obj_name)
        if not target_obj.Valid():
            print(f"Error: Target object '{target_obj_name}' not found.")
            return

        # Obstacle check
        obstacles = get_blocking_objects(self.rdk, target_obj, self.tool_item)
        if len(obstacles) > 0:
            print(f"\n obstacles detected on {target_obj.Name()}")
            clear_path(self.rdk, obstacles, self.robot, self.tool_item, self.gripper_mech)

        pick_up(self.rdk, target_obj, self.robot, self.tool_item, self.gripper_mech)

    def execute_place(self, task: Dict[str, Any]):
        destination_obj_name = task.get("destination_obj_name")
        destination_coords = task.get("destination_coords")
        
        destination_obj = None
        if destination_obj_name:
            destination_obj = self.rdk.Item(destination_obj_name)
            if not destination_obj.Valid():
                destination_obj = None # Fallback to coords if invalid

        # Identify what we are holding to pass as target_obj
        held_obj = get_held_obj(self.tool_item)

        # Pass both, put_down handles priority
        put_down(self.rdk, self.robot, self.tool_item, self.gripper_mech, 
                 target_obj=held_obj,
                 destination_obj=destination_obj, 
                 destination_coords=destination_coords)

    def execute_place_free_spot(self, task: Dict[str, Any]):
        destination_obj_name = task.get("destination_obj_name")
        destination_coords = task.get("destination_coords")
        
        destination_obj = None
        if destination_obj_name:
            destination_obj = self.rdk.Item(destination_obj_name)
            if not destination_obj.Valid():
                destination_obj = None

        # Identify what we are holding to pass as target_obj
        held_obj = get_held_obj(self.tool_item)
        
        if not held_obj or not held_obj.Valid():
            print("Error: Cannot place on free spot. Gripper is empty.")
            return

        # Pass both, put_down_on_free_spot handles priority
        put_down_on_free_spot(self.rdk, self.robot, self.tool_item, self.gripper_mech, 
                              target_obj=held_obj,
                              destination_obj=destination_obj, 
                              destination_coords=destination_coords)

    def execute_place_in_area(self, task: Dict[str, Any]):
        # "area_bounds" should be a dict like {'min_x': 300, 'max_x': 400, 'min_y': 100, 'max_y': 200}
        area_bounds = task.get("area_bounds")

        if not area_bounds:
            print("Error: 'place_in_area' task requires 'area_bounds' dictionary.")
            return

        # Identify what we are holding
        held_obj = get_held_obj(self.tool_item)

        if not held_obj or not held_obj.Valid():
            print("Error: Cannot place in area. Gripper is empty.")
            return

        # Execute
        put_down_in_area(self.rdk, self.robot, self.tool_item, self.gripper_mech, target_obj=held_obj, area_bounds=area_bounds)

    def execute_move_home(self, task: Dict[str, Any] = None):
        """Moves robot home. Accepts task dict for compatibility but ignores it."""
        move_home(self.rdk, self.robot, self.tool_item, self.gripper_mech)

    def execute_ensure_gripper_empty(self, task: Dict[str, Any] = None):
        """Ensures gripper is empty. Accepts task dict for compatibility."""
        check_and_clear_gripper(self.rdk, self.robot, self.tool_item, self.gripper_mech)

    def execute_pour(self, task: Dict[str, Any]):
        # The target_obj_name in the task refers to the destination container (e.g. beaker)
        destination_obj_name = task.get("target_obj_name")
        if not destination_obj_name:
            print("Error: 'pour' task requires 'target_obj_name' (destination).")
            return
            
        destination_obj = self.rdk.Item(destination_obj_name)
        if not destination_obj.Valid():
            print(f"Error: Destination object '{destination_obj_name}' not found.")
            return
        
        # Identify what we are holding (the source bottle/tube)
        held_obj = get_held_obj(self.tool_item)
        if not held_obj or not held_obj.Valid():
            print("Error: Cannot pour. Gripper is empty.")
            return
            
        pour(self.rdk, self.robot, self.tool_item, held_obj, destination_obj)

    def execute_shake(self, task: Dict[str, Any]):
        target_obj_name = task.get("target_obj_name")
        if not target_obj_name:
            print("Error: 'shake' task requires 'target_obj_name'.")
            return
            
        target_obj = self.rdk.Item(target_obj_name)
        if not target_obj.Valid():
            print(f"Error: Target object '{target_obj_name}' not found.")
            return
            
        shake(self.robot, self.tool_item, target_obj)

    def execute_swirl(self, task: Dict[str, Any]):
        target_obj_name = task.get("target_obj_name")
        if not target_obj_name:
            print("Error: 'swirl' task requires 'target_obj_name'.")
            return
            
        target_obj = self.rdk.Item(target_obj_name)
        if not target_obj.Valid():
            print(f"Error: Target object '{target_obj_name}' not found.")
            return
        
        # Extract optional parameters
        count = task.get("count", 5)
        radius = task.get("radius", 20)
        speed = task.get("speed", 1000)
            
        swirl(self.robot, self.tool_item, target_obj, count=count, radius=radius, speed=speed)

    def execute_wait(self, task: Dict[str, Any]):
        raw_seconds = task.get("seconds", 0)
        try:
            seconds = float(raw_seconds)
        except (ValueError, TypeError):
            print(f"Error: Invalid wait time '{raw_seconds}'. Skipping.")
            return

        if seconds <= 0:
            print("Warning: Wait time is 0 or invalid. Skipping.")
            return

        wait_task(seconds)

    def execute_ask_user(self, task: Dict[str, Any]):
        question = task.get("question", "Please clarify.")
        print(f"\n[ROBOT]: {question}")
        print("(System is waiting for your response to refine the plan...)")