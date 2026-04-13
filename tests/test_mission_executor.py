import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# add the project root to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.mission_executor import MissionExecutor

class TestMissionExecutor(unittest.TestCase):

    def setUp(self):
        # RoboDK setup
        self.patcher_setup_robodk = patch('core.mission_executor.setup_robodk')
        self.patcher_setup_robot = patch('core.mission_executor.setup_robot')
        
        self.mock_setup_robodk = self.patcher_setup_robodk.start()
        self.mock_setup_robot = self.patcher_setup_robot.start()
        
        # Setup mock returns
        self.mock_rdk = MagicMock()
        self.mock_robot = MagicMock()
        self.mock_tool = MagicMock()
        self.mock_gripper = MagicMock()
        
        self.mock_setup_robodk.return_value = self.mock_rdk
        self.mock_setup_robot.return_value = (self.mock_robot, self.mock_tool, self.mock_gripper)
        
        # ensure patches are stopped after tests
        self.addCleanup(patch.stopall)

        # sample config data
        self.config_data = {
            "settings": {"simulation_speed": 1},
            "tasks": []
        }

    @patch('core.mission_executor.pick_up')
    @patch('core.mission_executor.get_blocking_objects')
    @patch('core.mission_executor.clear_path')
    def test_execute_pick(self, mock_clear_path, mock_get_blocking, mock_pick_up):
        # setup task
        self.config_data["tasks"] = [
            {"type": "pick", "target_obj_name": "test_tube_1"}
        ]
        
        # mock item validation
        mock_item = MagicMock()
        mock_item.Valid.return_value = True
        self.mock_rdk.Item.return_value = mock_item
        
        # mock blocking objects (empty list = no obstacles)
        mock_get_blocking.return_value = []
        
        # initialize executor
        executor = MissionExecutor(config_data=self.config_data)
        executor.execute()
        
        # verify calls
        self.mock_rdk.Item.assert_called_with("test_tube_1")
        mock_pick_up.assert_called_once()

    @patch('core.mission_executor.put_down')
    @patch('core.mission_executor.get_object_height')
    @patch('core.mission_executor.get_corrected_position')
    @patch('core.mission_executor.get_aligned_rotation')
    def test_execute_place(self, mock_get_rot, mock_get_pos, mock_get_height, mock_put_down):
        # setup task
        self.config_data["tasks"] = [
            {
                "type": "place", 
                "destination_obj_name": "bin",
                "destination_coords": {"x": 0, "y": 0}
            }
        ]
        
        # mock item validation
        mock_item = MagicMock()
        mock_item.Valid.return_value = True
        self.mock_rdk.Item.return_value = mock_item
        
        # Mock position return (x, y, z)
        mock_get_pos.return_value = (100, 200, 0)
        
        # Initialize Executor
        executor = MissionExecutor(config_data=self.config_data)
        executor.execute()
        
        # verify calls
        self.mock_rdk.Item.assert_called_with("bin")
        mock_put_down.assert_called_once()

    @patch('core.mission_executor.move_home')
    def test_execute_move_home(self, mock_move_home):
        # setup task
        self.config_data["tasks"] = [
            {"type": "move_home"}
        ]
        
        # initialize executor
        executor = MissionExecutor(config_data=self.config_data)
        executor.execute()
        
        # verify calls
        mock_move_home.assert_called_once()

    @patch('core.mission_executor.put_down_on_free_spot')
    def test_execute_place_free_spot(self, mock_put_down_free):
        # setup task
        self.config_data["tasks"] = [
            {
                "type": "place_free_spot", 
                "destination_obj_name": "None",
                "destination_coords": {"x": 100, "y": 100}
            }
        ]
        
        # mock item validation, destination invalid so use coordinates
        mock_item = MagicMock()
        mock_item.Valid.return_value = False
        self.mock_rdk.Item.return_value = mock_item
        
        # initialize executor
        executor = MissionExecutor(config_data=self.config_data)
        executor.execute()
        
        # verify calls
        mock_put_down_free.assert_called_once()

if __name__ == '__main__':
    unittest.main()