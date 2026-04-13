import unittest
import sys
import os
import time
import math

# add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from robodk.robomath import transl, rotz
from core.robot_control import (
    setup_robodk, setup_robot, check_and_clear_gripper, move_home,
    pick_up, put_down, put_down_on_free_spot, put_down_in_area, pour, shake, swirl, clear_path,
    get_position, get_aligned_rotation, get_grip_z_offset, get_resolve_destination, is_holding_target,
    get_blocking_objects, get_object_size, is_spot_occupied, find_free_spot_from_area,
    find_free_spot_from_centre, get_smart_drop_z, get_nearby_obstacles
)
from config.config import Config

class TestRobotControl(unittest.TestCase):
    
    def setUp(self):
        # setup robodk before each test
        print(f"\n--- Setting up {self._testMethodName} ---")
        self.rdk = setup_robodk(speed=1, collision_active=False)
        self.robot, self.tool_item, self.gripper_mech = setup_robot(self.rdk)
        
        # Ensure safe start
        # check_and_clear_gripper(self.rdk, self.robot, self.tool_item, self.gripper_mech)
        # move_home(self.rdk, self.robot, self.tool_item, self.gripper_mech)

    def tearDown(self):
        # cleanup after test
        print(f"\n--- Tearing down {self._testMethodName} ---")
        # Optional: Reset scene or just ensure gripper is empty
        # check_and_clear_gripper(self.rdk, self.robot, self.tool_item, self.gripper_mech)
        # move_home(self.rdk, self.robot, self.tool_item, self.gripper_mech)

    def test_get_grip_z_offset(self):
        # test grip offset calculation for tall and short objects
        # tall object (>50) -> top grip (height - 25)
        self.assertEqual(get_grip_z_offset(100), 75)
        self.assertEqual(get_grip_z_offset(51), 26)

        # short object (<=50) -> middle grip (height / 2)
        self.assertEqual(get_grip_z_offset(50), 25)
        self.assertEqual(get_grip_z_offset(10), 5)

    def test_get_resolve_destination(self):
        # test coordinate resolution from dict or object
        # case A: coordinates
        coords = {'x': 100, 'y': 200, 'z': 50}
        x, y, z = get_resolve_destination(None, coords)
        self.assertEqual((x, y, z), (100, 200, 50))

        # case B: object
        target_name = "test_tube_blood1"
        target_obj = self.rdk.Item(target_name)
        if target_obj.Valid():
            tx, ty, tz = get_position(target_obj)
            rx, ry, rz = get_resolve_destination(target_obj, None)
            self.assertEqual((tx, ty, tz), (rx, ry, rz))

    def test_get_blocking_objects(self):
        # test detection of stacked objects blocking target
        # 1. setup: place top object above bottom object
        bottom_name = "test_tube_blood"
        top_name = "test_tube_DNA"

        bottom_obj = self.rdk.Item(bottom_name)
        top_obj = self.rdk.Item(top_name)

        if not bottom_obj.Valid() or not top_obj.Valid():
            self.skipTest("Objects missing for blocking test")

        # move top object physically above bottom
        bx, by, bz = get_position(bottom_obj)
        _, _, b_height = get_object_size(bottom_obj)

        # place 5mm above bottom
        new_pose = transl(bx, by, bz + b_height + 5) * top_obj.Pose().Rotation()
        top_obj.setPoseAbs(new_pose)

        # 2. check blocking
        blocking_list = get_blocking_objects(self.rdk, bottom_obj, self.tool_item)

        # 3. assert
        self.assertTrue(len(blocking_list) > 0, "Should detect blocking object")
        self.assertEqual(blocking_list[0].Name(), top_name, "Should identify the specific blocking item")

    def test_get_nearby_obstacles(self):
        # test radius search for nearby obstacles
        print("\n--- Testing Radius Search ---")

        # setup: move 2 objects close to (300, 300)
        obj1 = self.rdk.Item("test_tube_blood")
        obj2 = self.rdk.Item("test_tube_DNA")

        if not obj1.Valid() or not obj2.Valid(): self.skipTest("Missing objects")

        obj1.setPoseAbs(transl(300, 300, 0))
        obj2.setPoseAbs(transl(310, 310, 0))  # very close

        # search at (300,300) with radius 50 should find both
        obstacles = get_nearby_obstacles(self.rdk, 300, 300, radius=50, ignore_items=[])

        self.assertTrue(len(obstacles) >= 2, "Should find at least the 2 objects we placed")

    def test_find_free_spot_blocked(self):
        # test finding free spot when all sides blocked (hard to automate, needs 4+ objects)
        print("\n--- Testing Blocked Center Search ---")

        center_obj = self.rdk.Item("test_tube_blood")
        target_obj = self.rdk.Item("test_tube_DNA")

        # skipping full implementation - requires 5+ items in station
        pass

    def test_is_spot_occupied(self):
        # test spot occupation detection
        print("\n--- Testing Spot Occupation ---")

        # 1. pick spot with existing object
        target_obj = self.rdk.Item("test_tube_blood")
        if not target_obj.Valid(): self.skipTest("Target missing")

        tx, ty, _ = get_position(target_obj)

        # 2. assert spot is occupied
        is_occ = is_spot_occupied(self.rdk, tx, ty, self.tool_item, ignore_item=None)
        self.assertTrue(is_occ, f"Spot ({tx},{ty}) should be occupied by {target_obj.Name()}")

        # 3. assert far spot is empty
        is_empty = is_spot_occupied(self.rdk, 999, 999, self.tool_item)
        self.assertFalse(is_empty, "Spot (999,999) should be empty")

    def test_smart_stacking(self):
        # test smart z calculation for stacking on existing object
        print("\n--- Testing Smart Stacking ---")

        # 1. setup: place base at (0,0)
        base_name = "test_tube_blood"
        stack_name = "test_tube_DNA"

        base_obj = self.rdk.Item(base_name)
        stack_obj = self.rdk.Item(stack_name)

        if not base_obj.Valid() or not stack_obj.Valid():
            self.skipTest("Missing objects for stacking test")

        # move base to (0,0,0)
        base_obj.setPoseAbs(transl(0, 0, 0))

        # 2. pick up object to stack
        pick_up(self.rdk, stack_obj, self.robot, self.tool_item, self.gripper_mech)

        # 3. calculate drop z at (0,0) - should be > 0 since base is there
        smart_z = get_smart_drop_z(self.rdk, 0, 0, self.tool_item, stack_obj)

        # 4. assert stacking height correct (1mm above base)
        _, _, base_h = get_object_size(base_obj)

        print(f"> Base Height: {base_h}, Calculated Smart Z: {smart_z}")
        self.assertAlmostEqual(smart_z, base_h + 1.0, delta=0.5, msg="Should stack 1mm above base object")

    def test_helper_functions(self):
        # test geometry and detection helper functions
        print("\n--- Testing Helper Functions ---")

        # 1. test get_object_size
        target_name = "test_tube_blood"
        target_obj = self.rdk.Item(target_name)
        if target_obj.Valid():
            x, y, z = get_object_size(target_obj)
            print(f"> Size of {target_name}: {x}x{y}x{z}")
            self.assertGreater(z, 0, "Object height should be positive")

        # 2. test get_aligned_rotation with 90deg rotation
        original_pose = target_obj.Pose()
        target_obj.setPose(original_pose * rotz(math.pi / 2))

        rot_matrix = get_aligned_rotation(target_obj)
        self.assertIsNotNone(rot_matrix)

        # reset pose
        target_obj.setPose(original_pose)

    # robot action tests
    def test_pick_up_success(self):
        # test picking up valid object
        target_name = "test_tube_1"
        target_obj = self.rdk.Item(target_name)

        if not target_obj.Valid():
            self.skipTest("Target object not found")

        pick_up(self.rdk, target_obj, self.robot, self.tool_item, self.gripper_mech)

        # assert holding target
        self.assertTrue(is_holding_target(self.tool_item, target_obj), "Robot failed to grab target")

    def test_validation_wrong_object(self):
        # test validation fails when holding wrong object
        target_name = "test_tube_blood"
        wrong_name = "biohazard_bin"  # Something else in scene

        target_obj = self.rdk.Item(target_name)
        wrong_obj = self.rdk.Item(wrong_name)

        if not target_obj.Valid(): self.skipTest("Target missing")

        # pick up correct object
        pick_up(self.rdk, target_obj, self.robot, self.tool_item, self.gripper_mech)

        # try to pour wrong object - should fail validation
        print("\n> Testing Validation Failure (Expected Error Below):")
        pour(self.robot, self.tool_item, wrong_obj)

        # verify still holding original object
        self.assertTrue(is_holding_target(self.tool_item, target_obj),
                        "Robot should still be holding original object after failed validation")

    def test_move_home(self):
        # test explicitly moving the robot to the home position
        print("\n> Testing Move Home...")

        # call move_home
        move_home(self.rdk, self.robot, self.tool_item, self.gripper_mech)

        # verify joint angles match home position
        current_joints = self.robot.Joints().list()
        expected_joints = [0, -90, -90, 0, 90, 0]

        # check each joint with tolerance (1 degree)
        for i, (curr, exp) in enumerate(zip(current_joints, expected_joints)):
            self.assertAlmostEqual(curr, exp, delta=1.0,
                                   msg=f"Joint {i + 1} is not at home position.")

    def test_put_down_on_object(self):
        # test putting down an object on another object
        target_name = "test_tube_blood"
        target_obj = self.rdk.Item(target_name)

        dest_name = "test_tube_DNA"
        dest_obj = self.rdk.Item(dest_name)

        if not target_obj.Valid():
            self.skipTest(f"{target_name} not found")
        if not dest_obj.Valid():
            self.skipTest(f"{dest_name} not found")

        # pick up object
        pick_up(self.rdk, target_obj, self.robot, self.tool_item, self.gripper_mech)

        # put down on destination
        put_down(self.rdk, self.robot, self.tool_item, self.gripper_mech, destination_obj=dest_obj)

        # verify gripper empty
        attached = self.tool_item.Childs()
        self.assertEqual(len(attached), 0, "Gripper should be empty after put down")

    def test_put_down_at_coords(self):
        # test putting down an object at specific coordinates
        target_name = "test_tube_DNA"
        target_obj = self.rdk.Item(target_name)

        if not target_obj.Valid():
            self.skipTest(f"{target_name} not found")

        # pick up object
        pick_up(self.rdk, target_obj, self.robot, self.tool_item, self.gripper_mech)

        # put down at coordinates
        dest_coords = {'x': 0, 'y': 100}
        put_down(self.rdk, self.robot, self.tool_item, self.gripper_mech, destination_coords=dest_coords)

        # verify gripper empty
        attached = self.tool_item.Childs()
        self.assertEqual(len(attached), 0, "Gripper should be empty after put down")

    def test_put_down_at_specific_height(self):
        # test putting down at a specific z height (e.g. shelf)
        target_name = "test_tube_blood"
        target_obj = self.rdk.Item(target_name)
        if not target_obj.Valid(): self.skipTest("Target missing")

        pick_up(self.rdk, target_obj, self.robot, self.tool_item, self.gripper_mech)

        # put down at z=100mm
        high_coords = {'x': 200, 'y': 300, 'z': 100}

        put_down(self.rdk, self.robot, self.tool_item, self.gripper_mech,
                 target_obj, destination_coords=high_coords)

        # verify z position (allow tolerance for gripper release)
        _, _, final_z = get_position(target_obj)
        self.assertGreaterEqual(final_z, 95, "Object should be placed near Z=100")

    def test_put_down_in_area(self):
        # test placing an object into a restricted custom zone
        print("\n--- Testing Put In Area ---")
        target_name = "test_tube_blood"
        target_obj = self.rdk.Item(target_name)
        if not target_obj.Valid(): self.skipTest("Target missing")

        # pick it up
        pick_up(self.rdk, target_obj, self.robot, self.tool_item, self.gripper_mech)

        # define custom area bounds
        custom_area = {
            'min_x': 300, 'max_x': 400,
            'min_y': 100, 'max_y': 200
        }

        # put down in area
        put_down_in_area(self.rdk, self.robot, self.tool_item, self.gripper_mech, target_obj, custom_area)

        # verify placement within bounds
        new_x, new_y, _ = get_position(target_obj)
        print(f"> Object placed at: ({new_x:.1f}, {new_y:.1f})")

        # small tolerance for floating point errors
        tolerance = 0.1

        self.assertGreaterEqual(new_x, 300 - tolerance)
        self.assertLessEqual(new_x, 400 + tolerance)

        self.assertGreaterEqual(new_y, 100 - tolerance)
        self.assertLessEqual(new_y, 200 + tolerance)

    def test_put_down_area_full(self):
        # test put_down_in_area when the area is too small/full
        print("\n--- Testing Put In Area (Full) ---")
        target_obj = self.rdk.Item("test_tube_blood")
        if not target_obj.Valid(): self.skipTest("Target missing")

        pick_up(self.rdk, target_obj, self.robot, self.tool_item, self.gripper_mech)

        # define impossibly small area (1x1 mm)
        tiny_area = {'min_x': 300, 'max_x': 301, 'min_y': 100, 'max_y': 101}

        # should place at max corner when area full
        put_down_in_area(self.rdk, self.robot, self.tool_item, self.gripper_mech, target_obj, tiny_area)

        new_x, new_y, _ = get_position(target_obj)

        # verify placed at max corner (301, 101)
        self.assertAlmostEqual(new_x, 301, delta=1.0)
        self.assertAlmostEqual(new_y, 101, delta=1.0)

    def test_smart_place_near_obstacle(self):
        # test 'put_down_on_free_spot' avoids collision
        target_name = ("test_tube_DNA")
        obstacle_name = ""

        target_obj = self.rdk.Item(target_name)
        obstacle_obj = self.rdk.Item(obstacle_name)

        if not target_obj.Valid() or not obstacle_obj.Valid():
            self.skipTest("Items missing")

        pick_up(self.rdk, target_obj, self.robot, self.tool_item, self.gripper_mech)

        # try to place exactly where obstacle is - should avoid
        put_down_on_free_spot(self.rdk, self.robot, self.tool_item, self.gripper_mech,
                              target_obj, destination_obj=obstacle_obj)

        # verify object not placed inside obstacle
        tx, ty, _ = get_position(target_obj)
        ox, oy, _ = get_position(obstacle_obj)

        distance = ((tx - ox) ** 2 + (ty - oy) ** 2) ** 0.5
        self.assertGreater(distance, 50, "Object was placed too close to obstacle!")

    def test_put_down_free_spot_near_coords(self):
        # test putting down on a free spot near specific coordinates
        target_name = "test_tube_blood"
        target_obj = self.rdk.Item(target_name)

        if not target_obj.Valid():
            self.skipTest(f"{target_name} not found")

        # pick up object
        pick_up(self.rdk, target_obj, self.robot, self.tool_item, self.gripper_mech)

        # put down on free spot near coordinates
        dest_coords = {'x': 310, 'y': 300}
        print(f"Attempting to place near {dest_coords}")
        put_down_on_free_spot(self.rdk, self.robot, self.tool_item, self.gripper_mech, target_obj,
                              destination_coords=dest_coords)

        # verify gripper empty
        attached = self.tool_item.Childs()
        self.assertEqual(len(attached), 0, "Gripper should be empty after put down")

    def test_shake(self):
        # test shaking action
        target_name = "test_tube_blood"
        target_obj = self.rdk.Item(target_name)
        
        if not target_obj.Valid():
            self.skipTest(f"{target_name} not found")

        # pick up object
        pick_up(self.rdk, target_obj, self.robot, self.tool_item, self.gripper_mech)

        # shake object
        shake(self.robot, self.tool_item, target_obj)

        # verify still holding after shake
        attached = self.tool_item.Childs()
        self.assertTrue(len(attached) > 0, "Should still hold object after shaking")

    def test_swirl(self):
        # test swirling action
        target_name = "test_tube_blood"
        target_obj = self.rdk.Item(target_name)
        
        if not target_obj.Valid():
            self.skipTest(f"{target_name} not found")

        # pick up object
        pick_up(self.rdk, target_obj, self.robot, self.tool_item, self.gripper_mech)

        # swirl object
        swirl(self.robot, self.tool_item, target_obj)

        # verify still holding after swirl
        attached = self.tool_item.Childs()
        self.assertTrue(len(attached) > 0, "Should still hold object after swirling")

    def test_pour_action(self):
        # test pouring from one beaker into another
        source_name = "test_tube_blood"
        dest_name = "test_tube_1"

        source_obj = self.rdk.Item(source_name)
        dest_obj = self.rdk.Item(dest_name)

        if not source_obj.Valid() or not dest_obj.Valid():
            self.skipTest("Source or Destination object missing")

        # pick up source
        pick_up(self.rdk, source_obj, self.robot, self.tool_item, self.gripper_mech)

        # pour into destination
        pour(self.rdk, self.robot, self.tool_item,
             held_obj=source_obj,
             destination_obj=dest_obj)

        # verify didn't drop during pour
        self.assertTrue(is_holding_target(self.tool_item, source_obj),
                        "Robot dropped the beaker while pouring!")

    def test_check_and_clear_gripper(self):
        """
        Test state management:
        1. If holding correct object -> Do nothing, return True.
        2. If holding wrong object -> Clear gripper, return False.
        3. If empty -> Return False.
        """
        print("\n--- Testing Check and Clear Gripper ---")

        obj_a_name = "test_tube_blood"
        obj_b_name = "test_tube_DNA"

        obj_a = self.rdk.Item(obj_a_name)
        obj_b = self.rdk.Item(obj_b_name)

        if not obj_a.Valid() or not obj_b.Valid():
            self.skipTest("Missing objects for gripper state test")

        # scenario 1: holding nothing
        check_and_clear_gripper(self.rdk, self.robot, self.tool_item, self.gripper_mech)

        result = check_and_clear_gripper(self.rdk, self.robot, self.tool_item, self.gripper_mech, target_obj=obj_a)
        self.assertFalse(result, "Should return False when gripper is empty")
        self.assertFalse(is_holding_target(self.tool_item, None), "Gripper should remain empty")

        # scenario 2: holding correct object
        pick_up(self.rdk, obj_a, self.robot, self.tool_item, self.gripper_mech)

        result = check_and_clear_gripper(self.rdk, self.robot, self.tool_item, self.gripper_mech, target_obj=obj_a)
        self.assertTrue(result, "Should return True when already holding the target")
        self.assertTrue(is_holding_target(self.tool_item, obj_a), "Should still be holding Object A")

        # scenario 3: holding wrong object
        print("> Testing wrong object scenario (Should drop Object A)...")
        result = check_and_clear_gripper(self.rdk, self.robot, self.tool_item, self.gripper_mech, target_obj=obj_b)

        self.assertFalse(result, "Should return False because we were holding the wrong object")

        # verify object a was dropped (gripper now empty)
        children = self.tool_item.Childs()
        self.assertEqual(len(children), 0, "Gripper should be empty (Object A cleared)")

    def test_clear_path_execution(self):
        # test full clear_path sequence (moves obstacle to storage)
        print("\n--- Testing Direct Clear Path Execution ---")

        obstacle_name = "test_tube_blood"
        obstacle = self.rdk.Item(obstacle_name)

        if not obstacle.Valid():
            self.skipTest("Obstacle object missing")

        # move to random spot first to ensure it moves
        obstacle.setPoseAbs(transl(400, 0, 0))

        # call clear_path (expects list of objects)
        print(f"> clearing path for: {obstacle_name}")
        clear_path(self.rdk, [obstacle], self.robot, self.tool_item, self.gripper_mech)

        # verify moved to storage area
        new_x, new_y, _ = get_position(obstacle)

        # check against storage area bounds
        area = Config.STORAGE_AREA
        min_x, max_x = area["min_x"], area["max_x"]
        min_y, max_y = area["min_y"], area["max_y"]

        print(f"> Obstacle relocated to: ({new_x:.1f}, {new_y:.1f})")
        print(f"> Storage Bounds: X[{min_x}:{max_x}], Y[{min_y}:{max_y}]")

        # allow tolerance for placement jitter
        self.assertGreaterEqual(new_x, min_x - 10)
        self.assertLessEqual(new_x, max_x + 10)

        # verify y is within bounds
        self.assertGreaterEqual(new_y, min_y - 10)
        self.assertLessEqual(new_y, max_y + 10)

if __name__ == '__main__':
    unittest.main()