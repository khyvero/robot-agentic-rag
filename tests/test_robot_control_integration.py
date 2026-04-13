import sys
import os
import time

# add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.robot_control import (
    setup_robodk, setup_robot, check_and_clear_gripper,
    pick_up, put_down, put_down_on_free_spot, move_home, pour, shake
)
from config.config import Config

def run_integration_test():
    print(" *** Starting Robot Control Integration Test ***")
    print("Ensure RoboDK is open and the station is loaded.")
    print("Objects required: test_tube_blood1, test_tube_blood2, biohazard_bin")
    
    # setup
    print("\n Setting up RoboDK connection")
    rdk = setup_robodk(speed=1, collision_active=False)
    robot, tool_item, gripper_mech = setup_robot(rdk)
    
    # move home
    print("\n Testing move_home()")
    move_home(robot)
    time.sleep(1)

    # check gripper
    print("\n Testing check_and_clear_gripper()")
    check_and_clear_gripper(rdk, robot, tool_item, gripper_mech)
    
    # 4. standard pick up
    target_name = "test_tube_blood1"
    print(f"\n Testing pick_up('{target_name}')")
    target_obj = rdk.Item(target_name)
    if not target_obj.Valid():
        print(f" Error: {target_name} not found in station.")
        return
        
    pick_up(rdk, target_obj, robot, tool_item, gripper_mech)
    time.sleep(1)
    
    # Test Pour
    print("\n Testing pour()")
    # We are holding target_obj (test_tube_blood1)
    pour(robot, tool_item, target_obj)
    time.sleep(1)
    
    # Test Shake
    print("\n Testing shake()")
    shake(robot, tool_item, target_obj)
    time.sleep(1)
    
    # standard put down, move obj/s on table if stacking
    print("\n Testing put_down() at (0, 300)")
    # drop at x=0, y=300 (arbitrary spot on table)
    put_down(rdk, 0, 300, robot, tool_item, gripper_mech)
    time.sleep(1)

    # pick up
    print(f"\n Picking up '{target_name}' again")
    pick_up(rdk, target_obj, robot, tool_item, gripper_mech)
    time.sleep(1)

    # put down on free spot
    print("\n Testing put_down_on_free_spot() near (0, 300)")
    put_down_on_free_spot(rdk, 0, 300, robot, tool_item, gripper_mech)
    time.sleep(1)
    
    # pick up another object
    target_name_2 = "test_tube_blood2"
    print(f"\n Testing pick_up('{target_name_2}')")
    target_obj_2 = rdk.Item(target_name_2)
    if target_obj_2.Valid():
        pick_up(rdk, target_obj_2, robot, tool_item, gripper_mech)
        
        # put down in bin
        print("\n Testing put_down() into bin")
        bin_obj = rdk.Item("biohazard_bin")
        if bin_obj.Valid():
            # get bin position
            bin_pose = bin_obj.PoseAbs()
            bin_x = bin_pose.Pos()[0]
            bin_y = bin_pose.Pos()[1]
            put_down(rdk, bin_x, bin_y, robot, tool_item, gripper_mech)
        else:
            print("Bin not found, skipping bin placement.")
    else:
        print(f"{target_name_2} not found, skipping second pick test.")

    # final home
    print("\n Returning Home...")
    move_home(robot)

    print("\n *** Robot Control Integration Test Complete ***")

if __name__ == "__main__":
    run_integration_test()