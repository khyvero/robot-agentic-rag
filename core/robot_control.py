# robot control functions for robodk
from robodk.robolink import *
from robodk.robomath import *
import time
import math
from config.config import Config


def setup_robodk(speed=1, collision_active=False):
    rdk = Robolink()
    rdk.setRunMode(RUNMODE_SIMULATE)
    rdk.setSimulationSpeed(speed)

    if collision_active:
        rdk.setCollisionActive(COLLISION_ON)
        print(f"> Connected to RoboDK (Speed: {speed}x, Collision: ON)")
    else:
        rdk.setCollisionActive(COLLISION_OFF)
        print(f"> Connected to RoboDK (Speed: {speed}x, Collision: OFF)")

    return rdk


def setup_robot(rdk):
    robot = rdk.Item(Config.ROBOT)
    gripper_mech = rdk.Item(Config.GRIPPER)
    tool_item = rdk.Item(Config.TOOL)

    if not robot.Valid():
        print(f"* ERROR: Robot '{Config.ROBOT}' not found.")
        quit()

    robot.setTool(tool_item)
    print(f"> Robot '{Config.ROBOT}' and gripper '{Config.GRIPPER}' setup complete.")

    tool_item.setPoseTool(transl(0, 0, 160))

    return robot, tool_item, gripper_mech


def get_down_orientation():
    return Mat([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])


def get_object_size(obj):
    try:
        bbox = obj.BoundingBox()
        size_x = bbox[3] - bbox[0]
        size_y = bbox[4] - bbox[1]
        size_z = bbox[5] - bbox[2]
        print(f"> Size of {obj.Name()}: {size_x:.0f}x{size_y:.0f}x{size_z:.0f}mm")
        return size_x, size_y, size_z

    except AttributeError:
        print(f"> RoboDK version is old, BoundingBox is not working")
        if "test_tube" in obj.Name().lower():
            return 50.0, 50.0, 50.0
        elif "dropper" in obj.Name().lower():
            return 50.0, 50.0, 100.0
        elif "reagent_vial" in obj.Name().lower():
            return 50.0, 50.0, 50.0
        elif "beaker" in obj.Name().lower():
            return 50.0, 50.0, 150.0
        elif "bin" in obj.Name().lower():
            return 220.0, 330.0, 10.0
        elif "bunsen_burner" in obj.Name().lower():
            return 100.0, 100.0, 100.0
        else:
            return 0.0, 0.0, 25.0


def get_position(obj):
    wx, wy, wz = obj.PoseAbs().Pos()
    return wx, wy, wz


def get_aligned_rotation(obj):
    obj_angles = Pose_2_TxyzRxyz(obj.PoseAbs())
    rot_z_rad = obj_angles[5]
    rot_z_deg = rot_z_rad * 180 / math.pi
    print(f"> Aligning Gripper to {rot_z_deg:.1f} degrees")

    base_orient = get_down_orientation()
    rot_matrix = rotz(rot_z_rad)

    return rot_matrix * base_orient


def set_gripper(status, gripper_mech, tool_item, obj_to_grab=None):
    if status == "open":
        print("Action: Opening Gripper")
        gripper_mech.setJoints([0])
        tool_item.DetachAll()
    elif status == "close":
        print("Action: Closing Gripper")
        gripper_mech.setJoints([0.85])

        if obj_to_grab is not None and obj_to_grab.Valid():
            print(f"> Force Attaching: {obj_to_grab.Name()}")
            obj_to_grab.setParentStatic(tool_item)
        else:
            tool_item.AttachClosest()

def get_grip_z_offset(height):
    if height > 50:
        return height - 25
    return height / 2

def is_holding_target(tool_item, target_obj):
    held_obj = get_held_obj(tool_item)

    if not held_obj:
        print("> Error: Gripper is empty. Cannot proceed.")
        return False

    if target_obj and held_obj.Name() != target_obj.Name():
        print(f"> Error: Gripper is holding '{held_obj.Name()}', but expected '{target_obj.Name()}'. Aborting.")
        return False

    return True

def get_held_obj(tool_item):
    print("\n> Checking gripper status...")
    attached_objects = tool_item.Childs()
    if len(attached_objects) > 0:
        held_obj = attached_objects[0]
        print(f"> Gripper is holding: {held_obj.Name()}")
        return held_obj
    else:
        print("> Gripper is empty.")
        return None

def held_obj_valid(obj):
    return obj is not None and obj.Valid()


def clear_gripper(rdk, robot, tool_item, gripper_mech):
    print("Action: Clearing gripper...")

    held_obj = get_held_obj(tool_item)

    if not held_obj:
        print("> Gripper is already empty. Nothing to clear.")
        return

    current_x, current_y, _ = robot.Pose().Pos()
    print(f"Action: Landing at current location: X={current_x:.1f}, Y={current_y:.1f}")

    put_down_on_free_spot(rdk, robot, tool_item, gripper_mech, target_obj=held_obj, destination_coords={'x': current_x, 'y': current_y})

def get_blocking_objects(rdk, target_obj, tool_item):
    print(f"\n> Checking for obstacles on top of {target_obj.Name()}")

    tx, ty, tz = get_position(target_obj)
    blocking_list = []

    all_objects = rdk.ItemList(ITEM_TYPE_OBJECT)

    for obj in all_objects:
        if obj.Name() == target_obj.Name(): continue
        if obj.Name() == tool_item.Name(): continue

        ox, oy, oz = get_position(obj)

        distance_xy = math.sqrt((tx - ox) ** 2 + (ty - oy) ** 2)

        if distance_xy < 50 and oz > tz:
            print(f"> Found Blocking Object: {obj.Name()} (Z={oz:.1f})")
            blocking_list.append(obj)

    blocking_list.sort(key=lambda x: get_position(x)[2], reverse=True)

    return blocking_list


def get_nearby_obstacles(rdk, x, y, radius, ignore_items):
    obstacles = []
    ignore_names = [item.Name() for item in ignore_items if item and item.Valid()]

    for obj in rdk.ItemList(ITEM_TYPE_OBJECT):
        if obj.Name() in ignore_names: continue

        ox, oy, _ = get_position(obj)
        if math.sqrt((x - ox) ** 2 + (y - oy) ** 2) < radius:
            obstacles.append(obj)
    return obstacles


def get_biggest_object(objects):
    biggest_obj = None
    max_vol = -1
    for obj in objects:
        sx, sy, sz = get_object_size(obj)
        vol = sx * sy * sz
        if vol > max_vol:
            max_vol = vol
            biggest_obj = obj
    print(f"> Biggest object: {biggest_obj.Name()}")
    return biggest_obj


def find_free_spot_from_area(rdk, tool_item, min_point=None, max_point=None, held_object=None):
    print("\n> Scanning Storage Area for free space...")

    if min_point is not None and max_point is not None:
        min_x, min_y = min_point
        max_x, max_y = max_point
    else:
        area = Config.STORAGE_AREA
        min_x = area["min_x"]
        max_x = area["max_x"]
        min_y = area["min_y"]
        max_y = area["max_y"]

        # 2. Determine Step Size
        step = Config.SHIFT_STEP
        if held_object and held_object.Valid():
            sx, sy, _ = get_object_size(held_object)
            step = max(sx, sy) + 30
            print(f"> Calculated dynamic step size: {step:.0f}mm")
        else:
            print(f"> Using default step size: {step}mm")

        # 3. Iterate through the grid
        # converting range inputs to integers is safer
        for x in range(int(min_x), int(max_x) + 1, int(step)):
            for y in range(int(min_y), int(max_y) + 1, int(step)):
                # Use the unified sensor 'is_spot_occupied'
                if not is_spot_occupied(rdk, x, y, tool_item, ignore_item=held_object):
                    print(f" > Found empty storage spot at ({x}, {y}).")
                    return x, y

    # Fallback
    print("> Warning: Storage Area Full! Returning last corner.")
    return max_x, max_y


def find_free_spot_from_centre(rdk, tool_item, ref_obj, target_obj):
    bx, by, _ = get_position(ref_obj)
    bw, bd, _ = get_object_size(ref_obj)
    hw, hd, _ = get_object_size(target_obj)
    gap = 30  # mm

    # Define the 4 candidate spots
    candidates = [
        (bx + bw / 2 + hw / 2 + gap, by),  # Right
        (bx - bw / 2 - hw / 2 - gap, by),  # Left
        (bx, by + bd / 2 + hd / 2 + gap),  # Back
        (bx, by - bd / 2 - hd / 2 - gap)  # Front
    ]

    for cx, cy in candidates:
        if not is_spot_occupied(rdk, cx, cy, tool_item, ignore_item=target_obj):
            print(f"> Found spot next to {ref_obj.Name()} at ({cx:.1f}, {cy:.1f})")
            return True, (cx, cy)

    return False, None


def get_smart_drop_z(rdk, target_x, target_y, tool_item, target_object=None, verbose=True):
    if verbose:
        print(f"> Scanning Drop Zone ({target_x:.1f}, {target_y:.1f})")

    # start with the table level
    highest_z_found = Config.TABLE_Z

    # get a list of all objects in the station
    all_objects = rdk.ItemList(ITEM_TYPE_OBJECT)

    obstacle_found = False

    for obj in all_objects:
        # don't check the robot tool itself
        if obj.Name() == tool_item.Name():
            continue

        # ignore the currently held object
        if target_object is not None and obj.Name() == target_object.Name():
            continue

        # check distance to this specific item
        obj_x, obj_y, obj_z = obj.PoseAbs().Pos()

        # Get object size to determine collision boundary
        size_x, size_y, size_z = get_object_size(obj)

        # Calculate bounds (assuming center pivot)
        min_x = obj_x - size_x / 2
        max_x = obj_x + size_x / 2
        min_y = obj_y - size_y / 2
        max_y = obj_y + size_y / 2

        # Check if target is within bounds (with some margin)
        margin = 10  # mm
        is_inside_x = (min_x - margin) <= target_x <= (max_x + margin)
        is_inside_y = (min_y - margin) <= target_y <= (max_y + margin)

        if is_inside_x and is_inside_y:
            if verbose:
                print(f"> Found object '{obj.Name()}' at drop zone (Size: {size_x:.0f}x{size_y:.0f}).")
            obstacle_found = True

            # calculate top of this obstacle
            top_of_object = obj_z + size_z

            if top_of_object > highest_z_found:
                highest_z_found = top_of_object

    if obstacle_found:
        if verbose: print(f"> Stack detected. Landing at Z={highest_z_found:.1f} + 1mm")
        return highest_z_found + 1.0
    else:
        if verbose: print("> No Obstacles found. Placing on Table (Z = 0).")
        return Config.TABLE_Z


def get_resolve_destination(destination_obj, destination_coords):
    if destination_obj and destination_obj.Valid():
        print(f"> Destination Object specified: {destination_obj.Name()}")
        # get_position already returns (x, y, z)
        dest_x, dest_y, dest_z = get_position(destination_obj)
        return dest_x, dest_y, dest_z
    elif destination_coords:
        dest_x = destination_coords.get('x', 0)
        dest_y = destination_coords.get('y', 0)
        dest_z = destination_coords.get('z', 0)
        print(f"> Destination Coords specified: ({dest_x}, {dest_y}, {dest_z})")
        return dest_x, dest_y, dest_z
    else:
        print("> Error: No destination specified.")
        return None, None, None


def get_true_center(obj):
    try:
        bbox = obj.BoundingBox()
        center_x = (bbox[0] + bbox[3]) / 2
        center_y = (bbox[1] + bbox[4]) / 2
        top_z = bbox[5]
        return center_x, center_y, top_z

    except AttributeError:
        print(f"> Warning: BoundingBox not found for {obj.Name()}. Using position fallback.")

        x, y, z = get_position(obj)
        _, _, height = get_object_size(obj)

        top_z = z + height
        return x, y, top_z


def is_spot_occupied(rdk, check_x, check_y, tool_item, ignore_item=None):
    all_objects = rdk.ItemList(ITEM_TYPE_OBJECT)

    for obj in all_objects:
        # ignore tool and the currently held object
        if obj.Name() == tool_item.Name():
            continue
        if ignore_item is not None and obj.Name() == ignore_item.Name():
            continue

        ox, oy, _ = get_position(obj)

        # Get object size to determine collision boundary
        size_x, size_y, _ = get_object_size(obj)

        # Calculate bounds (assuming center pivot)
        min_x = ox - size_x / 2
        max_x = ox + size_x / 2
        min_y = oy - size_y / 2
        max_y = oy + size_y / 2

        # Check if target is within bounds (with some margin)
        margin = 10  # mm
        is_inside_x = (min_x - margin) <= check_x <= (max_x + margin)
        is_inside_y = (min_y - margin) <= check_y <= (max_y + margin)

        if is_inside_x and is_inside_y:
            print(f"> {obj.Name()} is at this spot (Size: {size_x:.0f}x{size_y:.0f}).")
            return True

    return False


def move_to_safe_mixing_pose(robot):
    safe_joints = [0, -90, -90, -90, 90, 0]

    print(f"> Moving to Safe Mixing Position (Joints: {safe_joints})...")

    robot.setSpeed(100)

    try:
        robot.MoveJ(safe_joints)
        robot.WaitMove()
        return True
    except Exception as e:
        print(f"> Error moving to safe mixing position: {e}")
        return False


def wait_task(seconds):
    print(f"\n *** Waiting for {seconds} seconds... ***")
    time.sleep(seconds)
    print("Wait Complete.")

def check_and_clear_gripper(rdk, robot, tool_item, gripper_mech, target_obj=None):
    held_obj = get_held_obj(tool_item)

    if held_obj:
        if target_obj and held_obj.Name() == target_obj.Name():
            print(f"> Already holding target object '{target_obj.Name()}'. Skipping pick up.")
            return True

        clear_gripper(rdk, robot, tool_item, gripper_mech)

    return False


def move_home(rdk, robot, tool_item, gripper_mech):
    print("\n *** Moving to Home Position ***")

    check_and_clear_gripper(rdk, robot, tool_item, gripper_mech)

    home_joints = [0, -90, -90, 0, 90, 0]
    robot.MoveJ(home_joints)
    robot.WaitMove()
    print("Home Position Reached.")


def clear_path(rdk, target_objs, robot, tool_item, gripper_mech):
    print(f"Moving {len(target_objs)} object/s to storage spots")

    for obj in target_objs:
        print(f"\n> Relocating object: {obj.Name()}")

        area = Config.STORAGE_AREA
        min_point = area["min_x"], area["min_y"]
        max_point = area["max_x"], area["max_y"]
        store_x, store_y = find_free_spot_from_area(rdk, tool_item, min_point, max_point, obj)

        pick_up(rdk, obj, robot, tool_item, gripper_mech)
        put_down(rdk, robot, tool_item, gripper_mech, destination_coords={'x': store_x, 'y': store_y})

    print("Path cleared. Resuming mission")


def pick_up(rdk, target_obj, robot, tool_item, gripper_mech):
    print("\n *** Starting Pick Up ***")
    already_holding = check_and_clear_gripper(rdk, robot, tool_item, gripper_mech, target_obj=target_obj)

    if already_holding:
        print(f"> Already holding {target_obj}. Skipping pick up sequence.")
        return

    _, _, height = get_object_size(target_obj)

    wx, wy, wz = get_position(target_obj)

    final_orient = get_aligned_rotation(target_obj)

    print(f"> Target Position: X={wx:.1f}, Y={wy:.1f}, Z={wz:.1f}")

    grab_z_offset = get_grip_z_offset(height)
    print(f"> Object Height: {height:.0f}mm -> Grip Offset: {grab_z_offset:.1f}mm")

    grip_pose = transl(wx, wy, wz + grab_z_offset) * final_orient
    approach_pose = grip_pose * transl(0, 0, -100)

    print(f"> Gripping Position: X={wx:.1f}, Y={wy:.1f}, Z={wz + grab_z_offset:.1f}")

    if robot.SolveIK(approach_pose).list() == []:
        print("Error: The target is out of reach!")
        print("> Please move the target object closer to the robot.")
        quit()

    print("Action: Moving to approach")
    robot.MoveJ(approach_pose)
    robot.WaitMove()

    set_gripper("open", gripper_mech, tool_item)
    time.sleep(0.5)

    print("Action: Descending")
    robot.MoveL(grip_pose)
    robot.WaitMove()

    print("Action: Grabbing")
    set_gripper("close", gripper_mech, tool_item, target_obj)
    time.sleep(1.0)

    print("Action: Lifting")
    lift_pose = grip_pose * transl(0, 0, -Config.LIFT_Z)
    robot.MoveL(lift_pose)
    robot.WaitMove()
    print("Pick Up Complete.")


def put_down(rdk, robot, tool_item, gripper_mech, target_obj=None, destination_obj=None, destination_coords=None, drop_orientation=None):
    print("\n *** Starting Put Down ***")

    if not is_holding_target(tool_item, target_obj):
        return

    dest_x, dest_y, dest_z = get_resolve_destination(destination_obj, destination_coords)

    if dest_x is None:
        print("> No valid destination found. Defaulting to (0,0,0).")
        dest_x, dest_y, dest_z = 0, 0, 0

    if destination_obj and destination_obj.Valid() and not drop_orientation:
        drop_orientation = get_aligned_rotation(destination_obj)

    current_held_object = get_held_obj(tool_item)
    grip_z_offset = 0

    if current_held_object and current_held_object.Valid():
        _, _, height = get_object_size(current_held_object)
        grip_z_offset = get_grip_z_offset(height)
        print(f"> Object Height: {height:.0f}mm -> Hold Offset: {grip_z_offset:.1f}mm")
    else:
        print("> Warning: Gripper seems empty. Proceeding with 0 offset.")

    smart_z = get_smart_drop_z(rdk, dest_x, dest_y, tool_item, current_held_object)

    destination_z = max(smart_z, dest_z)

    final_drop_z = destination_z + grip_z_offset

    print(f"> Drop at: X={dest_x:.1f}, Y={dest_y:.1f}, Z={final_drop_z:.1f} (Base Z={destination_z:.1f})")

    target_orient = drop_orientation if drop_orientation else get_down_orientation()

    drop_pose = transl(dest_x, dest_y, final_drop_z) * target_orient
    approach_pose = drop_pose * transl(0, 0, -100)

    print("Action: Moving to drop approach")
    robot.MoveJ(approach_pose)
    robot.WaitMove()

    print("Action: Descending to Table")
    robot.MoveL(drop_pose)
    robot.WaitMove()

    print("Action: Releasing Object")
    set_gripper("open", gripper_mech, tool_item)
    time.sleep(1.0)

    print("Action: Retracting")
    robot.MoveL(approach_pose)
    robot.WaitMove()
    print("Put Down Complete.")


def put_down_on_free_spot(rdk, robot, tool_item, gripper_mech, target_obj, destination_obj=None, destination_coords=None, drop_orientation=None):
    print("\n *** Starting Put Down on Free Spot ***")

    if not is_holding_target(tool_item, target_obj):
        return

    dest_x, dest_y, dest_z = get_resolve_destination(destination_obj, destination_coords)
    if dest_x is None: return

    obstacles = get_nearby_obstacles(rdk, dest_x, dest_y, radius=150, ignore_items=[tool_item, target_obj])

    final_x, final_y = dest_x, dest_y

    if obstacles:
        print(f"> Found {len(obstacles)} obstacle(s). Adjusting placement...")

        biggest_obj = get_biggest_object(obstacles)
        print(f"> Biggest obstacle: {biggest_obj.Name()}")

        spot_found, (adj_x, adj_y) = find_free_spot_from_centre(rdk, tool_item, biggest_obj, target_obj)

        if spot_found:
            final_x, final_y = adj_x, adj_y
        else:
            print("> Warning: Immediate neighbors blocked. Spiraling outward...")
            bx, by, _ = get_position(biggest_obj)
            final_x, final_y = find_free_spot_from_centre(rdk, bx, by, tool_item, target_obj)

    else:
        print("> No major obstacles. Checking exact point...")
        if is_spot_occupied(rdk, dest_x, dest_y, tool_item, ignore_item=target_obj):
            final_x, final_y = find_free_spot_from_centre(rdk, dest_x, dest_y, tool_item, target_obj)
        else:
            final_x, final_y = dest_x, dest_y

    put_down(rdk, robot, tool_item, gripper_mech, destination_coords={'x': final_x, 'y': final_y},
             drop_orientation=drop_orientation)


def put_down_in_area(rdk, robot, tool_item, gripper_mech, target_obj, area_bounds):
    print(f"\n *** Putting {target_obj.Name()} into defined area ***")

    if not is_holding_target(tool_item, target_obj):
        return

    min_point = (area_bounds['min_x'], area_bounds['min_y'])
    max_point = (area_bounds['max_x'], area_bounds['max_y'])

    print(f"> Target Area: X[{min_point[0]}:{max_point[0]}], Y[{min_point[1]}:{max_point[1]}]")

    final_x, final_y = find_free_spot_from_area(rdk, tool_item, min_point=min_point, max_point=max_point,held_object=target_obj)

    put_down(rdk, robot, tool_item, gripper_mech, target_obj=target_obj, destination_coords={'x': final_x, 'y': final_y}
    )


def shake(robot, tool_item, target_obj):
    print("\n *** Shaking ***")

    if not is_holding_target(tool_item, target_obj):
        return

    move_to_safe_mixing_pose(robot)

    joints = robot.Joints().list()

    shake_angle = 30

    for _ in range(4):
        joints[5] += shake_angle
        robot.MoveJ(joints)

        joints[5] -= 2 * shake_angle
        robot.MoveJ(joints)

        joints[5] += shake_angle
        robot.MoveJ(joints)

    robot.WaitMove()
    print("Shaking Complete.")


def swirl(robot, tool_item, target_obj, count=5, radius=20, speed=1000):
    print(f"\n *** Swirling {count} times (Radius: {radius}mm) ***")

    if not is_holding_target(tool_item, target_obj):
        return

    move_to_safe_mixing_pose(robot)

    print("> Starting Vortex")
    center_pose = robot.Pose()
    robot.setSpeed(speed)

    start_pose = center_pose * transl(radius, 0, 0)
    robot.MoveL(start_pose)

    for i in range(count):
        steps = 12
        for s in range(1, steps + 1):
            angle = s * (2 * math.pi / steps)

            x_offset = radius * math.cos(angle)
            y_offset = radius * math.sin(angle)

            target = center_pose * transl(x_offset, y_offset, 0)
            robot.MoveL(target)

    robot.MoveL(center_pose)
    robot.setSpeed(100)
    print("Action: Swirl Complete.")


def pour(rdk, robot, tool_item, held_obj, destination_obj):
    print(f"\n *** Pouring from {held_obj.Name()} into {destination_obj.Name()} ***")

    if not is_holding_target(tool_item, held_obj):
        return

    dest_x, dest_y, dest_z = get_true_center(destination_obj)

    _, _, held_h = get_object_size(held_obj)

    hover_z = dest_z + (held_h / 2) + 50

    shift_back_distance = -30

    final_x = dest_x
    final_y = dest_y + shift_back_distance

    print(f"> Pour Target: {destination_obj.Name()} (True Center: {dest_x:.1f}, {dest_y:.1f})")
    print(f"> Adjusted Approach: X={final_x:.1f}, Y={final_y:.1f}, Z={hover_z:.1f}")

    down_orient = get_down_orientation()
    pour_pose = transl(dest_x, dest_y, hover_z) * down_orient

    try:
        robot.MoveJ(pour_pose)
        robot.WaitMove()
    except Exception as e:
        print(f"> Error reaching pour position: {e}")
        return

    current_joints = robot.Joints().list()

    print("> Tilting to pour...")

    original_wrist_angle = current_joints[4]

    current_joints[4] += 110.0

    robot.MoveJ(current_joints)
    robot.WaitMove()

    print("> Pouring...")
    time.sleep(2.0)

    print("> Stopping flow...")
    current_joints[4] = original_wrist_angle
    robot.MoveJ(current_joints)
    robot.WaitMove()

    print("Pouring Complete.")