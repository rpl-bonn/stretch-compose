import json
import numpy as np
import os
import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
import tf2_ros
from threading import Thread
import traceback

from scripts.my_robot_scripts import searchnet_planning
from stretch_package.stretch_images.aligned_depth2color_subscriber import AlignedDepth2ColorSubscriber
from stretch_package.stretch_images.rgb_image_subscriber import RGBImageSubscriber
from stretch_package.stretch_movement.move_body import BaseController
from stretch_package.stretch_movement.move_head import HeadJointController
from stretch_package.stretch_movement.move_to_pose import JointPoseController
from stretch_package.stretch_movement.move_to_position import JointPositionController
from stretch_package.stretch_movement.stow_arm import StowArmController
from stretch_package.stretch_state.frame_transformer import FrameTransformer
from utils.coordinates import Pose3D
from utils.recursive_config import Config
from utils.robot_utils.advanced_movement import *
from utils.robot_utils.basic_movement import *
from utils.robot_utils.basic_perception import *
from utils.zero_shot_object_detection_sam3 import yolo_detect_object, detect_handle, detect_door_handle, detect_door_handle_sam3

# Adaptable
VIS_BLOCK = False
SAVE_BLOCK = True
NO_PROPOSALS = 3

# Config and Paths
config = Config()
scan_path = config.get_subpath("ipad_scans")
graph_path = config.get_subpath("scene_graph")
pcd_path = config.get_subpath("aligned_point_clouds")
ending = config["pre_scanned_graphs"]["high_res"]
SCAN_DIR = os.path.join(scan_path, ending)
GRAPH_DIR = os.path.join(graph_path, ending)
PCD_DIR = os.path.join(pcd_path, ending)
IMG_DIR = config.get_subpath("images")
image_topic = "/gripper_camera/color/image_rect_raw"
depth_topic = "/gripper_camera/aligned_depth_to_color/image_raw"

class TransformManager:
    def __init__(self, node: Node):
        self.node = node
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=100))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.node)
        self.node.get_logger().info("TransformListener initialized.")
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self._spin_thread = Thread(target=self._spin, daemon=True)
        self._spin_thread.start()
        
    def _spin(self):
        self.executor.spin()
            
    def _shutdown(self):
        self.executor.shutdown()
        self._spin_thread.join()
        self.node.get_logger().info("TransformListener stopped.")

    
def execute_search(drawer_id: int) -> bool:
    """
    Search for a queried object inside the drawers of the environment.

    Args:
        OBJECT (str): Object to search for and grasp.

    Returns:
        bool: True if the search and grasp was successful, False otherwise.
    """
    rclpy.init(args=None)
    success = False
    
    # Initialize ROS nodes
    node = rclpy.create_node('transform_manager_node')
    transform_manager = TransformManager(node)
    stow_node = StowArmController()
    base_node = BaseController()
    joint_pose_node = JointPoseController()
    head_node = HeadJointController(transform_manager.tf_buffer)
    joint_position_node = JointPositionController(transform_manager.tf_buffer)
    transform_node = FrameTransformer(transform_manager.tf_buffer)
    
    prompts = ["door", "drawer", "knob", "handle"]
    
    try: 
        print(f"Opening door {drawer_id}.")
        with open(os.path.join(GRAPH_DIR, "graph.json"), "r") as file:
            graph_data = json.load(file)
        with open(os.path.join(GRAPH_DIR, "scene.json"), "r") as file:
            scene_data = json.load(file)
        connections = graph_data["connections"]
        
        ## Check each drawer for the object    

        with open(os.path.join(GRAPH_DIR, "drawers", f"{drawer_id}.json"), "r") as file:
            drawer_data = json.load(file)
        drawer_center = Pose3D(np.array(drawer_data["centroid"]))
        furniture_id = str(connections[str(drawer_id)])
        furniture_center = scene_data["furniture"][furniture_id]["centroid"]
        furniture_name = scene_data["furniture"][furniture_id]["label"]
        print(f"Furniture name: {furniture_name}, Furniture center: {furniture_center}, Drawer center: {drawer_center}")
        
        body_pose = None
        front_normal = None
        
        # Move robot in front of the door
        body_pose, front_normal = plan_search(furniture_name, furniture_center, drawer_center, purpose="door")
            
        print('----------------------------------------------')
        odom = get_odom()
        current_pos = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y])
        print(f"Current robot position: {current_pos}")
        print(f"Body pose: {body_pose}, Front normal: {front_normal}")
        print('----------------------------------------------')
        
        move_in_front_of(stow_node, base_node, head_node, joint_pose_node, body_pose, drawer_center, 0.0, 0.0, 0.10, 0.09, stow=True, grasp=True)           
        look_for_door(joint_pose_node, wrist=0.01)
        
        # Take image of drawer to detect handle
        rgb_img = get_rgb_picture(RGBImageSubscriber, joint_pose_node, image_topic, gripper=True)
        depth_img = get_depth_picture(AlignedDepth2ColorSubscriber, joint_pose_node, depth_topic, gripper=True)
        look_for_door(joint_pose_node, wrist=0.01)
        # Skip if no handle detected
        
        IMG_DIR = config.get_subpath("images")
        vis_path = os.path.join(IMG_DIR, "gripper_cam.png")
        cv2.imwrite(vis_path, rgb_img)
        print(f"Image saved to {vis_path}")
        
        # handle_pose, door_type, _ = detect_door_handle(transform_node, depth_img, rgb_img)
        handle_pose, door_type, _ = detect_door_handle_sam3(transform_node, depth_img, rgb_img, prompts)
        print("----------------------------------------------------------")
        print(f"HANDLE POSE: {handle_pose}")
        print("----------------------------------------------------------")
        print(f"DOOR TYPE: {door_type}")
        print("----------------------------------------------------------")
        
        
        handle_pose.set_rot_from_direction(-front_normal)
        print("Next check Plan Drawer Search towards handle")
        # Refine robot position towards handle
        body_pose, front_normal = plan_search(furniture_name, furniture_center, handle_pose, purpose="handle")
            
        # Open drawer and check for object inside
        if door_type == "right":
            print("----------------- RIGHT DOOR OPENING -----------------")
            new_body_pos = new_pose_right(body_pose, handle_pose)
            door_exec(stow_node, base_node, head_node, joint_pose_node, joint_position_node, body_pose, new_body_pos, handle_pose, furniture_name, furniture_center, rgb_img, drawer_center)
            
        if door_type == "left":
            print("----------------- LEFT DOOR OPENING -----------------")
            new_body_pos = new_pose_left(body_pose, handle_pose)
            door_exec(stow_node, base_node, head_node, joint_pose_node, joint_position_node, body_pose, new_body_pos, handle_pose, furniture_name, furniture_center, rgb_img, drawer_center)
            
    
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        success = False
    
    finally:               
        # Destroy nodes
        transform_manager._shutdown()
        node.destroy_node()  
        stow_node.destroy_node()
        base_node.destroy_node()
        joint_pose_node.destroy_node()
        head_node.destroy_node()
        joint_position_node.destroy_node()
        transform_node.destroy_node()
        rclpy.shutdown()

    return success

def door_exec(stow_node: StowArmController, base_node: BaseController, head_node: HeadJointController, joint_pose_node: JointPoseController, joint_position_node: JointPositionController,
              body_pose: Pose3D, new_body_pos: Pose3D, handle_pose: Pose3D, furniture_name: str, furniture_center: np.ndarray, rgb_img: np.ndarray, drawer_center: np.ndarray) -> None:
    
    move_in_front_of(stow_node, base_node, head_node, joint_pose_node, body_pose, handle_pose, 0.0, 0.0, 0.0, 0.04, stow=False, grasp=True)   
     
    print(f'Body pose: {body_pose}')
    print(f'Handle pose: {handle_pose}')
    print(f"Calculated new body pose: {new_body_pos}")
    
    if "kitchen" in furniture_name:
        move_in_side_of(stow_node, base_node, head_node, joint_pose_node, new_body_pos, handle_pose, 0.0, 0.0, 0.0, 0.05, stow=False, grasp=True, small=True)
        print(f"Executing door opening for {furniture_name}")
        open_door_ik(joint_position_node, joint_pose_node, handle_pose, roll=0.0)
    elif "shelf" in furniture_name:
        move_in_side_of(stow_node, base_node, head_node, joint_pose_node, new_body_pos, handle_pose, 0.0, 0.0, 0.0, 0.05, stow=False, grasp=True, small=False)
        print(f"Executing door opening for {furniture_name}")
        open_door_ik(joint_position_node, joint_pose_node, handle_pose, roll=0.0)
    
    # Return to front of drawer
    
    body_pose_return, _ = plan_search(furniture_name, furniture_center, drawer_center, purpose="look")
    move_in_front_of(stow_node, base_node, head_node, joint_pose_node, body_pose_return, drawer_center, 0.0, 0.0, 0.0, 0.05, stow=True, grasp=True)
    
    # Look into door and capture image
    look_into_door(joint_pose_node, drawer_center)
    get_rgb_picture(RGBImageSubscriber, joint_pose_node, image_topic, gripper=True, save_block=True, vis_block=VIS_BLOCK)
    time.sleep(1.0)
    
    # Return back
    arm_move_back(joint_pose_node)

def plan_search(furniture_name: str, furniture_center: np.ndarray, target_pose: Pose3D, purpose: str = "door") -> tuple[Pose3D, np.ndarray]:
    body_pose = None
    front_normal = None
    
    if "kitchen" in furniture_name:
        print("Planning for Kitchen Counter")
        if purpose == "door":
            body_pose, front_normal = searchnet_planning.plan_door_search(furniture_name, furniture_center, target_pose, 1.0)
        elif purpose == "handle":
            body_pose, front_normal = searchnet_planning.plan_door_search(furniture_name, furniture_center, target_pose, 1.0)
        elif purpose == "look":
            body_pose, front_normal = searchnet_planning.plan_door_search(furniture_name, furniture_center, target_pose, 1.0)
            
    if "bookshelf" in furniture_name:
        print("Planning for Bookshelf")
        if purpose == "door":
            body_pose, front_normal = searchnet_planning.plan_door_search(furniture_name, furniture_center, target_pose, 0.6)
        elif purpose == "handle":
            body_pose, front_normal = searchnet_planning.plan_door_search(furniture_name, furniture_center, target_pose, 0.85)
        elif purpose == "look":
            body_pose, front_normal = searchnet_planning.plan_door_search(furniture_name, furniture_center, target_pose, 0.6)
        
    return body_pose, front_normal

def new_pose_right(body_pose:Pose3D, handle_pose:Pose3D) -> Pose3D:
    print("--------------------------------------------------")
    print(" BODY POSE: ", body_pose)
    print(" HANDLE POSE: ", handle_pose)
    print("-------------------------------------------------")
    x1, y1, z1 = body_pose.coordinates
    x2, y2, z2 = handle_pose.coordinates
    
    dx = x2 - x1
    dy = y2 - y1
    
    print(f"DIFFERENCE IN dx: {dx}, dy: {dy}")
    
    if abs(dx) < abs(dy):
        new_x = x1 - (dy / 2)
        new_y = y1 + dy / 2
    else:
        new_x = x1 + dx / 2
        new_y = y1 + (dx / 2)    
    
    new_pose = Pose3D(np.array([new_x, new_y, z1]))
    new_pose.direction = body_pose.direction
    print("NEW POSE (right handle): ", new_pose)
    print("--------------------------------------------------")
    return new_pose

def new_pose_left(body_pose:Pose3D, handle_pose:Pose3D) -> Pose3D:
    print("--------------------------------------------------")
    print(" BODY POSE: ", body_pose)
    print(" HANDLE POSE: ", handle_pose)
    print("-------------------------------------------------")
    x1, y1, z1 = body_pose.coordinates
    x2, y2, z2 = handle_pose.coordinates
    
    dx = x2 - x1
    dy = y2 - y1
    
    print(f"DIFFERENCE IN dx: {dx}, dy: {dy}")
    
    if abs(dx) < abs(dy):
        new_x = x1 + (dy / 2)
        new_y = y1 + dy / 2
    else:
        new_x = x1 + dx / 2
        new_y = y1 - (dx / 2)    
    
    new_pose = Pose3D(np.array([new_x, new_y, z1]))
    new_pose.direction = body_pose.direction
    print("NEW POSE (left handle): ", new_pose)
    print("--------------------------------------------------")
    return new_pose

def open_door_ik(joint_position_node: JointPositionController, joint_pose_node: JointPoseController, handle_pose: Pose3D, roll: float) -> None:
    handle_pose.coordinates[1] += -0.04
    move_arm(joint_position_node, handle_pose, roll=roll)
    set_gripper(joint_pose_node, 0.65)
    adjust_door(joint_pose_node, handle_pose, pitch=-0.0, roll=roll, lift=0.07)
    set_gripper(joint_pose_node, -0.35)
    open_door(joint_pose_node)
    set_gripper(joint_pose_node, 0.65)
    

if __name__ == "__main__":
    # docker exec -it heuristic_khorana bash -c "source /opt/ros/humble/setup.bash && source /home/ws/install/setup.bash && ros2 run sam3_inference sam3_service.py"
    
    args = os.sys.argv[1]
    execute_search(args)
    
    # id=28
    # execute_search(id)
    