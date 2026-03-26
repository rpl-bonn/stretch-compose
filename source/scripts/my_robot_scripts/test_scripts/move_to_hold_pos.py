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
import time
from scripts.my_robot_scripts import searchnet_planning
from stretch_package.stretch_images.aligned_depth2color_subscriber import AlignedDepth2ColorSubscriber
from stretch_package.stretch_images.rgb_image_subscriber import RGBImageSubscriber
from stretch_package.stretch_movement.move_body import BaseController
from stretch_package.stretch_movement.move_head import HeadJointController
from stretch_package.stretch_movement.move_to_pose import JointPoseController
from stretch_package.stretch_movement.move_to_position import JointPositionController
from stretch_package.stretch_movement.stow_arm import StowArmController
from stretch_package.stretch_state.frame_transformer import FrameTransformer
from utils.coordinates import Pose2D, Pose3D, from_a_to_b_distanced, pose_distanced, get_door_opening_poses
from utils.recursive_config import Config
from utils.robot_utils.advanced_movement import *
from utils.robot_utils.basic_movement import *
from utils.robot_utils.basic_perception import *
from utils.zero_shot_object_detection import yolo_detect_object, detect_handle, detect_door_handle
from scripts.my_robot_scripts.test_scripts.detect_handle_standalone import get_handle_pose

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

def move_to_hold():
    rclpy.init(args=None)
    # Initialize ROS nodes
    node = rclpy.create_node('move_to_hold_node')
    stow_node = StowArmController()
    base_node = BaseController()
    joint_pose_node = JointPoseController()
    transform_manager = TransformManager(node)
    
    sleep_time=1
    
    base_pos1 = [-0.954, -1.119, 0.0, 0.0, 0.28, -0.74896684]

    body_pose = np.array([-0.66751, -0.45766, -0.45])
    target_pose = np.array([-0.66751, -1.1577, 0.66751])
    
    pose = Pose3D(body_pose)
    pose.set_rot_from_direction(target_pose - body_pose)

    try:
        print("Moving to look position...")
        
        move_body_test(base_node, base_pos1)
        print("BASE MOVED")
        
        turn_body_test(joint_pose_node, base_pos1)
        print("BASE TURNED")
        
        time.sleep(sleep_time)
        
        print("Pose sequence completed.")
        
    except Exception as e:
        print("Error occurred while sending joint pose:", e)
        
    finally:
        joint_pose_node.destroy_node()
        node.destroy_node()
        rclpy.shutdown()

def get_handle_pose(transform_manager, stow_node, base_node, joint_pose_node):
    transform_node = FrameTransformer(transform_manager.tf_buffer)

    # Adaptable
    VIS_BLOCK = False
    SAVE_BLOCK = True
    NO_PROPOSALS = 3
    OBJECT = "tennis ball"

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

    print("All ROS nodes initialized.")

    try:
        print(f"Searching for {OBJECT} in the environment...")
        with open(os.path.join(GRAPH_DIR, "graph.json"), "r") as file:
            graph_data = json.load(file)
        with open(os.path.join(GRAPH_DIR, "scene.json"), "r") as file:
            scene_data = json.load(file)
        connections = graph_data["connections"]

        rgb_img = get_rgb_picture(RGBImageSubscriber, joint_pose_node, "/gripper_camera/color/image_rect_raw", gripper=True)
        depth_img = get_depth_picture(AlignedDepth2ColorSubscriber, joint_pose_node, "/gripper_camera/aligned_depth_to_color/image_raw", gripper=True)

        handle_pose, drawer_type, hinge_pose = detect_door_handle(transform_node, depth_img, rgb_img)
        print("----------------------------------------------------------")
        print(f"HANDLE POSE: {handle_pose}")
        print("----------------------------------------------------------")
        print(f"DRAWER TYPE: {drawer_type}")
        print("----------------------------------------------------------")
        print(f"HINGE POSE: {hinge_pose}")
        print("----------------------------------------------------------")
        
    except Exception as e:
        print("An error occurred during the operation:")
        print(str(e))
        traceback.print_exc()
        
def main(args=None):
    move_to_hold()

if __name__ == '__main__':
    main()
# rclpy.init(args=None)
# # Initialize ROS nodes
# node = rclpy.create_node('hold_door_node')
# stow_node = StowArmController()
# base_node = BaseController()
# joint_pose_node = JointPoseController()

# static_joint_positions = {
#             'joint_lift': 0.8792,
#             'joint_arm_l0': 0.11866,
#             'joint_arm_l1': 0.11866,
#             'joint_arm_l2': 0.11866,
#             'joint_arm_l3': 0.11866,
#             'joint_wrist_yaw': 0.8194,
#             'joint_gripper_finger_left': 0.4
#         }
# err = "test"
# sleep_time=5
# pos_1 = {'joint_lift': 0.9792 } # lift up
# pos_1_1= {'joint_wrist_yaw': 1.09615} # rotate wrist
# pos_2 = {'wrist_extension': (0.4911,40.0)} # extend wrist
# pos_3 = {'joint_gripper_finger_left': 0.35} # open gripper
# pos_4 = {'joint_wrist_pitch': 0.161 } # pitch
# pos_5 = {'joint_wrist_roll': 0.084} # roll
# pos_6 = {'joint_wrist_yaw': 0.8119} # rotate wrist
# pos_7 = {'joint_gripper_finger_left': -0.3}
# try:
#     joint_pose_node.send_joint_pose(pos_1)
#     spin_until_complete(joint_pose_node)
#     print("Executed Pose 1")
#     err = "after pose 1"
#     time.sleep(sleep_time)

#     joint_pose_node.send_joint_pose(pos_1_1)
#     spin_until_complete(joint_pose_node)
#     print("Executed Pose 1_1")
#     err = "after pose 1_1"
#     time.sleep(sleep_time)
    
#     joint_pose_node.send_joint_pose(pos_2)
#     spin_until_complete(joint_pose_node)
#     print("Executed Pose 2")
#     err = "after pose 2"
#     time.sleep(sleep_time)
    
#     joint_pose_node.send_joint_pose(pos_3)
#     spin_until_complete(joint_pose_node)
#     print("Executed Pose 3")
#     err = "after pose 3"
#     time.sleep(sleep_time)

#     joint_pose_node.send_joint_pose(pos_4)
#     spin_until_complete(joint_pose_node)
#     print("Executed Pose 4")
#     err = "after pose 4"
#     time.sleep(sleep_time)

#     joint_pose_node.send_joint_pose(pos_5)
#     spin_until_complete(joint_pose_node)
#     print("Executed Pose 5")
#     err = "after pose 5"
#     time.sleep(sleep_time)

#     joint_pose_node.send_joint_pose(pos_6)
#     spin_until_complete(joint_pose_node)
#     print("Executed Pose 6")
#     err = "after pose 6"
#     time.sleep(sleep_time)

#     joint_pose_node.send_joint_pose(pos_7)
#     spin_until_complete(joint_pose_node)
#     print("Executed Pose 7")
#     err = "after pose 7"
#     time.sleep(sleep_time)

#     print("Pose sequence completed.")
# except Exception as e:
    
#     print("Error occurred while sending joint pose:", e)
    
# finally:
#     joint_pose_node.destroy_node()
# node.destroy_node()
# rclpy.shutdown()