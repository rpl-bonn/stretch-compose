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
from utils.zero_shot_object_detection import yolo_detect_object, detect_handle



def execute_home_pose():
    rclpy.init(args=None)
    # Initialize ROS nodes
    node = rclpy.create_node('home_pose_node')
    stow_node = StowArmController()
    base_node = BaseController()
    joint_pose_node = JointPoseController()
    sleep_time=1
    
    pose_1 = {'wrist_extension': (0.1071,40.0)}
    pose_2 = {'joint_lift': 0.3729}
    pose_3 = {'joint_wrist_yaw': 0.8187}
    pose_4 = {'joint_wrist_pitch': -0.53229}
    pose_5 = {'joint_wrist_roll': 0.0874}
    pose_6 = {'joint_gripper_finger_left': 0.4}

    try:
        joint_pose_node.send_joint_pose(pose_6)
        spin_until_complete(joint_pose_node)
        print("Executed Pose 6")
        time.sleep(sleep_time)
        
        joint_pose_node.send_joint_pose(pose_1)
        spin_until_complete(joint_pose_node)
        print("Executed Pose 1")
        time.sleep(sleep_time)

        joint_pose_node.send_joint_pose(pose_2)
        spin_until_complete(joint_pose_node)
        print("Executed Pose 2")
        time.sleep(sleep_time)

        joint_pose_node.send_joint_pose(pose_3)
        spin_until_complete(joint_pose_node)
        print("Executed Pose 3")
        time.sleep(sleep_time)

        joint_pose_node.send_joint_pose(pose_4)
        spin_until_complete(joint_pose_node)
        print("Executed Pose 4")
        time.sleep(sleep_time)

        joint_pose_node.send_joint_pose(pose_5)
        spin_until_complete(joint_pose_node)
        print("Executed Pose 5")
        time.sleep(sleep_time)

        print("Pose sequence completed.")
        
    except Exception as e:
        print("Error occurred while sending joint pose:", e)
        
    finally:
        joint_pose_node.destroy_node()
    node.destroy_node()
    rclpy.shutdown()
            
def main(args=None):
    execute_home_pose()
    
if __name__ == '__main__':
    main()
# rclpy.init(args=None)
# # Initialize ROS nodes
# node = rclpy.create_node('transform_manager_node')
# stow_node = StowArmController()
# base_node = BaseController()
# joint_pose_node = JointPoseController()


# # static_joint_positions = {
# #             'wrist_extension': (0.0145,40.0),
# #             'joint_lift': 0.1729,
# #             'joint_wrist_yaw': 0.8187,
# #             'joint_wrist_pitch': -0.53229,
# #             'joint_wrist_roll': 0.0874,
# #             'joint_gripper_finger_left': 0.4
# #         }
# sleep_time=3
# pose_1 = {'wrist_extension': (0.1071,40.0)}
# pose_2 = {'joint_lift': 0.1729}
# pose_3 = {'joint_wrist_yaw': 0.8187}
# pose_4 = {'joint_wrist_pitch': -0.53229}
# pose_5 = {'joint_wrist_roll': 0.0874}
# pose_6 = {'joint_gripper_finger_left': 0.4}

# try:
#     joint_pose_node.send_joint_pose(pose_1)
#     spin_until_complete(joint_pose_node)
#     print("Executed Pose 1")
#     time.sleep(sleep_time)
    
#     joint_pose_node.send_joint_pose(pose_2)
#     spin_until_complete(joint_pose_node)
#     print("Executed Pose 2")
#     time.sleep(sleep_time)
    
#     joint_pose_node.send_joint_pose(pose_3)
#     spin_until_complete(joint_pose_node)
#     print("Executed Pose 3")
#     time.sleep(sleep_time)

#     joint_pose_node.send_joint_pose(pose_4)
#     spin_until_complete(joint_pose_node)
#     print("Executed Pose 4")
#     time.sleep(sleep_time)

#     joint_pose_node.send_joint_pose(pose_5)
#     spin_until_complete(joint_pose_node)
#     print("Executed Pose 5")
#     time.sleep(sleep_time)

#     joint_pose_node.send_joint_pose(pose_6)
#     spin_until_complete(joint_pose_node)
#     print("Executed Pose 6")
#     time.sleep(sleep_time)

#     print("Pose sequence completed.")
# except Exception as e:
#     print("Error occurred while sending joint pose:", e)
    
# finally:
#     joint_pose_node.destroy_node()
# node.destroy_node()
# rclpy.shutdown()