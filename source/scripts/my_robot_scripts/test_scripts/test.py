import json
import numpy as np
import os
import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
import tf2_ros
from threading import Thread
import time
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

def release_door():
    rclpy.init(args=None)
    # Initialize ROS nodes
    node = rclpy.create_node('release_door_node')
    stow_node = StowArmController()
    base_node = BaseController()
    joint_pose_node = JointPoseController()
    sleep_time=1
    pos_1 = {'wrist_extension': 0.04, 'joint_wrist_yaw': 0.3}
    
    try:
        joint_pose_node.send_joint_pose(pos_1)
        spin_until_complete(joint_pose_node)
        print("Executed Pose 1")
        time.sleep(sleep_time)

        print("Pose sequence completed.")
        
    except Exception as e:
        print("Error occurred while sending joint pose:", e)
        
    finally:
        joint_pose_node.destroy_node()
    node.destroy_node()
    rclpy.shutdown()
    
def main(args=None):
    release_door()
            
if __name__ == '__main__':
    main()
# rclpy.init(args=None)
# # Initialize ROS nodes
# node = rclpy.create_node('transform_manager_node')
# stow_node = StowArmController()
# base_node = BaseController()
# joint_pose_node = JointPoseController()

# sleep_time=5
# pose_1 = {'wrist_extension': (0.414,60.0)}
# pose_2 = {'joint_wrist_yaw': 0.633}
# pose_3 = {'wrist_extension': (0.3076,60.0)}
# pose_4 = {'joint_wrist_yaw': 0.35}
# pose_5 = {'wrist_extension': (0.066,60.0)}
# pose_6 = {'joint_gripper_finger_left': 0.2}
# pose_7 = {'joint_wrist_yaw': -0.324}

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
    
#     joint_pose_node.send_joint_pose(pose_7)
#     spin_until_complete(joint_pose_node)
#     print("Executed Pose 7")
#     time.sleep(sleep_time)

#     print("Pose sequence completed.")
    
# except Exception as e:
#     print("Error occurred while sending joint pose:", e)
    
# finally:
#     joint_pose_node.destroy_node()
# node.destroy_node()
# rclpy.shutdown()