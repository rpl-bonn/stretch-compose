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
from utils.coordinates import Pose3D
from utils.recursive_config import Config
from utils.robot_utils.advanced_movement import *
from utils.robot_utils.basic_movement import *
from utils.robot_utils.basic_perception import *
from utils.zero_shot_object_detection import yolo_detect_object, detect_handle



def hold_door():
    rclpy.init(args=None)
    # Initialize ROS nodes
    node = rclpy.create_node('hold_door_node')
    stow_node = StowArmController()
    base_node = BaseController()
    joint_pose_node = JointPoseController()
    
    sleep_time=1
    pos_1 = {'joint_lift': 0.9792 }# lift up
    pos_pitch_initial = {'joint_wrist_pitch': -0.9449} # initial pitch
    pos_yaw_initial = {'joint_wrist_yaw': 0.249911} # initial yaw
    pos_1_1= {'joint_wrist_yaw': 1.09615} # rotate wrist
    pos_2 = {'wrist_extension': (0.4981,40.0)} # extend wrist
    pos_3 = {'joint_gripper_finger_left': 0.35} # open gripper
    pos_4 = {'joint_wrist_pitch': 0.161 } # pitch
    pos_5 = {'joint_wrist_roll': 0.084} # roll
    pos_6 = {'joint_wrist_yaw': 0.8119} # rotate wrist
    pos_7 = {'joint_gripper_finger_left': -0.3} # close gripper
    
    try:
        joint_pose_node.send_joint_pose(pos_1)
        spin_until_complete(joint_pose_node)
        print("Executed Pose 1")
        time.sleep(sleep_time)
        
        joint_pose_node.send_joint_pose(pos_pitch_initial)
        spin_until_complete(joint_pose_node)
        print("Executed Pose 1")
        time.sleep(sleep_time)

        joint_pose_node.send_joint_pose(pos_yaw_initial)
        spin_until_complete(joint_pose_node)
        print("Executed Pose 1")
        time.sleep(sleep_time)

        joint_pose_node.send_joint_pose(pos_1_1)
        spin_until_complete(joint_pose_node)
        print("Executed Pose 1_1")
        time.sleep(sleep_time)

        joint_pose_node.send_joint_pose(pos_2)
        spin_until_complete(joint_pose_node)
        print("Executed Pose 2")
        time.sleep(sleep_time)

        joint_pose_node.send_joint_pose(pos_3)
        spin_until_complete(joint_pose_node)
        print("Executed Pose 3")
        time.sleep(sleep_time)

        joint_pose_node.send_joint_pose(pos_4)
        spin_until_complete(joint_pose_node)
        print("Executed Pose 4")
        time.sleep(sleep_time)

        joint_pose_node.send_joint_pose(pos_5)
        spin_until_complete(joint_pose_node)
        print("Executed Pose 5")
        time.sleep(sleep_time)

        joint_pose_node.send_joint_pose(pos_6)
        spin_until_complete(joint_pose_node)
        print("Executed Pose 6")
        time.sleep(sleep_time)

        joint_pose_node.send_joint_pose(pos_7)
        spin_until_complete(joint_pose_node)
        print("Executed Pose 7")
        time.sleep(sleep_time)

        print("Pose sequence completed.")
        
    except Exception as e:
        print("Error occurred while sending joint pose:", e)
        
    finally:
        joint_pose_node.destroy_node()
        node.destroy_node()
        rclpy.shutdown()

def main(args=None):
    hold_door()
    
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