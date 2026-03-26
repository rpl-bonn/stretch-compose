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
        
def release_door():
    rclpy.init(args=None)
    # Initialize ROS nodes
    node = rclpy.create_node('test2_node')
    transform_manager = TransformManager(node)
    joint_position_node = JointPositionController(transform_manager.tf_buffer)
    joint_pose_node = JointPoseController()
    sleep_time=2
    pos = np.array([-0.76, -0.99, 1.11])  # Example joint positions
    direc = np.array([-0.04, -1.00, -0.02])  # Example direction vector
    
    new_pose = Pose3D(pos)
    new_pose.set_rot_from_direction(direc)
    print(new_pose)
    print("SLEEEEEEEEP")
    
    try:
        set_gripper(joint_pose_node, 0.4)
        move_arm(joint_position_node,new_pose)
        adjust_door(joint_pose_node, new_pose, pitch=-0.0, roll=0.0, lift=0.09)
        set_gripper(joint_pose_node, -0.3)

        print("Pose sequence completed.")
        
    except Exception as e:
        print("Error occurred while sending joint pose:", e)
        
    finally:
        joint_position_node.destroy_node()
    node.destroy_node()
    rclpy.shutdown()
    
def main(args=None):
    release_door()
            
if __name__ == '__main__':
    main()
