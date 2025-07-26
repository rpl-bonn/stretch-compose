import os
import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from threading import Thread
import tf2_ros

from scripts.my_robot_scripts import graspnet_planning
from stretch_package.stretch_images.compressed_image_subscriber import CompressedImageSubscriber
from stretch_package.stretch_movement.move_body import BaseController
from stretch_package.stretch_movement.move_head import HeadJointController
from stretch_package.stretch_movement.move_to_pose import JointPoseController
from stretch_package.stretch_movement.move_to_position import JointPositionController
from stretch_package.stretch_movement.stow_arm import StowArmController
from utils.recursive_config import Config
from utils.robot_utils.advanced_movement import *
from utils.robot_utils.basic_movement import *
from utils.robot_utils.basic_perception import *
from utils.zero_shot_object_detection import yolo_detect_object

# Adaptable
VIS_BLOCK = True
SAVE_BLOCK = True
OBJECT = 'bottle'

# Config and Paths
config = Config()
scan_path = config.get_subpath("ipad_scans")
ending = config["pre_scanned_graphs"]["high_res"]
IMG_DIR = config.get_subpath("images")
SCAN_DIR = os.path.join(scan_path, ending)


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
        
            
def execute_poses() -> bool:
    """
    Execute the poses for the robot to grasp an object.
    The function initializes the robot's nodes, plans the poses for the robot body and the grasp,
    and executes the movements to grasp the object.

    Returns:
        bool: True if object is detected and successfully grasped, False otherwise.
    """
    rclpy.init(args=None)
    success = False
    
    # Initialize nodes
    node = rclpy.create_node('transform_manager_node')
    transform_manager = TransformManager(node)
    stow_node = StowArmController()
    base_node = BaseController()
    joint_pose_node = JointPoseController()
    head_node = HeadJointController(transform_manager.tf_buffer)
    joint_position_node = JointPositionController(transform_manager.tf_buffer)
    
    # Get body and grasp pose
    body_pose_distanced, grasp_pose_new, obj_width = graspnet_planning.plan_poses(OBJECT, VIS_BLOCK)
    print(f"Planned poses: body={body_pose_distanced}, grasp={grasp_pose_new}")

    # Stow robot 
    stow_arm(stow_node)

    # Move body to goal position
    move_body(base_node, body_pose_distanced.to_dimension(2))
    
    # Turn body towards goal object
    turn_body(joint_pose_node, grasp_pose_new.to_dimension(2), True)
    
    # Look at goal position & detect object
    move_head(head_node, grasp_pose_new, -0.01)
    time.sleep(1)
    get_rgb_picture(RGBImageSubscriber, joint_pose_node,  '/camera/color/image_raw', gripper=False, save_block=SAVE_BLOCK, vis_block=VIS_BLOCK)
    detected, _ = yolo_detect_object(OBJECT, "head", save_block=SAVE_BLOCK)
    print(f"Object detected at original location: {detected}")

    if detected:
        unstow_arm(joint_pose_node, grasp_pose_new, yaw=0.0)
        positional_grab(joint_position_node, joint_pose_node, grasp_pose_new, 0.15, -0.05)
        success = True   
    else:
        node.get_logger().info(f"{OBJECT} is not at its place anymore.") 

    # Destroy nodes
    transform_manager._shutdown()
    node.destroy_node()  
    stow_node.destroy_node()
    base_node.destroy_node()
    joint_pose_node.destroy_node()
    head_node.destroy_node()
    joint_position_node.destroy_node()
    rclpy.shutdown()
    
    return success


if __name__ == "__main__":
    execute_poses()
   