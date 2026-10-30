"""
One ROS lifecycle for a demo run
"""

from contextlib import contextmanager
from dataclasses import dataclass
from threading import Thread

import rclpy
import tf2_ros
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from stretch_package.stretch_movement.mode_controller import ModeController
from stretch_package.stretch_movement.move_body import BaseController
from stretch_package.stretch_movement.move_head import HeadJointController
from stretch_package.stretch_movement.move_to_pose import JointPoseController
from stretch_package.stretch_movement.move_to_position import JointPositionController
from stretch_package.stretch_movement.stow_arm import StowArmController
from stretch_package.stretch_state.frame_transformer import FrameTransformer


class TransformManager:
    """Spins a node in a background thread so the TF buffer stays filled."""

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


@dataclass
class Robot:
    """
    The robot's ROS handles
    """

    node: Node
    transform_manager: TransformManager
    stow: StowArmController
    base: BaseController
    pose: JointPoseController
    head: HeadJointController
    position: JointPositionController
    tf: FrameTransformer
    mode: ModeController

    def switch_to_position_mode(self) -> bool:
        return self.mode.switch_to_position_mode()

    def switch_to_navigation_mode(self) -> bool:
        return self.mode.switch_to_navigation_mode()

    def look_front(self) -> None:
        #move head camera to 0 position using send_joint_pose
        self.pose.send_joint_pose({'head_tilt': 0.0, 'head_pan': 0.0})


@contextmanager
def robot_session(node_name: str = 'transform_manager_node'):
    """
    Bring up every node a demo needs, then tear the whole context down again
    """
    
    rclpy.init(args=None)
    transform_manager = None
    nodes = []
    try:
        transform_manager = TransformManager(rclpy.create_node(node_name))
        tf_buffer = transform_manager.tf_buffer
        
        for controller in (StowArmController, BaseController, JointPoseController,
                           lambda: HeadJointController(tf_buffer),
                           lambda: JointPositionController(tf_buffer),
                           lambda: FrameTransformer(tf_buffer),
                           ModeController):
            nodes.append(controller())
        stow, base, pose, head, position, tf, mode = nodes

        yield Robot(node=transform_manager.node, transform_manager=transform_manager,
                    stow=stow, base=base, pose=pose, head=head, position=position, tf=tf,
                    mode=mode)
    finally:
        if transform_manager is not None:
            transform_manager._shutdown()
            transform_manager.node.destroy_node()
        for node in nodes:
            node.destroy_node()
        rclpy.shutdown()
