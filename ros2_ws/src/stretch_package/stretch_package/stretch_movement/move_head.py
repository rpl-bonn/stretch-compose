import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.time import Time
from action_msgs.msg import GoalStatus
import tf2_ros
import numpy as np
from scipy.spatial.transform import Rotation

class HeadJointController(Node):
    
    def __init__(self, buffer: tf2_ros.Buffer):
        super().__init__("head_joint_controller_node")
        self.buffer = buffer    
        self.action_client = ActionClient(self, FollowJointTrajectory, '/stretch_controller/follow_joint_trajectory')
        self.done = False
    
    
    def transform_goal(self, target_point_map):
        while rclpy.ok():
            try:
                if not self.buffer.can_transform('link_head', 'map', Time()):
                    self.get_logger().warn('Transform from map to link_head is not available yet. Retrying...')
                    rclpy.spin_once(self, timeout_sec=1.0)
                    continue
                tf = self.buffer.lookup_transform("link_head", "map", Time())
                t = np.eye(4)
                q = tf.transform.rotation
                x = tf.transform.translation
                t[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                t[:3, 3] = [x.x, x.y, x.z]
                return t.dot(target_point_map)[:3]
            except Exception as e:
                self.get_logger().warn(f"Transform from map to link_head failed: {e}")          
        
        
    def send_joint_pose(self, pos, tilt_bool=True):
        target_point_map = np.append(pos, 1.0) 
        target_point_head = self.transform_goal(target_point_map)
        print(f"target point: {target_point_head}")
        
        pan = np.arctan2(target_point_head[0], target_point_head[1])
        if tilt_bool:
            tilt = np.arctan2(target_point_head[2], np.sqrt(target_point_head[0]**2 + target_point_head[1]**2))
        else:
            tilt = 0.0
        print(f"pan: {pan}, tilt: {tilt}")
        
        joint_names = ['joint_head_pan', 'joint_head_tilt']
        
        trajectory_point = JointTrajectoryPoint()
        trajectory_point.positions = [-pan, tilt]
        trajectory_point.time_from_start = Duration(seconds=3.0).to_msg()
        
        trajectory_goal = FollowJointTrajectory.Goal()
        trajectory_goal.goal_time_tolerance = Duration(seconds=1.0).to_msg()
        trajectory_goal.trajectory.joint_names = joint_names
        trajectory_goal.trajectory.points.append(trajectory_point)
        
        self.action_client.wait_for_server()
        #self.get_logger().info('Sending goal to /follow_joint_trajectory')
        self.future = self.action_client.send_goal_async(trajectory_goal)
        self.future.add_done_callback(self.response_callback)
    
    
    def response_callback(self, future):
        result = future.result()
        if not result.accepted:
            self.get_logger().error('Goal rejected!')
            self.done = True
            return
        #self.get_logger().info('Goal accepted!')
        future = result.get_result_async()
        future.add_done_callback(self.result_callback)
      
        
    def result_callback(self, future):
        result = future.result().result
        status = future.result().status
        if status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error('Goal failed with status: {0}'.format(status))
        self.done = True
