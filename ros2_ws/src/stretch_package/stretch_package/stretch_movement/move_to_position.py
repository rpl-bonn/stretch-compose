import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
import tf2_ros
import numpy as np
import ikpy.urdf.utils
import ikpy.chain
from sensor_msgs.msg import JointState
from rclpy.duration import Duration
from scipy.spatial.transform import Rotation
from action_msgs.msg import GoalStatus
from rclpy.time import Time

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*fixed.*")

class JointPositionController(Node):
    
    def __init__(self, buffer: tf2_ros.Buffer):
        super().__init__("joint_position_controller_node")
        self.buffer = buffer
        self.action_client = ActionClient(self, FollowJointTrajectory, '/stretch_controller/follow_joint_trajectory')
        self.joint_state = None
        self.done = False
         
        iktuturdf_path = "/home/ws/data/stretch_description/tmp/stretch.urdf"
        self.chain = ikpy.chain.Chain.from_urdf_file(iktuturdf_path)
        #self.chain.active_links_mask = [link.joint_type != 'fixed' for link in self.chain.links]
        #print("Active links: ", self.chain.active_links_mask)
        
               
    def bound_range(self, name, value):
        names = [l.name for l in self.chain.links]
        index = names.index(name)
        bounds = self.chain.links[index].bounds
        return min(max(value, bounds[0]), bounds[1]) 
            
    def transform_goal(self, target_pos_map, target_dir_map):
        while rclpy.ok():
            try:
                if not self.buffer.can_transform('base_link', 'map', Time()):
                    self.get_logger().warn('Transform from map to base_link is not available yet. Retrying...')
                    rclpy.spin_once(self, timeout_sec=1.0)
                    continue
                tf = self.buffer.lookup_transform("base_link", "map", Time())       
                t = np.eye(4)
                q = tf.transform.rotation
                x = tf.transform.translation
                t[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                t[:3, 3] = [x.x, x.y, x.z]
                return t.dot(target_pos_map)[:3], t[:3, :3].dot(target_dir_map)
            except Exception as e:
                self.get_logger().warn(f"Transform from map to base_link failed: {e}")        
        
    def calculate_ik(self, pos, dir): 
        target_pos_map = np.append(pos, 1.0)
        target_pos_base, target_dir_base = self.transform_goal(target_pos_map, dir)
        
        roll = 0.0
        pitch = np.arctan2(-target_dir_base[2], np.sqrt(target_dir_base[0]**2 + target_dir_base[1]**2))
        yaw = np.arctan2(target_dir_base[1], target_dir_base[0])
        target_orientation = ikpy.utils.geometry.rpy_matrix(roll, pitch, yaw)
        print(f"roll: {roll}, pitch: {pitch}, yaw: {yaw}")
        
        # Set initial joint configuration
        q_base = 0.0
        q_lift = self.bound_range('joint_lift', self.joint_state.position[self.joint_state.name.index('joint_lift')])
        q_arml = self.bound_range('joint_arm_l0', self.joint_state.position[self.joint_state.name.index('joint_arm_l0')]/ 4.0)
        q_yaw = self.bound_range('joint_wrist_yaw', self.joint_state.position[self.joint_state.name.index('joint_wrist_yaw')])
        q_pitch = self.bound_range('joint_wrist_pitch', self.joint_state.position[self.joint_state.name.index('joint_wrist_pitch')])
        q_roll = self.bound_range('joint_wrist_roll', self.joint_state.position[self.joint_state.name.index('joint_wrist_roll')])
        q_init = [0.0, q_base, 0.0, q_lift, 0.0, q_arml, q_arml, q_arml, q_arml, q_yaw, 0.0, q_pitch, q_roll, 0.0, 0.0]
        
        # Calculate joint configuration using inverse kinematics
        q = self.chain.inverse_kinematics(target_pos_base, target_orientation, orientation_mode='all', initial_position=q_init)
        pose_values = [q[1], q[3], q[5]+q[6]+q[7]+q[8], q[9], q[11], q[12]]
        pose_names = ['translate_mobile_base', 'joint_lift', 'wrist_extension', 'joint_wrist_yaw', 'joint_wrist_pitch', 'joint_wrist_roll']
        pose = dict(zip(pose_names, pose_values))
        print(pose)

        err = np.linalg.norm(self.chain.forward_kinematics(q)[:3, 3] - target_pos_base)
        if not np.isclose(err, 0.0, atol=1e-2):
            print("IKPy did not find a valid solution")

        return pose
        
    def send_joint_pos(self, pos, dir, joint_state: JointState):
        self.joint_state = joint_state
        pose = self.calculate_ik(pos, dir)
        joint_names = [key for key in pose]

        trajectory_point = JointTrajectoryPoint()
        trajectory_point.positions = [pose[key] for key in joint_names]
        trajectory_point.time_from_start = Duration(seconds=3.0).to_msg()

        trajectory_goal = FollowJointTrajectory.Goal()
        trajectory_goal.goal_time_tolerance = Duration(seconds=1.0).to_msg()
        trajectory_goal.trajectory.joint_names = joint_names
        trajectory_goal.trajectory.points.append(trajectory_point)

        self.action_client.wait_for_server()
        self.get_logger().info('Sending goal to /follow_joint_trajectory')
        future = self.action_client.send_goal_async(trajectory_goal)
        future.add_done_callback(self.response_callback)
    
    
    def response_callback(self, future):
        result = future.result()
        if not result.accepted:
            self.get_logger().error('Goal rejected!')
            self.done = True
            return
        self.get_logger().info('Goal accepted!')
        future = result.get_result_async()
        future.add_done_callback(self.result_callback)


    def result_callback(self, future):  
        result = future.result().result
        status = future.result().status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Goal succeeded! Result: {0}'.format(result))
        else:
            self.get_logger().info('Goal failed with status: {0}'.format(status))
        self.done = True
