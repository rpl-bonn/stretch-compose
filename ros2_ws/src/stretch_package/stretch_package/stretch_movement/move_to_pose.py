from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from rclpy.duration import Duration

class JointPoseController(Node):
    
    def __init__(self):
        super().__init__('jointpose_controller_node')
        self.action_client = ActionClient(self, FollowJointTrajectory, '/stretch_controller/follow_joint_trajectory')
        self.goal_future = None
        self.result_future = None
        self.done = False
        # Whether the last goal reached its target; False if it was rejected, stopped on contact or preempted
        self.succeeded = False
        self.joint_names = []
      
        
    def send_joint_pose(self, pose, velocities=None):
        # A (position, contact_threshold) tuple guards that joint. stretch_driver reads effort[i]
        # for the i-th joint name and treats joints past the end of effort as unguarded, so the
        # guarded joints go first and effort lines up with them.
        # velocities, if given, holds a speed for every joint in pose (m/s or rad/s).
        guarded = [k for k, p in pose.items() if isinstance(p, (tuple, list))]
        joint_names = guarded + [k for k in pose if k not in guarded]

        trajectory_point = JointTrajectoryPoint()
        trajectory_point.positions = [float(pose[k][0]) if k in guarded else float(pose[k]) for k in joint_names]
        if guarded:
            trajectory_point.effort = [float(pose[k][1]) for k in guarded]
        if velocities is not None:
            trajectory_point.velocities = [float(velocities[k]) for k in joint_names]
        trajectory_point.time_from_start = Duration(seconds=3.0).to_msg()

        trajectory_goal = FollowJointTrajectory.Goal()
        trajectory_goal.goal_time_tolerance = Duration(seconds=1.0).to_msg()
        trajectory_goal.trajectory.joint_names = joint_names
        self.joint_names, self.succeeded = joint_names, False
        trajectory_goal.trajectory.points.append(trajectory_point)

        self.action_client.wait_for_server()
        #self.get_logger().info('Sending goal to /follow_joint_trajectory...')
        self.goal_future = self.action_client.send_goal_async(trajectory_goal)
        self.goal_future.add_done_callback(self.response_callback)
   
   
    def response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error(f'Goal for {self.joint_names} rejected!')
            self.done = True
            return
        #self.get_logger().info('Goal accepted!')
        self.result_future = goal_handle.get_result_async()
        self.result_future.add_done_callback(self.result_callback)
     
        
    def result_callback(self, future):
        result = future.result().result
        status = future.result().status
        self.succeeded = status == GoalStatus.STATUS_SUCCEEDED
        if not self.succeeded:
            self.get_logger().error(f'Goal for {self.joint_names} failed with status {status} '
                                    f'(error code {result.error_code}: {result.error_string!r})')
        self.done = True
                     