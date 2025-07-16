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
      
        
    def send_joint_pose(self, pose):
        joint_names = list(pose.keys())
        
        trajectory_point = JointTrajectoryPoint()
        poses = [pose[key] for key in joint_names]
        trajectory_point.positions = [p if isinstance(p, float) else p[0] for p in poses]
        efforts = [p[1] for p in poses if not isinstance(p, float)]
        if len(efforts) > 0:
            trajectory_point.effort = efforts
        trajectory_point.time_from_start = Duration(seconds=3.0).to_msg()

        trajectory_goal = FollowJointTrajectory.Goal()
        trajectory_goal.goal_time_tolerance = Duration(seconds=1.0).to_msg()
        trajectory_goal.trajectory.joint_names = joint_names
        trajectory_goal.trajectory.points.append(trajectory_point)

        self.action_client.wait_for_server()
        #self.get_logger().info('Sending goal to /follow_joint_trajectory...')
        self.goal_future = self.action_client.send_goal_async(trajectory_goal)
        self.goal_future.add_done_callback(self.response_callback)
   
   
    def response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected!')
            self.done = True
            return
        #self.get_logger().info('Goal accepted!')
        self.result_future = goal_handle.get_result_async()
        self.result_future.add_done_callback(self.result_callback)
     
        
    def result_callback(self, future):
        result = future.result().result
        status = future.result().status
        if status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error('Goal failed with status: {0}'.format(status))
        self.done = True
                     