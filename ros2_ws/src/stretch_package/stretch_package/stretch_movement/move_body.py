from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose

class BaseController(Node):
    
    def __init__(self):
        super().__init__('base_controller_node')
        self.action_client = ActionClient(self, NavigateToPose, '/move_base')  
        self.goal_future = None
        self.result_future = None
        self.done = False
    
    
    def send_goal(self, px, py, ox, oy):      
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = px
        msg.pose.position.y = py
        msg.pose.position.z = 0.0
        msg.pose.orientation.x = ox
        msg.pose.orientation.y = oy
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0
        
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = msg
        
        self.action_client.wait_for_server()
        #self.get_logger().info('Sending goal to /move_base')
        self.goal_future = self.action_client.send_goal_async(goal_msg)
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
