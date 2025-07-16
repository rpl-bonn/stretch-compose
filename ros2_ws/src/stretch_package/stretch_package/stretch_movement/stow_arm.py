from rclpy.node import Node
from std_srvs.srv import Trigger

class StowArmController(Node):
    
    def __init__(self):
        super().__init__('stow_arm_controller_node')
        self.stow_client = self.create_client(Trigger, '/stow_the_robot')
        if not self.stow_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn("Waiting for '/stow_the_robot' service.")
        #self.get_logger().info("Connected to '/stow_the_robot' service.")
        self.future = None
        
        
    def send_stow_request(self):
        request = Trigger.Request()
        self.future = self.stow_client.call_async(request)
        self.future.add_done_callback(self.response_callback)
        
        
    def response_callback(self, future):
        self.response = future.result()
        if not self.response.success:
            self.get_logger().error('Failed to stow the robot.')
