import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

class OdomSubscriber(Node):
    
    def __init__(self):
        super().__init__('odom_subscriber_node')     
        #self.get_logger().info('Waiting for messages on topic: /odom')
        self.subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.odom = None
        self.done = False
         
            
    def odom_callback(self, msg: Odometry):
        if not self.done:
            self.odom = msg
            self.done = True
            self.get_logger().info(f"Received odom:\n"
                                   f"Position: {msg.pose.pose.position}\n"
                                   f"Orientation: {msg.pose.pose.orientation}")


def main(args=None):
    rclpy.init(args=args)
    node = OdomSubscriber()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()