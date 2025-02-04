import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class JointStateSubscriber(Node):
    
    def __init__(self):
        super().__init__('joint_state_subscriber_node')
        self.get_logger().info('Waiting for messages on topic: /joint_states')
        self.subscription = self.create_subscription(JointState, '/joint_states', self.jointstate_callback, 10)
        self.jointstate = None
        self.done = False
        
        
    def jointstate_callback(self, msg: JointState):
        if not self.done:
            self.jointstate = msg
            self.done = True
            self.get_logger().info(f"Received joint states:\n"
                                   f"Names: {msg.name}\n"
                                   f"Positions: {msg.position}\n"
                                   f"Efforts: {msg.effort}")


def main(args=None):
    rclpy.init(args=args)
    node = JointStateSubscriber()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()