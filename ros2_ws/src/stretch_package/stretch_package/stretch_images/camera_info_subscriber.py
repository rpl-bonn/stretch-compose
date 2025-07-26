import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo

class CameraInfoSubscriber(Node):
    
    def __init__(self, topic):
        super().__init__('camera_info_subscriber_node')
        #self.get_logger().info(f'Waiting for messages on topic: {topic}')
        self.subscription = self.create_subscription(CameraInfo, topic, self.info_callback, 10)
        self.info = None
        self.done = False
      
        
    def info_callback(self, msg: CameraInfo):
        if not self.done:
            self.info = msg
            self.done = True
            self.get_logger().info(f"Received camera info:\n"
                                   f"Intrinsics: {msg.k}\n"
                                   f"Resolution: {msg.height}x{msg.width}")
            
            
def main(args=None):
    rclpy.init(args=args)
    node = CameraInfoSubscriber('/camera/color/camera_info')
    # node = CameraInfoSubscriber('/gripper_camera/color/camera_info')
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()