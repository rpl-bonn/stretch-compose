import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.wait_for_message import wait_for_message
import cv2
from cv_bridge import CvBridge

class ImageSubscriber(Node):
    
    def __init__(self, vis=False):
        super().__init__('navigation_image_subscriber_node')
        self.bridge = CvBridge()
        self.vis = vis
        self.image = None

        self.get_logger().info('Waiting for messages on topic: /navigation_camera/image_raw')
        try:
            _, msg = wait_for_message(Image, self, '/navigation_camera/image_raw', time_to_wait=50.0)
            if msg is not None:
                self.get_logger().info('Image message received!')
                self.process_image(msg)
        except Exception as e:
            self.get_logger().error(f'Error waiting for image: {e}')

    
    def process_image(self, msg):
        try:
            self.image = msg
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.get_logger().info('Converted Image message to OpenCV Image.') 
            image_path = '/home/ws/data/images/navigation_image.png'
            cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite(image_path, cv_image)
            self.get_logger().info(f'Image saved at {image_path}')
            if self.vis:           
                cv2.imshow('Image', cv_image)
                cv2.waitKey(0)  # Wait indefinitely until a key is pressed
                self.get_logger().info('Key pressed, closing the image window.')
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')
        

def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()