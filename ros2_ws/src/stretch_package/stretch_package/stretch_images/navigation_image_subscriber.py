import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.wait_for_message import wait_for_message
import cv2
from cv_bridge import CvBridge

class NavigationImageSubscriber(Node):
    
    def __init__(self, topic, rotate=True, save=False, vis=False):
        super().__init__('navigation_image_subscriber_node')
        self.bridge = CvBridge()
        self.rotate = rotate # Nav camera = True
        self.save = save
        self.vis = vis
        self.image = None
        self.cv_image = None

        #self.get_logger().info(f'Waiting for messages on topic: {topic}')
        try:
            _, msg = wait_for_message(Image, self, topic, time_to_wait=50.0)
            if msg is not None:
                self.get_logger().info('Image message received!')
                self.process_image(msg)
        except Exception as e:
            self.get_logger().error(f'Error waiting for image: {e}')

    
    def process_image(self, msg):
        try:
            self.image = msg
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            if self.rotate:
                self.cv_image = cv2.rotate(self.cv_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            #self.get_logger().info('Converted Image message to OpenCV Image.') 
            if self.save:
                self.save_image()
            if self.vis:           
                self.show_image()
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')
     
     
    def save_image(self):
        image_path = '/home/ws/data/images/navigation_image.png'
        cv2.imwrite(image_path, self.cv_image) 
        #self.get_logger().info(f'Image saved at {image_path}')
    
        
    def show_image(self):
        cv2.imshow('Image', self.cv_image)
        cv2.waitKey(0)  # Wait indefinitely until a key is pressed
        self.get_logger().info('Key pressed, closing the image window.')    


def main(args=None):
    rclpy.init(args=args)
    node = NavigationImageSubscriber('/navigation_camera/image_raw', rotate=True, save=True, vis=True)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
