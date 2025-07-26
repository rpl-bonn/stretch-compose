import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from rclpy.wait_for_message import wait_for_message
import cv2
from cv_bridge import CvBridge

class CompressedImageSubscriber(Node):
    
    def __init__(self, topic, rotate=True, save=False, vis=False):
        super().__init__('compressed_image_subscriber_node')
        self.bridge = CvBridge()
        self.rotate = rotate # Head camera = True, Gripper camera = False
        self.save = save
        self.vis = vis
        self.image = None
        self.cv_image = None
        
        #self.get_logger().info(f'Waiting for messages on topic: {topic}')
        try:
            _, msg = wait_for_message(CompressedImage, self, topic, time_to_wait=50.0)
            if msg is not None:
                self.process_image(msg)
        except Exception as e:
            self.get_logger().error(f'Error waiting for image: {e}')

    
    def process_image(self, msg):
        try:
            self.image = msg
            self.cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
            if self.rotate:
                self.cv_image = cv2.rotate(self.cv_image, cv2.ROTATE_90_CLOCKWISE)
            #self.get_logger().info('Converted CompressedImage message to OpenCV Image.') 
            if self.save:
                self.save_image()
            if self.vis:   
                self.show_image()        
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')
      
            
    def save_image(self):
        if self.rotate:
            image_path = '/home/ws/data/images/head_image_compressed.png'
        else:
            image_path = '/home/ws/data/images/gripper_image_compressed.png'
        cv2.imwrite(image_path, self.cv_image) 
        #self.get_logger().info(f'Image saved at {image_path}')
    
        
    def show_image(self):
        cv2.imshow('Compressed Image', self.cv_image)
        cv2.waitKey(0)  # Wait indefinitely until a key is pressed
        self.get_logger().info('Key pressed, closing the image window.')
        

def main(args=None):
    rclpy.init(args=args)
    node = CompressedImageSubscriber('/camera/color/image_raw/compressed', rotate=True, save=True, vis=True)
    # node = CompressedImageSubscriber('/gripper_camera/color/image_rect_raw/compressed', rotate=False, save=True, vis=True)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

