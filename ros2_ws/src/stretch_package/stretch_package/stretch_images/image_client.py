import rclpy
from rclpy.node import Node
from stretch_interfaces.srv import GetImage
import os
from cv_bridge import CvBridge
import cv2

class ImageClient(Node):
    
    def __init__(self):
        super().__init__('image_client')   
        self.bridge = CvBridge()       

        self.cli = self.create_client(GetImage, 'get_image')     
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')     
        self.send_request()
        
        
    def send_request(self):
        req = GetImage.Request()
        future = self.cli.call_async(req)
        self.get_logger().info('Request sent.')
        future.add_done_callback(self.handle_response)
        
        
    def handle_response(self, future):
        res = future.result()
        if res.image.data:
            self.get_logger().info('Received image.')
            #node.display_image(res.image)
            self.save_image(res.image, '/home/user/schmied1/Documents/received_images')
        else:
            self.get_logger().warn('No image received.')
        
        
    def display_image(self, ros_image):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, 'bgr8')
            cv2.imshow('Received image', cv_image)
            cv2.waitKey(0)
            self.get_logger().info('Button pressed. Window closed.')
        except Exception as e:
            self.get_logger().error(f'Failed to convert and display image: {e}')


    def save_image(self, ros_image, folder_path):
        try:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            file_name = f"image_{ros_image.header.stamp.sec}.png"
            file_path = os.path.join(folder_path, file_name)
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, 'bgr8')
            cv2.imwrite(file_path, cv_image)
            self.get_logger().info(f'Image saved at: {file_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to save image: {e}')
            
        
def main(args=None):
    rclpy.init(args=args)
    node = ImageClient()
    rclpy.spin(node)         
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()
