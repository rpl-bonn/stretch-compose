import rclpy
from rclpy.node import Node
from stretch_interfaces.srv import GetImage
from sensor_msgs.msg import Image

class ImageService(Node):
    
	def __init__(self):
		super().__init__('image_service')
		self.latest_img = None
        
		self.img_subscriber = self.create_subscription(Image, 'camera/color/image_raw', self.img_callback, 10)
		self.img_service = self.create_service(GetImage, 'get_image', self.img_server)
        
        
	def img_callback(self, msg):
		self.latest_img = msg
        
        
	def img_server(self, req, res):
		if self.latest_img is not None:
			res.image = self.latest_img
			self.get_logger().info("Returning latest image.")
		else:
			self.get_logger().warn("No image available.")
		return res


def main(args=None):
	rclpy.init(args=args)
	node = ImageService()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()
 
if __name__ == 'main':
	main()
