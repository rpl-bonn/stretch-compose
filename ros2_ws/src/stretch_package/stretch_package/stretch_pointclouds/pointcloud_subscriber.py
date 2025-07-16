import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from rclpy.wait_for_message import wait_for_message
import cv2
import numpy as np

class PointCloudSubscriber(Node):
    
    def __init__(self):
        super().__init__('pointcloud_subscriber_node')
        self.pointcloud = None
     
        #self.get_logger().info('Waiting for messages on topic: /camera/depth/color/points')
        try:
            _, msg = wait_for_message(PointCloud2, self, '/camera/depth/color/points', time_to_wait=50.0)
            if msg is not None:
                self.get_logger().info('PointCloud2 message received!')
                #self.process_pointcloud(msg)
            else:
                self.get_logger().warn('No PointCloud2 message received within the timeout.')
        except Exception as e:
            self.get_logger().error(f'Error waiting for point cloud: {e}')
    
    
    def process_pointcloud(self, msg):
        try:
            self.pointcloud = msg
            points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
            if not points:
                self.get_logger().warn('No valid points found in the PointCloud2 message')
                return
        
            z_values = np.array([point[2] for point in points])
            z_min, z_max = np.min(z_values), np.max(z_values)
            self.get_logger().info(f'Min depth: {z_min}, Max depth: {z_max}')
            
            z_normalized = 255 * (z_values - z_min) / (z_max - z_min)
            z_normalized = z_normalized.astype(np.uint8)
            
            depth_image = z_normalized.reshape((1, len(z_normalized)))
            #self.get_logger().info('Converted PointCloud2 message to OpenCV Image.')     
            cv2.imshow('Depth Image', depth_image)
            cv2.waitKey(0) # Wait indefinitely until a key is pressed
            self.get_logger().info('Key pressed, closing the image window.') 
              
        except Exception as e:
            self.get_logger().error(f'Error processing point cloud: {e}')
     
     
def main(args=None):
    rclpy.init(args=args)
    node = PointCloudSubscriber()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

