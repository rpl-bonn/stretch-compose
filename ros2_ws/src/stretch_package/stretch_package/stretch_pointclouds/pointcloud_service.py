import rclpy
from rclpy.node import Node
from stretch_interfaces.srv import GetPointcloud
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import TransformListener, Buffer
from scipy.spatial.transform import Rotation
from rclpy.duration import Duration
import numpy as np

class PointCloudService(Node):
    
	def __init__(self):
		super().__init__('pointcloud_service')
		self.latest_pc = None
		self.tf_buffer = Buffer(cache_time=Duration(seconds=60))
		self.tf_listener = TransformListener(self.tf_buffer, self)
        
		self.pc_subscriber = self.create_subscription(PointCloud2, 'camera/depth/color/points', self.pc_callback, 10)
		self.pc_service = self.create_service(GetPointcloud, 'get_pointcloud', self.pc_server)
     
     
	def pc_callback(self, msg):
		try:
			tf = self.tf_buffer.lookup_transform(
				"map", 
				msg.header.frame_id, 
				rclpy.time.Time())
				
			t = np.eye(4)
			q = tf.transform.rotation
			x = tf.transform.translation
			t[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
			t[:3, 3] = [x.x, x.y, x.z]
		
			points_data = pc2.read_points_numpy(msg, field_names=("x", "y", "z", "rgb"), skip_nans=True)
		
			transformed_points = []
		
			for p in points_data:
				point = np.array([p[0], p[1], p[2], 1.0])
				xyz = t.dot(point)[:3]
				rgb = p[3]
				transformed_points.append((*xyz, rgb))
			
			self.latest_pc = pc2.create_cloud(
				msg.header, 
				fields = [
					PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
					PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
					PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
					PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1)
				],
				points=transformed_points)
			self.get_logger().info('Transformed point cloud to map frame.')
		
		except Exception as e:
			self.get_logger().error(f'Could not transform point cloud: {e}')
		
  				    
	def pc_server(self, req, res):
		if self.latest_pc is not None:
			res.pointcloud = self.latest_pc
			self.get_logger().info("Returning transformed point cloud.")
		else:
			self.get_logger().warn("No point cloud available.")
		return res


def main(args=None):
	rclpy.init(args=args)
	node = PointCloudService()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()
 
if __name__ == 'main':
	main()
