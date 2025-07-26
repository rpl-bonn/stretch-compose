import rclpy
from rclpy.node import Node
from stretch_interfaces.srv import GetPointcloud
import sensor_msgs_py.point_cloud2 as pc2
import os
from datetime import datetime, timezone
import pytz
import struct
import open3d as o3d
import numpy as np
from source.utils.recursive_config import Config

config = Config()

class PointCloudClient(Node):
    
	def __init__(self):
		super().__init__('pointcloud_client')
		self.pc_saved = False

		self.cli = self.create_client(GetPointcloud, 'get_pointcloud')
		while not self.cli.wait_for_service(timeout_sec=1.0):
			self.get_logger().info('Service not available, waiting...')
		self.send_request()


	def send_request(self):
		req = GetPointcloud.Request()
		future = self.cli.call_async(req)
		#self.get_logger().info('Request sent.')
		future.add_done_callback(self.handle_response)


	def handle_response(self, future):
		res = future.result()
		if res.pointcloud.data:
			self.get_logger().info('Received point cloud.')
			path = config.get_subpath('autowalk_scans')
			ending = config["pre_scanned_graphs"]["high_res"]
			self.save_pointcloud(res.pointcloud, os.path.join(path, ending))
			self.pc_saved = True
		else:
			self.get_logger().warn('No point cloud received.')


	def save_pointcloud(self, ros_pc2, folder_path):
		try:
			if not os.path.exists(folder_path):
				os.makedirs(folder_path)
				
			points = pc2.read_points(ros_pc2, field_names=("x", "y", "z", "rgb"), skip_nans=True)
            
			point_list = []
			for p in points:
				x, y, z, rgb = p[0], p[1], p[2], p[3]
				distance = np.sqrt(x**2 + y**2 + z**2)
				if distance <= 10.0:
					rgb_int = struct.unpack('I', struct.pack('f', rgb))[0]
					r = int((rgb_int >> 16) & 0xFF)
					g = int((rgb_int >> 8) & 0xFF)
					b = int(rgb_int & 0xFF)
					point_list.append([x, y, z, r/255.0, g/255.0, b/255.0])

			point_array = np.array(point_list)
            
			single_cloud = o3d.geometry.PointCloud()
			single_cloud.points = o3d.utility.Vector3dVector(point_array[:, :3])
			single_cloud.colors = o3d.utility.Vector3dVector(point_array[:, 3:])
			utc_time = datetime.fromtimestamp(ros_pc2.header.stamp.sec, timezone.utc)
			cet_time = utc_time.astimezone(pytz.timezone('Europe/Berlin'))
			formatted_time = cet_time.strftime('%Y%m%d_%H%M%S')
			file_name = f"pointcloud_{formatted_time}.ply"
			single_ply = os.path.join(folder_path, file_name)
			o3d.io.write_point_cloud(single_ply, single_cloud)
			#self.get_logger().info(f'Saved point cloud to: {single_ply}.')
			
		except Exception as e:
			self.get_logger().error(f'Failed to save point cloud: {e}')
            

def main(args=None):
	rclpy.init(args=args)
	node = PointCloudClient()
	while rclpy.ok():
		rclpy.spin_once(node)
		if node.pc_saved:
			break
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()
    