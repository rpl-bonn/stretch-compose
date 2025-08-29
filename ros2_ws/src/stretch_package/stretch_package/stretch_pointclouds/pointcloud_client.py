import os
import sys
sys.path.append('/home/ws')  # Appending source file to the path

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
import time

config = Config()

class PointCloudClient(Node):

  def __init__(self):
    super().__init__('pointcloud_client')
    self.pc_saved = False
    self.future = None

    self.cli = self.create_client(GetPointcloud, 'get_pointcloud')
    while not self.cli.wait_for_service(timeout_sec=1.0):
      self.get_logger().info('Service not available, waiting...')
    self.send_request()

  def send_request(self):
    self.get_logger().info('Sending point cloud request...')
    req = GetPointcloud.Request()
    self.future = self.cli.call_async(req)
    self.get_logger().info('Request sent.')
    self.future.add_done_callback(self.handle_response)

  def handle_response(self, future):
    try:
      res = future.result()
      if res.pointcloud.data:
        self.get_logger().info('Received point cloud.')
        path = config.get_subpath('autowalk_scans')
        ending = config["pre_scanned_graphs"]["high_res"]
        self.save_pointcloud(res.pointcloud, os.path.join(path, ending))
        self.pc_saved = True
      else:
        self.get_logger().warn('No point cloud received.')
        self.pc_saved = False
    except Exception as e:
      self.get_logger().error(f'Service call failed: {e}')
      self.pc_saved = False

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
      self.get_logger().info(f'Saved point cloud to: {single_ply}.')

    except Exception as e:
      self.get_logger().error(f'Failed to save point cloud: {e}')


def ask_yes_no(prompt: str) -> bool:
  import sys
  print("[DEBUG] Entered ask_yes_no()", flush=True)

  try:
    tty = open('/dev/tty', 'r')
    src = tty
    print("[DEBUG] Using /dev/tty for input")
  except Exception as e:
    print(f"[DEBUG] Could not open /dev/tty: {e}, falling back to sys.stdin")
    src = sys.stdin

  try:
    while True:
      sys.stdout.write(f"{prompt} (Y/N): ")
      sys.stdout.flush()
      print("[DEBUG] Waiting for input...")
      line = src.readline()
      print(f"[DEBUG] Raw input read: {repr(line)}")

      if not line:
        print("[DEBUG] EOF or no input detected")
        return False

      ans = line.strip().lower()
      print(f"[DEBUG] Normalized answer: {ans}")

      if ans in ('y', 'yes'):
        print("[DEBUG] Returning True")
        return True
      if ans in ('n', 'no'):
        print("[DEBUG] Returning False")
        return False

      print("[DEBUG] Invalid input, asking again...")
  finally:
    if src is not sys.stdin:
      src.close()
      print("[DEBUG] Closed /dev/tty")


def capture_once(node):
  req = GetPointcloud.Request()
  future = node.cli.call_async(req)
  rclpy.spin_until_future_complete(node, future, node=node)  # wait here

  try:
    res = future.result()
  except Exception as e:
    node.get_logger().error(f"Service call failed: {e}")
    return False

  if res and res.pointcloud.data:
    node.get_logger().info("Received point cloud.")
    path = config.get_subpath('autowalk_scans')
    ending = config["pre_scanned_graphs"]["high_res"]
    node.save_pointcloud(res.pointcloud, os.path.join(path, ending))
    return True
  else:
    node.get_logger().warn("No point cloud received.")
    return False


def main(args=None):
  rclpy.init(args=args)
  node = PointCloudClient()
  try:
    # Spin until first capture finishes (PointCloudClient.__init__ already sent one)
    while rclpy.ok() and not node.pc_saved:
      rclpy.spin_once(node, timeout_sec=0.1)

    while rclpy.ok():
      if not ask_yes_no("Do you want to save pointcloud again?"):
        break

      node.pc_saved = False
      node.send_request()
      # Important: keep spinning until the new response arrives
      while rclpy.ok() and not node.pc_saved:
        rclpy.spin_once(node, timeout_sec=0.1)

  finally:
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
  main()
