#!/usr/bin/env python3
# 2-space indentation, minimal comments
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
if not hasattr(np, 'float'):
    np.float = float
import cv2
import tf2_ros
import tf_transformations

from stretch_package.gpd_ros2_interface.constrained_grasp_iface import ConstrainedGraspIface
import open3d as o3d

from ultralytics import YOLO, YOLOE

CAM_COLOR = '/gripper_camera/color/image_rect_raw'
CAM_DEPTH = '/gripper_camera/aligned_depth_to_color/image_raw'
CAM_INFO  = '/gripper_camera/color/camera_info'
CAM_FRAME = 'gripper_camera_color_optical_frame'
BASE_FRAME = 'base_link'

PROMPT = ["blue bottle", "water bottle", "bottle"]  # adjust if needed
YOLOE_MODEL = "/home/ws/yoloe-11l-seg.pt"   # adjust if model path differs

class TFHelper:
  def __init__(self, node):
    self.buf = tf2_ros.Buffer()
    self.lst = tf2_ros.TransformListener(self.buf, node)
  def T(self, target, source, timeout=5.0):
    t = self.buf.lookup_transform(target, source, rclpy.time.Time(),
                                  rclpy.duration.Duration(seconds=timeout))
    q = t.transform.rotation; p = t.transform.translation
    R = tf_transformations.quaternion_matrix([q.x,q.y,q.z,q.w])[:3,:3]
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = [p.x,p.y,p.z]; return T
    
  def get_tf_matrix(self, target: str, source: str) -> np.ndarray:
    t = self.buf.lookup_transform(target, source, rclpy.time.Time(), rclpy.duration.Duration(seconds=5.0))
    q = t.transform.rotation
    p = t.transform.translation
    R = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])[:3,:3]
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3,3]  = np.array([p.x, p.y, p.z], dtype=np.float64)
    return T

def to_o3d(points, colors=None):
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points)
  if colors is not None:
    pcd.colors = o3d.utility.Vector3dVector(colors)
  return pcd

class BottleGraspsYOLOE(Node):
  def __init__(self):
    super().__init__('bottle_grasps_yoloe')
    self.bridge = CvBridge()
    self.K = None; self.color = None; self.depth = None
    self.create_subscription(CameraInfo, CAM_INFO, self.cb_info, 10)
    self.create_subscription(Image, CAM_COLOR, self.cb_color, 10)
    self.create_subscription(Image, CAM_DEPTH, self.cb_depth, 10)
    self.tf = TFHelper(self)
    self.model = YOLOE(YOLOE_MODEL)

  def cb_info(self, msg):
    if self.K is None:
      self.K = np.array(msg.k, dtype=np.float32).reshape(3,3)
  def cb_color(self, msg):
    if self.color is None:
      self.color = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
  def cb_depth(self, msg):
    if self.depth is None:
      d = self.bridge.imgmsg_to_cv2(msg)
      self.depth = d.astype(np.float32)
      if msg.encoding.lower() in ['16uc1','mono16']:
        self.depth *= 1e-3

  def run_once(self):
    self.get_logger().info('Waiting for RGB-D frame…')
    while rclpy.ok() and (self.K is None or self.color is None or self.depth is None):
      rclpy.spin_once(self, timeout_sec=0.1)
    img = self.color.copy(); h,w = img.shape[:2]

    print("Height and width:", h, w)

    self.model.set_classes(PROMPT, self.model.get_text_pe(PROMPT))  # only 'person' class

    results = self.model.predict(img)
    if len(results) == 0 or results[0].masks is None:
      self.get_logger().warn('No object found by YOLOE')
      return
    # Print boxes and confidences
    r = results[0]
    if r.boxes is not None:
        for i, box in enumerate(r.boxes):
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            label = r.names[cls] if hasattr(r, 'names') else str(cls)
            self.get_logger().info(
            f"Detection {i}: {label} conf={conf:.2f} box={xyxy.tolist()}"
            )

    # Print masks info
    if r.masks is not None:
        self.get_logger().info(f"Got {len(r.masks.data)} masks, shape={r.masks.data[0].shape}")

    # Use the best mask (highest conf)
    idx_best = int(np.argmax(r.boxes.conf.cpu().numpy()))
    mask = r.masks.data[idx_best].cpu().numpy().astype(np.uint8) * 255

    # Resize mask to depth image resolution
    mask_resized = cv2.resize(mask,
                              (self.depth.shape[1], self.depth.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
    mask_bool = mask_resized.astype(bool)

    # ---- Visualization: overlay mask on RGB and Depth ----
    # RGB overlay: draw mask in red
    overlay_rgb = img.copy()
    overlay_rgb[mask_resized > 0] = (0, 0, 255)  # red mask
    alpha = 0.5
    blended_rgb = cv2.addWeighted(img, 1 - alpha, overlay_rgb, alpha, 0)

    # Depth visualization
    depth_vis = cv2.normalize(self.depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_vis = cv2.applyColorMap(depth_vis.astype(np.uint8), cv2.COLORMAP_JET)
    overlay_depth = depth_vis.copy()
    overlay_depth[mask_resized > 0] = (0, 0, 255)  # red mask
    blended_depth = cv2.addWeighted(depth_vis, 1 - alpha, overlay_depth, alpha, 0)

    #cv2.imshow("RGB + Mask", blended_rgb)
    #cv2.imshow("Depth + Mask", blended_depth)
    #cv2.waitKey(1)

    fx,fy = self.K[0,0], self.K[1,1]; cx,cy = self.K[0,2], self.K[1,2]
    ys,xs = np.indices(self.depth.shape, dtype=np.float32)
    Z = self.depth; valid = (Z > 0.05) & (Z < 2.5)
    X = (xs-cx)/fx * Z; Y = (ys-cy)/fy * Z
    P = np.stack([X,Y,Z], -1).reshape(-1,3)
    C = img[...,::-1].reshape(-1,3)/255.0
    mflat = (mask_bool & valid).reshape(-1); vflat = valid.reshape(-1)

    pcd_obj = to_o3d(P[mflat], C[mflat])
    pcd_env = to_o3d(P[vflat & (~mflat)], C[vflat])
    
    # Voxelize point clouds for downsampling
    voxel_size = 0.01  # 10mm, adjust as needed
    pcd_obj = pcd_obj.voxel_down_sample(voxel_size)
    pcd_env = pcd_env.voxel_down_sample(voxel_size)
    
    
    
    # Axis-aligned bounding box in camera frame
    def get_aabb(pcd):
      pts = np.asarray(pcd.points)
      if pts.shape[0] == 0:
        return None
      xyz_min = pts.min(axis=0)
      xyz_max = pts.max(axis=0)
      return xyz_min, xyz_max

    def print_aabb(label, xyz_min, xyz_max, frame):
      print(f"{label} AABB in {frame}: "
            f"x=[{xyz_min[0]:.3f},{xyz_max[0]:.3f}] "
            f"y=[{xyz_min[1]:.3f},{xyz_max[1]:.3f}] "
            f"z=[{xyz_min[2]:.3f},{xyz_max[2]:.3f}]")

    # Camera frame
    aabb_obj_cam = get_aabb(pcd_obj)
    aabb_env_cam = get_aabb(pcd_env)
    if aabb_obj_cam:
      print_aabb("Object", *aabb_obj_cam, CAM_FRAME)
    if aabb_env_cam:
      print_aabb("Env", *aabb_env_cam, CAM_FRAME)

    # Transform to base frame
    T_cam2base = self.tf.get_tf_matrix(BASE_FRAME, CAM_FRAME)
    def transform_aabb(xyz_min, xyz_max, T):
      corners = np.array([[x, y, z] for x in [xyz_min[0], xyz_max[0]]
                                    for y in [xyz_min[1], xyz_max[1]]
                                    for z in [xyz_min[2], xyz_max[2]]])
      corners_h = np.hstack([corners, np.ones((8,1))])
      corners_tf = (T @ corners_h.T).T[:, :3]
      return corners_tf.min(axis=0), corners_tf.max(axis=0)

    if aabb_obj_cam:
      aabb_obj_base = transform_aabb(*aabb_obj_cam, T_cam2base)
      print_aabb("Object", *aabb_obj_base, BASE_FRAME)
    if aabb_env_cam:
      aabb_env_base = transform_aabb(*aabb_env_cam, T_cam2base)
      print_aabb("Env", *aabb_env_base, BASE_FRAME)
      
    # Prune environment points to those within the object's AABB (in base frame) + 20cm margin
    if aabb_obj_base:
      margin = 0.20  # 20 cm
      xyz_min, xyz_max = aabb_obj_base
      xyz_min_margin = xyz_min - margin
      xyz_max_margin = xyz_max + margin

      # Transform environment points to base frame
      env_points_cam = np.asarray(pcd_env.points)
      env_colors_cam = np.asarray(pcd_env.colors)
      env_points_h = np.hstack([env_points_cam, np.ones((env_points_cam.shape[0], 1))])
      env_points_base = (T_cam2base @ env_points_h.T).T[:, :3]

      # Mask: keep only points within the margin-extended AABB
      mask = np.all((env_points_base >= xyz_min_margin) & (env_points_base <= xyz_max_margin), axis=1)
      if env_points_cam.shape[0] == env_colors_cam.shape[0]:
        pcd_env = to_o3d(env_points_cam[mask], env_colors_cam[mask])
      else:
        pcd_env = to_o3d(env_points_cam[mask])
    
    o3d.visualization.draw_geometries([pcd_obj], window_name="Object Point Cloud")
    o3d.visualization.draw_geometries([pcd_env], window_name="Environment Point Cloud")

    iface = ConstrainedGraspIface(self,
                                  srv_name='/detect_constrained_grasps',
                                  base_frame=BASE_FRAME,
                                  camera_frame=CAM_FRAME)
    T,W,S = iface.detect_auto(pcd_obj=pcd_obj, pcd_env=pcd_env,
                              tf_node=self.tf, obj_frame=CAM_FRAME,
                              env_frame=CAM_FRAME, camera_frame=CAM_FRAME,
                              base_frame=BASE_FRAME, thresh_deg=55.0)
    if len(S)==0:
      self.get_logger().warn("No grasps found")
      return
    k = np.argsort(-S)[:5]
    for i,idx in enumerate(k):
      t = T[idx]; pos = t[:3,3]
      rpy = tf_transformations.euler_from_matrix(t[:3,:3], 'sxyz')
      self.get_logger().info(
        f'#{i+1} score={S[idx]:.3f} width={W[idx]*1000:.0f}mm '
        f'pos=[{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}] '
        f'rpy=[{rpy[0]:.2f},{rpy[1]:.2f},{rpy[2]:.2f}]'
      )

def main():
  rclpy.init()
  n = BottleGraspsYOLOE()
  n.get_logger().info("Waiting 2 seconds for TF buffer to fill…")
  rclpy.spin_once(n, timeout_sec=0.1)  # spin a bit immediately
  n.create_timer(2.0, lambda: None)    # dummy timer
  rclpy.spin_once(n, timeout_sec=2.0)  # keep spinning

  try:
    n.run_once()
  finally:
    n.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
  main()
