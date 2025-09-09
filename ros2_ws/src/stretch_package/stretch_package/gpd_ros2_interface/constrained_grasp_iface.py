#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header, Int64
from sensor_msgs.msg import PointField, PointCloud2
from geometry_msgs.msg import Point
from gpd_ros2_msgs.msg import CloudIndexed, CloudSources
from geometry_msgs.msg import Vector3,  Transform, Vector3, Quaternion


# adjust imports to the actual package names that define your custom types
from gpd_ros2_msgs.srv import DetectConstrainedGrasps
from gpd_ros2_msgs.msg import GraspParams, CloudIndexed
import open3d as o3d
import tf_transformations
import copy
from stretch_package.gpd_ros2_interface.grasp_visualizer import GraspVisualizer, GraspTFPublisher  # optional, for RViz visualization

def _ensure_normals(pcd: o3d.geometry.PointCloud, radius=None, max_nn=30):
    if pcd.has_normals() and len(pcd.normals) == len(pcd.points):
        return
    if radius is None:
        bb = pcd.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(bb.get_extent())
        radius = 0.02 * diag if diag > 0 else 0.03
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))

def _o3d_to_cloud2_xyz_normals(pcd: o3d.geometry.PointCloud, frame_id: str, stamp: Time | None = None) -> PointCloud2:
    pts = np.asarray(pcd.points, dtype=np.float32)
    nrm = np.asarray(pcd.normals, dtype=np.float32)
    arr = np.concatenate([pts, nrm], axis=1).astype(np.float32)  # [N,6]
    msg = PointCloud2()
    msg.header = Header(frame_id=frame_id, stamp=stamp or Time())
    msg.height = 1
    msg.width = arr.shape[0]
    msg.fields = [
        PointField(name='x',        offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y',        offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z',        offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='normal_x', offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name='normal_y', offset=16, datatype=PointField.FLOAT32, count=1),
        PointField(name='normal_z', offset=20, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 24
    msg.row_step = msg.point_step * arr.shape[0]
    msg.is_dense = True
    msg.data = arr.tobytes()
    return msg

def make_cloudindexed_from_o3d(pcd_obj: o3d.geometry.PointCloud,
                               pcd_env: o3d.geometry.PointCloud,
                               frame_id: str,
                               stamp: Time | None = None,
                               camera_index: int = 0,
                               view_point_xyz=(0.0, 0.0, 0.0)) -> CloudIndexed:
    _ensure_normals(pcd_obj)
    _ensure_normals(pcd_env)
    n_obj = np.asarray(pcd_obj.points).shape[0]
    n_env = np.asarray(pcd_env.points).shape[0]
    if n_obj == 0:
        raise RuntimeError('Object cloud is empty.')

    # merge
    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(
        np.vstack([np.asarray(pcd_obj.points), np.asarray(pcd_env.points)])
    )
    pcd_all.normals = o3d.utility.Vector3dVector(
        np.vstack([np.asarray(pcd_obj.normals), np.asarray(pcd_env.normals)])
    )
    cloud2 = _o3d_to_cloud2_xyz_normals(pcd_all, frame_id=frame_id, stamp=stamp)

    # camera_source for single camera
    cam_src = [Int64(data=camera_index) for _ in range(n_obj + n_env)]
    vp = Point(x=float(view_point_xyz[0]), y=float(view_point_xyz[1]), z=float(view_point_xyz[2]))

    cs = CloudSources()
    cs.cloud = cloud2
    cs.camera_source = cam_src
    cs.view_points = [vp]

    ci = CloudIndexed()
    ci.cloud_sources = cs
    ci.indices = [Int64(data=i) for i in range(n_obj)]  # only object points
    # For verification: visualize pcd_all in Open3D, object points (indices) in red, rest in blue
    obj_indices = np.arange(n_obj)
    all_points = np.asarray(pcd_all.points)
    colors = np.tile([0.0, 0.0, 1.0], (all_points.shape[0], 1))  # blue for all
    colors[obj_indices] = [1.0, 0.0, 0.0]  # red for object points

    vis_pcd = copy.deepcopy(pcd_all)
    vis_pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([vis_pcd], window_name="pcd_all: red=obj, blue=env")
    return ci



SE3_LIMITS = {
    'lift_min': 0.02,     # m
    'lift_max': 1.5,     # m
    'wrist_pitch_max': 1.20,  # rad (≈ 69° up/down)
    'gripper_thickness': 0.06, # m, from wrist pitch to fingertips roughly
    'clearance_top': 0.05,     # m, needed vertical clearance for top grasp
}

def _pcd_to_base_points(pcd: o3d.geometry.PointCloud, T_base_from_cam: np.ndarray) -> np.ndarray:
    pts_cam = np.asarray(pcd.points, dtype=np.float64)
    if pts_cam.size == 0:
        return pts_cam
    R = T_base_from_cam[:3,:3]; t = T_base_from_cam[:3,3]
    return (pts_cam @ R.T) + t

def _aabb_with_margin(pts_base: np.ndarray, margin_xy=0.02, margin_z=0.01):
    mins = pts_base.min(axis=0)
    maxs = pts_base.max(axis=0)
    mins[:2] -= margin_xy; maxs[:2] += margin_xy
    mins[2]  -= margin_z;  maxs[2]  += margin_z
    return mins, maxs  # arrays [x,y,z]

def _estimate_table_height(pts_base: np.ndarray, bin_percentile=0.15):
    """Robust table height from env cloud in base_link: lower quantile of z."""
    if pts_base.shape[0] == 0:
        return None
    z = np.sort(pts_base[:,2])
    idx = max(0, int(len(z) * bin_percentile) - 1)
    return float(z[idx])

def _top_grasp_feasible(table_z: float, limits=SE3_LIMITS) -> bool:
    """
    Very conservative: top grasp needs the tool to come from above the object,
    which requires lift >= table_z + clearance_top and enough wrist pitch.
    If table is high near lift_max, top-down becomes infeasible.
    """
    if table_z is None:
        return False
    required_lift = table_z + limits['clearance_top']
    if required_lift > (limits['lift_max'] - 0.02):  # small reserve
        return False
    # Pitch check (approximate): if wrist can pitch ~90° down is not available (limit ~69°),
    # top approach may still work if object is low. We already guard by lift height above.
    # Keep it simple: rely on lift constraint primarily.
    return True


class ConstrainedGraspIface:
    """
    Wraps DetectConstrainedGrasps.srv and returns numpy-friendly results.
    """

    def __init__(self, node: Node,
                 srv_name: str = '/detect_constrained_grasps',
                 base_frame: str = 'base_link',
                 camera_frame: str = 'gripper_camera_color_optical_frame'):
        self.n = node
        self.base_frame = base_frame
        self.camera_frame = camera_frame
        self.cli = node.create_client(DetectConstrainedGrasps, srv_name)
        while not self.cli.wait_for_service(timeout_sec=0.5):
            node.get_logger().info(f'Waiting for {srv_name} ...')
            
        self.vis = GraspVisualizer()  # optional, for RViz visualization
        self.grasp_tf_pub = GraspTFPublisher()  # optional, to publish TF of selected grasp

    def _make_params(self,
                 workspace_base_xyzminmax,
                 approach_dir_base,
                 thresh_rad,
                 T_cam_base_rowmajor,
                 camera_positions=None,
                 enable_filtering=True):
        gp = GraspParams()

        # Approach direction (Vector3)
        vec = Vector3()
        vec.x, vec.y, vec.z = [float(v) for v in approach_dir_base]
        gp.approach_direction = vec

        # Camera positions (optional: flatten list of 3D positions)
        if camera_positions is not None:
            gp.camera_position = [float(v) for xyz in camera_positions for v in xyz]
        else:
            gp.camera_position = [0.0, 0.0, 0.0]

        # Transform camera->base (geometry_msgs/Transform)
        # Here T_cam_base_rowmajor is a 4x4 numpy matrix, row-major
        T = np.array(T_cam_base_rowmajor).reshape(4, 4)
        t = Transform()
        t.translation.x = float(T[0, 3])
        t.translation.y = float(T[1, 3])
        t.translation.z = float(T[2, 3])
        # rotation matrix -> quaternion
        q = tf_transformations.quaternion_from_matrix(T)
        t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w = [float(x) for x in q]
        gp.transform_camera2base = t

        # Workspace
        gp.workspace = [float(v) for v in workspace_base_xyzminmax]

        # Enable approach filtering and threshold
        gp.enable_approach_dir_filtering = True
        gp.approach_dir_threshold = 2.0 #float(thresh_rad)

        return gp

    def detect(self,
               pcd_obj, pcd_env,
               tf_node,                      # your FrameTransformer
               workspace_base_xyzminmax,
               approach_dir_base=(0.0, -1.0, 0.0),
               thresh_deg: float = 55.0,
               camera_frame: str | None = None):
        """
        Returns (tf_matrices[K,4,4], widths[K], scores[K]).
        - pcd_obj, pcd_env: Open3D point clouds
        - tf_node: provides get_tf_matrix(target, source)
        """
        # Frame transforms
        cam_frame = camera_frame or self.camera_frame
        T_base_from_cam = tf_node.get_tf_matrix(self.base_frame, cam_frame)  # 4x4
        T_cam_base_rowmajor = T_base_from_cam.tolist()  # server expects camera->base or base<-camera? you used camera2base

        # Build CloudIndexed
        ci = make_cloudindexed_from_o3d(
            pcd_obj=pcd_obj,
            pcd_env=pcd_env,
            frame_id=cam_frame,
            stamp=None,                # or self.get_clock().now().to_msg()
            camera_index=0,
            view_point_xyz=(0.0, 0.0, 0.0)  # camera at origin in camera frame
            )

        # Params
        gp = self._make_params(
            workspace_base_xyzminmax=workspace_base_xyzminmax,
            approach_dir_base=approach_dir_base,
            thresh_rad=np.deg2rad(thresh_deg),
            T_cam_base_rowmajor=[x for row in T_base_from_cam for x in row],
        )

        # Request (use correct fields)
        req = DetectConstrainedGrasps.Request()
        req.cloud_indexed = ci
        req.params_policy = DetectConstrainedGrasps.Request.USE_REQUEST_PARAMS   # or USE_CFG_FILE
        req.grasp_params = gp

        fut = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self.n, fut, timeout_sec=60.0)
        if not fut.done() or fut.result() is None:
            raise RuntimeError('DetectConstrainedGrasps call failed or timed out')
        resp = fut.result()

        #self.vis.publish_grasps(resp.grasp_configs)  # optional, for RViz visualization
        grasp_idx = 0
        grasp_configs = resp.grasp_configs.grasps
        for g in grasp_configs:
            self.grasp_tf_pub.publish_grasp_tf(g, self.camera_frame, f"grasp_{grasp_idx}")
            grasp_idx += 1

        # ---- Parse response into numpy arrays ----
        # Expect either: resp.hands[] with pose + width + score
        tf_mats, widths, scores = [], [], []
        if hasattr(resp, 'hands'):
            for h in resp.hands:
                # Case A: pose as geometry_msgs/Pose
                if hasattr(h, 'pose'):
                    p = h.pose.position; q = h.pose.orientation
                    R = quat_to_mat([q.x, q.y, q.z, q.w])
                    t = np.array([p.x, p.y, p.z], dtype=np.float64)
                    T = np.eye(4); T[:3,:3] = R; T[:3,3] = t
                # Case B: flattened 4x4 row-major float[16]
                elif hasattr(h, 'pose_rowmajor') and len(h.pose_rowmajor) == 16:
                    T = np.array(h.pose_rowmajor, dtype=np.float64).reshape(4,4)
                else:
                    # If GPD returns frame in camera, convert to base for downstream filters
                    if hasattr(h, 'T_cam_grasp') and len(h.T_cam_grasp) == 16:
                        T_cam_grasp = np.array(h.T_cam_grasp, dtype=np.float64).reshape(4,4)
                        T = T_base_from_cam @ T_cam_grasp
                    else:
                        continue
                w = float(getattr(h, 'width', 0.05))
                s = float(getattr(h, 'score', 0.0))
                tf_mats.append(T)
                widths.append(w)
                scores.append(s)
        else:
            # If your srv packs fields differently, adapt here
            raise RuntimeError('Response has no "hands" field; adjust parser.')

        if len(tf_mats) == 0:
            raise RuntimeError('No grasps returned by service')

        return np.stack(tf_mats, axis=0), np.array(widths), np.array(scores)
    
    # constrained_grasp_iface.py (only the detect_auto method changed;
# keep the helpers from the previous version)

    def detect_auto(
        self,
        pcd_obj: o3d.geometry.PointCloud,
        pcd_env: o3d.geometry.PointCloud,
        tf_node,
        obj_frame: str = 'map',            # <- NEW: actual frame of incoming clouds
        env_frame: str = 'map',            # <- NEW
        camera_frame: str | None = None,
        base_frame: str | None = None,
        horiz_axis_base = np.array([0.0, -1.0, 0.0]),
        top_axis_base   = np.array([0.0,  0.0, -1.0]),
        thresh_deg: float = 55.0,
        env_margin_xy: float = 0.02,
        env_margin_z: float = 0.01,
        clamp_to_reach: bool = True,
        reach_xy = (0.02, 0.49),
        reach_z  = (0.05, 1.05),
    ):
        base = base_frame or self.base_frame
        cam  = camera_frame or self.camera_frame

        # --- TFs we need ---
        T_base_from_cam = tf_node.get_tf_matrix(base, cam)   # T(base<-camera)
        T_base_from_map = np.eye(4)#tf_node.get_tf_matrix(base, 'map') # T(base<-map)
        T_cam_from_map  = np.linalg.inv(T_base_from_cam) @ T_base_from_map #tf_node.get_tf_matrix(cam,  'map') # T(camera<-map)

        # --- 1) Workspace from env in base_link ---
        if env_frame == 'map':
            pts_env_base = _pcd_to_base_points(pcd_env, T_base_from_map)
        elif env_frame == cam:
            pts_env_base = _pcd_to_base_points(pcd_env, T_base_from_cam)
        else:
            # generic: T(base<-env) = T(base<-map) @ T(map<-env)
            T_map_from_env = tf_node.get_tf_matrix('map', env_frame)
            T_base_from_env = T_base_from_map @ T_map_from_env
            pts_env_base = _pcd_to_base_points(pcd_env, T_base_from_env)

        if pts_env_base.size == 0:
            raise RuntimeError('Env PCD is empty; cannot derive workspace.')

        mins, maxs = _aabb_with_margin(pts_env_base, env_margin_xy, env_margin_z)
        if clamp_to_reach:
            mins[0] = max(mins[0], -reach_xy[1]); maxs[0] = min(maxs[0],  reach_xy[1])
            mins[1] = max(mins[1], -reach_xy[1]); maxs[1] = min(maxs[1],  reach_xy[1])
            mins[2] = max(mins[2],  reach_z[0]);  maxs[2] = min(maxs[2],  reach_z[1])
        workspace = [float(mins[0]), float(maxs[0]),
                    float(mins[1]), float(maxs[1]),
                    float(mins[2]), float(maxs[2])]
        
        workspace = [-0.876,1.276, -1.598,-0.164, 0.894,1.500]  # DEBUG

        # choose approach (top vs horizontal) from table height
        table_z = _estimate_table_height(pts_env_base, bin_percentile=0.15)
        use_top = _top_grasp_feasible(table_z, SE3_LIMITS)
        approach = top_axis_base if use_top else horiz_axis_base
        self.n.get_logger().info(
            f'[gpd] workspace(base)={np.round(workspace,3)} '
            f'table_z={table_z if table_z is not None else -1:.3f} '
            f'approach={"top" if use_top else "horizontal"}'
        )

        # --- 2) Re-express clouds in the camera frame for the server ---
        def to_cam_frame(pcd_in: o3d.geometry.PointCloud, src_frame: str):
            if src_frame == cam:
                return pcd_in
            elif src_frame == 'map':
                R = T_cam_from_map[:3,:3]; t = T_cam_from_map[:3,3]
            else:
                # T(camera<-src) = T(camera<-map) @ T(map<-src)
                T_map_from_src = tf_node.get_tf_matrix('map', src_frame)
                T_cam_from_src = T_cam_from_map @ T_map_from_src
                R = T_cam_from_src[:3,:3]; t = T_cam_from_src[:3,3]
            p = np.asarray(pcd_in.points, dtype=np.float64)
            q = (p @ R.T) + t
            out = o3d.geometry.PointCloud()
            out.points = o3d.utility.Vector3dVector(q)
            if pcd_in.has_colors():
                out.colors = pcd_in.colors
            return out

        pcd_obj_cam = to_cam_frame(pcd_obj, obj_frame)
        pcd_env_cam = to_cam_frame(pcd_env, env_frame)

        # --- 3) Call the existing detect() with derived params ---
        return self.detect(
            pcd_obj=pcd_obj_cam,
            pcd_env=pcd_env_cam,
            tf_node=tf_node,
            workspace_base_xyzminmax=workspace,
            approach_dir_base=approach.tolist(),
            thresh_deg=thresh_deg,
            camera_frame=cam,
        )


# small math helper
def quat_to_mat(qxyzw):
    x,y,z,w = qxyzw
    n = x*x + y*y + z*z + w*w
    if n < 1e-12: return np.eye(3)
    s = 2.0 / n
    X = x*s;  Y = y*s;  Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    return np.array([
        [1.0-(yY+zZ), xY-wZ,       xZ+wY      ],
        [xY+wZ,       1.0-(xX+zZ), yZ-wX      ],
        [xZ-wY,       yZ+wX,       1.0-(xX+yY)],
    ], dtype=np.float64)
