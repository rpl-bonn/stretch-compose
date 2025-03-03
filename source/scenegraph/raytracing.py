import open3d as o3d
import numpy as np
import json
import matplotlib.pyplot as plt

def parse_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        
    intrinsics = np.array(data["intrinsics"]).reshape((3, 3))
    camera_pose = np.array(data["cameraPoseARFrame"]).reshape((4, 4))

    R = camera_pose[:3, :3]
    t = camera_pose[:3, 3]

    
    extrinsics = np.eye(4)
    extrinsics[:3, :3] = R
    extrinsics[:3, 3] = t
    
    return intrinsics, extrinsics

# Main workflow
json_file_path = "/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/iPad-Scan-1/frame_00359.json"
obj_file_path = "/home/tjark/Documents/growing_scene_graphs/SceneGraph-Dataset/iPad-Scan-1/export.obj"
# bbox = np.load("tmp/bboxes.npy")[0]
# image = np.load("tmp/image.npy")

intrinsics, extrinsics = parse_json(json_file_path)
extrinsics = np.linalg.inv(extrinsics)

# R = extrinsics[:3, :3]  # Rotation part
# t = extrinsics[:3, 3]   # Translation part

# print("Intrinsic Matrix:\n", intrinsics)
# print("Extrinsic Matrix:\n", extrinsics)

mesh = o3d.io.read_triangle_mesh(obj_file_path)
mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

# image_width, image_height = image.shape[1], image.shape[0]


scene = o3d.t.geometry.RaycastingScene()
_ = scene.add_triangles(mesh)

intrinsic_tensor = o3d.core.Tensor(intrinsics, dtype=o3d.core.Dtype.Float32)
extrinsic_tensor = o3d.core.Tensor(extrinsics, dtype=o3d.core.Dtype.Float32)

# Use the intrinsic and extrinsic tensors to create rays
rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
    intrinsic_tensor, extrinsic_tensor, 1920, 1440
)



# R = extrinsics[:3, :3]
# t = extrinsics[:3, 3]

# # Inverse of the intrinsic matrix
# invK = np.linalg.inv(intrinsics)

# # Array to store the viewing directions
# viewing_directions = np.zeros((image_height, image_width, 3), dtype=np.float32)

# R_180_y = np.array([
#     [-1, 0, 0],
#     [0, 1, 0],
#     [0, 0, -1]
# ])

# for y in range(image_height):
#     for x in range(image_width):
#         px = np.array([x + 0.5, y + 0.5, 1.0], dtype=np.float32)
        
#         normalized_camera_coords = np.dot(invK, px)
#         # rotated_camera_coords = np.dot(R_180_y, normalized_camera_coords)
#         # normalized_camera_coords[2] = -normalized_camera_coords[2]
#         world_coords_direction = np.dot(R.T, normalized_camera_coords)
#         world_coords_direction /= np.linalg.norm(world_coords_direction)
        
#         viewing_directions[y, x] = world_coords_direction

# translation_vectors = np.tile(t, (image_height*image_width, 1))
# translation_vectors = translation_vectors.reshape(image_height, image_width, 3)

# rays = np.block([translation_vectors, viewing_directions])

# rays[:, :, 5] = -rays[:, :, 5]


# # for DEBUGGING: visualize the rays
# ray = rays[0, 0]
# ray_origin = ray[:3]
# ray_direction = ray[3:]
# ray_end = ray_origin + ray_direction * 10

# line_set = o3d.geometry.LineSet()
# points = [ray_origin, ray_end]
# lines = [[0, 1]]

# colors = [[1, 0, 0]]  # Red color for the ray
# line_set.points = o3d.utility.Vector3dVector(points)
# line_set.lines = o3d.utility.Vector2iVector(lines)
# line_set.colors = o3d.utility.Vector3dVector(colors)

# ray_direction[2] = -ray_direction[2]
# ray_end2 = ray_origin + ray_direction * 10



# line_set2 = o3d.geometry.LineSet()
# points2 = [ray_origin, ray_end2]
# lines2 = [[0, 1]]

# colors2 = [[1, 0, 0]]  # Red color for the ray
# line_set2.points = o3d.utility.Vector3dVector(points2)
# line_set2.lines = o3d.utility.Vector2iVector(lines2)
# line_set2.colors = o3d.utility.Vector3dVector(colors2)

# mesh_frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4, origin=[0, 0, 0])
# mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4, origin=[0, 0, 0])
# mesh_frame.transform(extrinsics)


# o3d.visualization.draw_geometries([mesh.to_legacy(), line_set, line_set2, mesh_frame, mesh_frame1])

# image creation
ans = scene.cast_rays(rays)
plt.imshow(ans['t_hit'].numpy())
plt.gca().invert_yaxis()
plt.savefig("raycasting.png")
print("whats going on")
plt.close()


def create_rays_pinhole(fov_deg, center, eye, up, width_px, height_px):
    # Compute focal length
    focal_length = 0.5 * width_px / np.tan(0.5 * (np.pi/180) * fov_deg)
    # focal_length = 0.5 * width_px / np.tan(0.5 * np.radians(fov_deg))

    # Create intrinsic matrix
    intrinsic_matrix = np.eye(3, dtype=np.float64)
    intrinsic_matrix[0, 0] = focal_length
    intrinsic_matrix[1, 1] = focal_length
    intrinsic_matrix[0, 2] = 0.5 * width_px
    intrinsic_matrix[1, 2] = 0.5 * height_px

    R = np.eye(3, dtype=np.float64)
    R[1, :] = up / np.linalg.norm(up)
    R[2, :] = center - eye
    R[2, :] = R[2, :] / np.linalg.norm(R[2, :])
    R[0, :] = np.cross(R[1, :], R[2, :])
    R[0, :] = R[0, :] / np.linalg.norm(R[0, :])
    R[1, :] = np.cross(R[2, :], R[0, :])

    t = eye

    extrinsic_matrix = np.eye(4, dtype=np.float64)
    extrinsic_matrix[:3, :3] = R.T
    extrinsic_matrix[:3, 3] = t

    # # Convert to Open3D tensors
    # intrinsic_tensor = o3d.core.Tensor(intrinsic_matrix, dtype=o3d.core.Dtype.Float64)
    # extrinsic_tensor = o3d.core.Tensor(extrinsic_matrix, dtype=o3d.core.Dtype.Float64)

    # # Use the intrinsic and extrinsic tensors to create rays
    # rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
    #     intrinsic_tensor, extrinsic_tensor, width_px, height_px
    # )

    return intrinsic_matrix, extrinsic_matrix
