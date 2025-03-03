import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import cv2, os, glob, pickle, sys
from sklearn.cluster import MeanShift, KMeans, DBSCAN
from math import ceil
from scenegraph.projecting import detections_to_bboxes
from scenegraph.drawer_detection import predict_yolodrawer
from scenegraph.light_switch_detection import predict_light_switches
import scipy.cluster.hierarchy as hcluster
import json
from scenegraph.projecting import project_points_bbox
from collections import namedtuple

BBox = namedtuple("BBox", ["xmin", "ymin", "xmax", "ymax"])
Detection = namedtuple("Detection", ["file", "name", "conf", "bbox"])

def parse_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    intrinsics = np.array(data["intrinsics"]).reshape(3, 3)
    # projection_matrix = np.array(data["projectionMatrix"]).reshape(4, 4)
    camera_pose = np.array(data["cameraPoseARFrame"]).reshape(4, 4)
    return intrinsics, camera_pose

def parse_txt(file_path):
    with open(file_path, 'r') as file:
        extrinsics = file.readlines()
        extrinsics = [parts.split() for parts in extrinsics]
        extrinsics = np.array(extrinsics).astype(float)

    return extrinsics

def compute_iou(array1, array2):
    intersection = np.intersect1d(array1, array2)
    union = np.union1d(array1, array2)
    iou = len(intersection) / len(union)
    return iou

def dynamic_threshold(detection_counts, n_clusters=2):
    differences = np.array([abs(j - i) for i, j in zip(detection_counts[:-1], detection_counts[1:])]).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(differences)
    cluster_centers = np.sort(kmeans.cluster_centers_.flatten())
    
    if len(cluster_centers) > 1:
        threshold = (cluster_centers[0] + cluster_centers[1]) / 2
    else:
        threshold = cluster_centers[0]
    
    return threshold

def cluster_detections(detections, points_3d, aligned=False):
    if not detections:
        return []
    dels = []
    for idx, det in enumerate(detections):
        if det[1] == 0:
            dels.append(idx)

    detections_filtered = [item for i, item in enumerate(detections) if i not in dels]

    data_file = []
    data_name = []
    data_num = []
    for dets in detections_filtered:
        dets_per_image = dets[0]
        for det in dets_per_image:
            # data.append([det.file, det.conf, det.name, det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3]])
            data_name.append(det.name)
            data_num.append([det.conf, det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3]])
            data_file.append(det.file)

    data_num = np.array(data_num)
    data_name = np.array(data_name)
    data_file = np.array(data_file)

    center_coord_3d = []
    center_index = []
    rays_world = []
    origins_world = []
    points_bb_3d_list = []
    for idx, det in enumerate(data_num):
        u = (det[1] + det[3]) / 2
        v = (det[2] + det[4]) / 2
        bbox = det[1:5]

        if aligned:
            intrinsics, _ = parse_json(data_file[idx]+ ".json")
            cam_pose = parse_txt(data_file[idx]+ ".txt")
        else:
            intrinsics, cam_pose = parse_json(data_file[idx]+ ".json")
        # test
        # closest_point, closest_distance, closest_index, origin_world, ray_world = project_point_center(points_3d.copy(), cam_pose.copy(), intrinsics.copy(), u.copy(), v.copy())
        # center_coord_3d.append(closest_point)
        # center_index.append(closest_index)
        # rays_world.append(ray_world)
        # origins_world.append(origin_world)

        image = cv2.imread(data_file[idx] + ".jpg")
        width, height = image.shape[1], image.shape[0]

        indices_bb_3d, points_bb_3d = project_points_bbox(points_3d, cam_pose, intrinsics, width, height, bbox.copy())

        centroid = np.mean(points_bb_3d, axis=0)
        dist = np.linalg.norm(points_3d - centroid, axis=1)
        closest_index = np.argmin(dist)
        closest_point = points_3d[closest_index]

        center_coord_3d.append(closest_point)
        center_index.append(closest_index)
        points_bb_3d_list.append(points_bb_3d)



        a = 2

    center_coord_3d = np.array(center_coord_3d)
    center_index = np.array(center_index)

    # return data_num, data_name, data_file, origins_world, rays_world



    clusters = hcluster.fclusterdata(center_coord_3d, 0.15, criterion="distance")
    data_num = np.column_stack((data_num, center_coord_3d, center_index, clusters))
    return data_num, data_name, data_file, points_bb_3d_list

def cluster_images(detections):
    if not detections:
        return []
    
    detection_counts = [n for (_, n) in detections]
    
    threshold = ceil(dynamic_threshold(detection_counts))
    clusters = []
    current_cluster = []

    for index, count in enumerate(detection_counts):
        if not current_cluster or (index > 0 and abs(detection_counts[index - 1] - count) <= threshold):
            current_cluster.append((index, count))
        else:
            if current_cluster[-1][1] > 0: 
                clusters.append(current_cluster)
            current_cluster = [(index, count)]
    
    if current_cluster:
        clusters.append(current_cluster)

    return clusters

def select_optimal_images(clusters):
    optimal_images = []
    for cluster in clusters:
        if cluster:

            optimal_images.append(max(cluster, key=lambda x: x[1])[0])
    return optimal_images

def register_drawers(dir_path):
    # stores tuples containing the detected box(es) and its/their confidence(s)
    detections = []
    if os.path.exists(os.path.join(dir_path, 'detections.pkl')):
        with open(os.path.join(dir_path, 'detections.pkl'), 'rb') as f:
            detections = pickle.load(f)
    else:
        for image_name in sorted(glob.glob(os.path.join(dir_path, 'frame_*.jpg'))):
            img_path = os.path.join(dir_path, image_name)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections += [predict_yolodrawer(image, image_name[:-4], vis_block=False)]
        with open(os.path.join(dir_path, 'detections.pkl'), 'wb') as f:
            pickle.dump(detections, f)
        
    clusters = cluster_images(detections)

    # optimal_images = mean_shift_clustering(detections)
    
    optimal_images = select_optimal_images(clusters)
    
    detections = [det for subdets in [detections[opt][0] for opt in optimal_images] for det in subdets]
    
    pcd_original = o3d.io.read_point_cloud(os.path.join(dir_path, 'mesh_labeled.ply'))
    bboxes_3d = detections_to_bboxes(np.asarray(pcd_original.points), detections)

    # mesh = o3d.io.read_triangle_mesh(os.path.join(dir_path, 'textured_output.obj'))
    # bboxes_3d = detections_to_bboxes(np.asarray(mesh.vertices), detections)

    all_bbox_indices = [(np.array(bbox.get_point_indices_within_bounding_box(pcd_original.points)), conf) for bbox, conf in bboxes_3d]
    # all_bbox_indices = [(np.array(bbox.get_point_indices_within_bounding_box(mesh.vertices)), conf) for bbox, conf in bboxes_3d]

    registered_indices = []
    for indcs, conf in all_bbox_indices:     
        for idx, (reg_indcs, confidence) in enumerate(registered_indices):
            iou = compute_iou(reg_indcs, indcs)
            if iou > 0.1:  # Check if the overlap is greater than 10%
                if conf > confidence:
                    registered_indices[idx] = (indcs, conf)
                break
        else:
            registered_indices.append((indcs, conf))
    
    return [indcs for (indcs, _) in sorted(registered_indices, key=lambda x: x[1])]


def register_light_switches(dir_path, vis_block=False, transform=False):
    # stores tuples containing the detected box(es) and its/their confidence(s)
    detections = []
    if os.path.exists(os.path.join(dir_path, 'detections_lightswitch.pkl')):
        with open(os.path.join(dir_path, 'detections_lightswitch.pkl'), 'rb') as f:
            detections = pickle.load(f)
    else:
        for image_name in sorted(glob.glob(os.path.join(dir_path, 'frame_*.jpg'))):
            img_path = os.path.join(dir_path, image_name)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections += [predict_light_switches(image, image_name[:-4], vis_block=False)]
        with open(os.path.join(dir_path, 'detections_lightswitch.pkl'), 'wb') as f:
            pickle.dump(detections, f)

    pcd_original = o3d.io.read_point_cloud(os.path.join(dir_path, 'mesh_labeled.ply'))
    points = np.asarray(pcd_original.points)

    data_num, data_name, data_file, points_bb_3d_list = cluster_detections(detections, points)
    num_clusters = len(np.unique(data_num[:, -1]))
    detections = []
    test_centroids_idx = []
    for cluster in range(1, num_clusters+1):
        idx = np.where(data_num[:, -1] == cluster)
        idx_start = np.min(idx)
        det_per_cluster = data_num[data_num[:, -1] == cluster]

        optimal_detection_idx = np.argmax(det_per_cluster[:, 0]) + idx_start

        file = data_file[optimal_detection_idx]
        name = data_name[optimal_detection_idx]
        bbox = BBox(xmin=data_num[optimal_detection_idx][1], ymin=data_num[optimal_detection_idx][2],
                    xmax=data_num[optimal_detection_idx][3], ymax=data_num[optimal_detection_idx][4])
        detections.append(Detection(file=file, name=name, conf=data_num[optimal_detection_idx][0], bbox=bbox))
        test_centroids_idx.append(data_num[optimal_detection_idx][-2])

    bboxes_3d = detections_to_bboxes(np.asarray(pcd_original.points), detections, threshold=0.7)

    all_bbox_indices = [(np.array(bbox.get_point_indices_within_bounding_box(pcd_original.points)), conf) for bbox, conf in bboxes_3d]

    registered_indices = []
    for indcs, conf in all_bbox_indices:
        for idx, (reg_indcs, confidence) in enumerate(registered_indices):
            iou = compute_iou(reg_indcs, indcs)
            if iou > 0.1:  # Check if the overlap is greater than 10%
                if conf > confidence:
                    registered_indices[idx] = (indcs, conf)
                break
        else:
            registered_indices.append((indcs, conf))


    if vis_block:
        # highlight bboxes
        all_colors = np.asarray(pcd_original.colors)
        for (ind, conf) in all_bbox_indices:
            all_colors[ind] = np.random.rand(3)
        pcd_original.colors = o3d.utility.Vector3dVector(all_colors)

        # highlight normals
        # rays = []
        # length = 0.1
        # color = [1, 0, 0]
        # for bbox_3d in enumerate(bboxes_3d):
        #     points = [bbox_3d[1][0].center, bbox_3d[1][0].center + bbox_3d[1][1] * length]
        #     lines = [[0, 1]]
        #     colors = [color for _ in lines]
        #     line_set = o3d.geometry.LineSet()
        #     line_set.points = o3d.utility.Vector3dVector(points)
        #     line_set.lines = o3d.utility.Vector2iVector(lines)
        #     line_set.colors = o3d.utility.Vector3dVector(colors)
        #     rays.append(line_set)

        o3d.visualization.draw_geometries([pcd_original])

    # test
    # transform the points and plane normal to the ground frame
    # T_IG = parse_txt("/home/cvg-robotics/tim_ws/spot-compose-tim/data/prescans/24-08-05a/icp_tform_ground.txt")
    # pts = np.array([i[0].center for i in bboxes_3d]).T
    # pts = np.vstack((pts, np.ones(pts.shape[1])))
    # pts_IG = np.dot(T_IG, pts)
    # normals = np.array([i[1] for i in bboxes_3d]).T
    # normals_IG = np.dot(T_IG[:3, :3], normals)

    return [indcs for (indcs, _) in sorted(registered_indices, key=lambda x: x[1])]


def dbscan_clustering(detections):

    features = [{'image_id': id, 'num_drawers': n} for (id, n) in detections]

    # Convert detection counts to numpy array for clustering
    num_detections = np.array([dc['num_drawers'] for dc in features]).reshape(-1, 1)

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=1, min_samples=5)  # eps and min_samples can be tuned based on your data
    labels = dbscan.fit_predict(num_detections)

    # Identify the core cluster with the most images
    unique_labels, counts = np.unique(labels, return_counts=True)
    core_cluster = unique_labels[np.argmax(counts[unique_labels != -1])]  # Exclude noise label (-1)

    # Filter images based on the core cluster
    selected_indices = np.where(labels == core_cluster)[0]
    refined_detections = [detections[i] for i in selected_indices]

    # print(f"Selected {len(refined_detections)} images from the core cluster with the most detections.")

def mean_shift_clustering(detections):
    # features are only the number of detections per image
    features = np.array([np.array([i, n]) for i, (_, n) in enumerate(detections)])
    counts = np.array([n for (_, n) in detections])

    mean_shift = MeanShift()
    mean_shift.fit(features)
    labels = mean_shift.labels_

    image_indices = []
    for i in range(max(labels), -1, -1):
        indices = np.where(labels == i)[0]
        max_val = np.max(counts[indices])
        max_indexes = indices[np.where(counts[indices] > (max_val - (max_val // 4)))[0]]
        if max_indexes.size > 1:
            image_indices.extend(max_indexes.tolist())
        else:
            max_index = indices[np.where(counts[indices] == max_val)[0]]
            image_indices.extend(max_index.tolist())
            
    return image_indices

if __name__ == "__main__":
    # _ = register_drawers("/home/cvg-robotics/tim_ws/spot-compose-tim/data/prescans/24-08-01a", vis_block=True)
    _ = register_light_switches("/home/cvg-robotics/tim_ws/spot-compose-tim/data/prescans/24-08-17a", vis_block=True)
    # _ = register_light_switches_aligned(dir_path="/home/cvg-robotics/tim_ws/spot-compose-tim/data/", pcd_name= "24-08-01a")
