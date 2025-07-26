import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
import cv2, os, glob, pickle, sys
from sklearn.cluster import MeanShift, KMeans, DBSCAN
from math import ceil
from utils.preprocessing_utils.projecting import detections_to_bboxes
from utils.preprocessing_utils.drawer_detection import predict_yolodrawer
import scipy.cluster.hierarchy as hcluster
import json
from utils.preprocessing_utils.projecting import project_points_bbox
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
    """
    Computes the Intersection over Union (IoU) between two arrays.

    :param array1: First array.
    :param array2: Second array.
    :return: IoU score as a float, representing the overlap between the two arrays.
    """
    intersection = np.intersect1d(array1, array2)
    union = np.union1d(array1, array2)
    iou = len(intersection) / len(union)
    return iou

def dynamic_threshold(detection_counts, n_clusters=2):
    """
    Calculates a dynamic threshold for detection count differences using k-means clustering.

    This function computes the differences between consecutive detection counts, clusters the differences 
    using k-means, and calculates a threshold based on the cluster centers. The threshold is set to the midpoint 
    between the two closest cluster centers, providing a dynamic way to separate high and low change rates in detection counts.

    :param detection_counts: List of detection counts over a sequence, used to calculate consecutive differences.
    :param n_clusters: Number of clusters for k-means. Defaults to 2.
    :return: Calculated threshold as a float, representing the midpoint between cluster centers.
    """
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
    """
    Clusters 3D detection points and organizes results with metadata.

    This function clusters detected points in 3D space based on the input detections and their coordinates.
    It optionally aligns the detections if the `aligned` flag is set. The function returns metadata about the 
    clustered detections and a list of bounding boxes in 3D space for each cluster.

    :param detections: List of detection data, containing details for each detected object or region.
    :param points_3d: Array of 3D points corresponding to the detections, typically of shape (N, 3).
    :param aligned: Flag indicating if detections should be aligned prior to clustering. Defaults to False.
    :return: tuple containing:
        - data_num: Integer representing the number of clusters or detections processed.
        - data_name: String indicating the name or identifier for the detection dataset.
        - data_file: String representing the filename or source of the detection data.
        - points_bb_3d_list: List of 3D numpy arrays, each representing the bounding box points for a cluster.
    """
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

        image = cv2.imread(data_file[idx] + ".jpg")
        width, height = image.shape[1], image.shape[0]

        _, points_bb_3d = project_points_bbox(points_3d, cam_pose, intrinsics, width, height, bbox.copy())

        centroid = np.mean(points_bb_3d, axis=0)
        dist = np.linalg.norm(points_3d - centroid, axis=1)
        closest_index = np.argmin(dist)
        closest_point = points_3d[closest_index]

        center_coord_3d.append(closest_point)
        center_index.append(closest_index)
        points_bb_3d_list.append(points_bb_3d)

    center_coord_3d = np.array(center_coord_3d)
    center_index = np.array(center_index)

    clusters = hcluster.fclusterdata(center_coord_3d, 0.15, criterion="distance")
    data_num = np.column_stack((data_num, center_coord_3d, center_index, clusters))
    return data_num, data_name, data_file, points_bb_3d_list

def cluster_images(detections):
    """
    Groups temporally close images based on detection data into clusters.

    :param detections: List of the detection entries and the corresponding number of detections for each image, used for clustering.
    :return: List of clusters.
    """
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
    """
    Selects the optimal image from each cluster based on a scoring criterion - the imae
    with the maximum number of detections in the cluster.

    :param clusters: List of clusters, where each cluster contains tuples of images and their scores.
    :return: List of optimal images, with one image selected per cluster based on the highest score.
    """
    optimal_images = []
    for cluster in clusters:
        if cluster:

            optimal_images.append(max(cluster, key=lambda x: x[1])[0])
    return optimal_images

def register_drawers(dir_path):
    """
    Registers drawers from a YOLO detection algorithm in the 3D scene.

    :param dir_path: Path to the directory containing drawer data for registration.
    :return: List of sorted indices representing registered drawers.
    """
    detections = []
    if os.path.exists(os.path.join(dir_path, 'detections.pkl')):
        with open(os.path.join(dir_path, 'detections.pkl'), 'rb') as f:
            detections = pickle.load(f)
    else:
        for image_name in sorted(glob.glob(os.path.join(dir_path, 'frame_*.jpg'))):
            img_path = os.path.join(dir_path, image_name)
            image = cv2.imread(img_path)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections += [predict_yolodrawer(image, image_name[:-4], vis_block=False)]
        with open(os.path.join(dir_path, 'detections.pkl'), 'wb') as f:
            pickle.dump(detections, f)
        
    clusters = cluster_images(detections)
    
    optimal_images = select_optimal_images(clusters)
    
    detections = [det for subdets in [detections[opt][0] for opt in optimal_images] for det in subdets]
    
    pcd_original = o3d.io.read_point_cloud(os.path.join(dir_path, 'mesh_labeled.ply'))
    bboxes_3d = detections_to_bboxes(np.asarray(pcd_original.points), detections)

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
    _ = register_drawers("/home/cvg-robotics/tim_ws/spot-compose-tim/data/prescans/24-08-01a", vis_block=True)
