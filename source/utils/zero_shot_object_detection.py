"""
Util functions for object detection and segmentation.
"""

from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import os
from PIL import Image
from scipy.optimize import linear_sum_assignment
import sys
import time
import torch
from transformers import Owlv2ForObjectDetection, Owlv2Processor
import traceback
from ultralytics import YOLOWorld

from stretch_package.stretch_state.frame_transformer import FrameTransformer
from utils.coordinates import Pose3D
from utils.drawer_detection import predict_yolodrawer as drawer_predict
from utils.object_detetion import BBox, Detection, Match
from utils.recursive_config import Config
from utils.robot_utils.basic_movement import spin_until_complete
from utils.robot_utils.basic_perception import intrinsics_from_camera
from utils.time import convert_time
from utils.vis import normalize_image, draw_boxes

sys.path.append(os.path.abspath("/home/ws/source/sam2"))
from sam2.build_sam import build_sam2 # type: ignore
from sam2.sam2_image_predictor import SAM2ImagePredictor # type: ignore

# Fixed
_PROCESSOR = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
_MODEL = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
_SCORE_THRESH = 0.5

# Adaptable
VIS_BLOCK = False
CLASSES = ["potted plant", "watering can", "herbs", "bottle", "pot", "pan", "cup", "plate", "bowl", "milk carton", "box", "stove", "oven",
           "football", "football plushy", "tennis ball", "image frame", "cat plushy", "shark plushy", "folder", "drawer", "door"]

# Config and Paths
config = Config()
ending = config["pre_scanned_graphs"]["high_res"]
scan_path = config.get_subpath("ipad_scans")
SCAN_DIR = os.path.join(scan_path, ending)
IMG_DIR = config.get_subpath("images")


def show_masks(image: Image.Image, masks: np.ndarray, scores: np.ndarray, point_coords: np.ndarray=None, input_labels: np.ndarray=None, borders: bool=True) -> None:
    """
    Show the masks of the detected objects on the image.
    This function displays the image with the masks overlaid, along with the sample points and their labels.

    Args:
        image (Image.Image): Image to show
        masks (np.ndarray): Masks of the detected objects
        scores (np.ndarray): Scores of the detected objects
        point_coords (np.ndarray, optional): Coordinates of sample points. Defaults to None.
        input_labels (np.ndarray, optional): Labels of detected objects. Defaults to None.
        borders (bool, optional): Whether to show the detection contours. Defaults to True.
    """
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        
        # Show mask
        color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        
        # Show contours
        if borders:
            contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
        plt.gca().imshow(mask_image)
        
        # Show points
        if point_coords is not None:
            assert input_labels is not None
            pos_points = point_coords[input_labels==1]
            neg_points = point_coords[input_labels==0]
            plt.gca().scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=375, edgecolor='white', linewidth=1.25)
            plt.gca().scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=375, edgecolor='white', linewidth=1.25) 
        
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()
        
def save_masks(j: int, image: Image.Image, masks: np.ndarray, scores: np.ndarray, point_coords: np.ndarray=None, input_labels: np.ndarray=None, borders: bool=True) -> None:
    """
    Save the masks of the detected objects on the image.
    This function saves the image with the masks overlaid, along with the sample points and their labels.

    Args:
        j (int): Index of the viewpoint
        image (Image.Image): Image to save
        masks (np.ndarray): Masks of the detected objects
        scores (np.ndarray): Scores of the detected objects
        point_coords (np.ndarray, optional): Coordinates of sample points. Defaults to None.
        input_labels (np.ndarray, optional): Labels of detected objects. Defaults to None.
        borders (bool, optional): Whether to show the detection contours. Defaults to True.
    """
    for i, (mask, score) in enumerate(zip(masks, scores)):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)

        # Show mask
        color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        # Show contours
        if borders:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image.copy(), contours, -1, (1, 1, 1, 0.5), thickness=2)
        ax.imshow(mask_image)

        # Show points
        if point_coords is not None and input_labels is not None:
            pos_points = point_coords[input_labels == 1]
            neg_points = point_coords[input_labels == 0]
            ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=375, edgecolor='white', linewidth=1.25)
            ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=375, edgecolor='white', linewidth=1.25)

        if len(scores) > 1:
            ax.set_title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        ax.axis('off')

        # Save the figure
        save_path = f"/home/ws/data/images/viewpoints/"
        #os.path.join(save_dir, f"mask_{i+1}_score_{score:.3f}.png")
        fig.savefig(save_path+f"mask_{j}", bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    
        
def draw_detection(model, image, box, camera) -> None:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls_id = int(box.cls[0])
    confidence = box.conf[0]
    cls_label = f"{model.names[cls_id]} {confidence:.2f}"
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), cv2.FILLED)
    cv2.putText(image, cls_label, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(IMG_DIR,f"{camera}_detection.png"), image)
    if VIS_BLOCK:
        cv2.imshow("Object Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

def owlv2_detect_objects() -> None:
    """
    Detect objects in images using the OWL-ViT model.
    """
    for image_file in [f for f in os.listdir(IMG_DIR) if f.startswith("frame")]:
        image = cv2.imread(os.path.join(IMG_DIR, image_file)) 
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = normalize_image(image)
        texts = [f"a photo of a {item}" for item in CLASSES]
        image_pil = Image.fromarray(image)
        
        # Detect objects
        inputs = _PROCESSOR(text=[texts], images=image_pil, return_tensors="pt")
        outputs = _MODEL(**inputs)
        target_sizes = torch.Tensor([image_pil.size[::-1]])
        results = _PROCESSOR.post_process_object_detection(outputs=outputs, threshold=_SCORE_THRESH, target_sizes=target_sizes)
        
        # Get detection results
        predictions = results[0]
        detections = []
        scores = predictions["scores"].cpu().detach().numpy()
        labels = predictions["labels"].cpu().detach().numpy()
        boxes = predictions["boxes"].cpu().detach().numpy()
        for box, score, label in zip(boxes, scores, labels):
            bbox = BBox(*box)
            detection = Detection(name=CLASSES[label], conf=score, bbox=bbox)
            detections.append(detection)
        
        if VIS_BLOCK:
            draw_boxes(image, detections)
            
        print(f"OWL-v2 {detections=}")
        
     
def yolo_detect_objects() -> None:
    """
    Detect objects in images using the YOLO-World model.
    """
    # Load model
    model = YOLOWorld("yolov8x-worldv2.pt")
    model.set_classes(CLASSES)
    
    for image_file in [f for f in os.listdir(IMG_DIR) if f.startswith("frame")]:
        image_path = os.path.join(IMG_DIR, image_file)
        
        # Detect objects
        results = model.predict(image_path, conf=0.2)
        if results and len(results[0].boxes) > 0:
            results[0].show()
            
            # Save output image with detection bounding boxes
            img_array = results[0].plot()
            output_path = os.path.join(IMG_DIR, "detections", f"yolo_detected_{image_file}.png")
            cv2.imwrite(output_path, img_array)
            
            if VIS_BLOCK:
                cv2.imshow("Object Detection", img_array)
                cv2.waitKey(0)
                cv2.destroyAllWindows() 


def yolo_detect_object(obj: str, camera: str, conf: float=0.2, save_block: bool = False) -> tuple[bool, dict]:
    """
    Detect a specific object in an image using the YOLO-World model.
    This function loads the YOLO-World model, sets the class to detect, processes the image, and returns the detection results.

    Args:
        obj (str): Object to detect in the image
        camera (str): Camera with which the image was taken
        conf (float, optional): Confidence threshold. Defaults to 0.2.
        save_block (bool, optional): Whether to save the image with detections. Defaults to False.

    Returns:
        tuple[bool, dict]: Tuple containing a boolean indicating if the object was detected and a dictionary with detection information.
    """
    detected = False
    detection_dict = {}
    
    # Load model
    model = YOLOWorld("yolov8x-worldv2.pt")
    model.set_classes([obj])
    results = model.predict(os.path.join(IMG_DIR, f"{camera}_image_rgb.png"), conf=conf)
    
    # Get, save, and show detection results
    if results and len(results[0].boxes) > 0:
        detected = True
        results[0].show()
        img_array = results[0].plot()
        
        if save_block:
            output_path = os.path.join(IMG_DIR, f"{camera}_yolo_detection.png")  
            cv2.imwrite(output_path, img_array)
             
        if VIS_BLOCK:
            cv2.imshow("Object Detection", img_array)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Save detections in dictionary
        box = max(results[0].boxes[0], key=lambda b: b.conf[0]) # Box with highest confidence
        class_id = int(box.cls[0])  # Class ID
        class_label = f"{model.names[class_id]}" # Class label
        confidence = box.conf[0]  # Confidence score
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        detection_dict={'class_id': class_id,
                            'label': class_label,
                            'confidence': confidence,
                            'box': (x1, y1, x2, y2)}
        print(f"Detection found: {detection_dict}")
                            
    return detected, detection_dict


def sam_detect_object(camera: str, x: int, y: int, i: int) -> tuple[np.array, np.array, np.array]:
    """
    Detect and segment an object in an image using the Segment Anything Model (SAM).
    This function loads the SAM model, processes the image, and returns the mask, score, and logits of the detected object.

    Args:
        camera (str): Camera with which the image was taken
        x (int): x-coordinate of the sample point
        y (int): y-coordinate of the sample point

    Returns:
        tuple[np.array, np.array, np.array]: Tuple containing the mask, score, and logits of the detected object.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # Load model and image
    sam2_checkpoint = "source/sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, sam2_checkpoint, device=device))
    image = Image.open(os.path.join(IMG_DIR, f"{camera}_image_rgb.png"))
    predictor.set_image(image)
    
    # Get mask, score, and logits of the detected object
    input_point = np.array([[x, y]])
    input_label = np.array([1])
    masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=False)
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]

    if VIS_BLOCK:
        show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)
        
    save_masks(i, image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)
    
    return masks[0], scores[0], logits[0]


def pixel2coords(x: int, y: int, depth: float, fx: float, fy: float, cx: float, cy: float) -> np.array:
    """
    Convert pixel coordinates to 3D coordinates in the camera frame.
    This function takes the pixel coordinates (x, y), depth value, and camera intrinsics (fx, fy, cx, cy)
    and returns the corresponding 3D coordinates in the camera frame.

    Args:
        x (int): x-coordinate of the pixel
        y (int): y-coordinate of the pixel
        depth (float): Depth value of the pixel
        fx (float): Focal length in x-direction
        fy (float): Focal length in y-direction
        cx (float): Principal point in x-direction
        cy (float): Principal point in y-direction

    Returns:
        np.array: 3D coordinate in the camera frame
    """
    z = depth / 1000.0
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    return np.array((y, -x, z)) # Note: Coordinates are swapped to match the camera's rotation


def get_position_from_head_detection(detection: dict, tf_node: FrameTransformer) -> tuple[Pose3D, float, float]:
    """
    Get the position of an object from a head detection.
    This function takes the bounding box coordinates of the detected object and the camera intrinsics,
    calculates the 3D position of the object in the map frame and the width and height of the object.

    Args:
        detection (dict): Detection dictionary containing the bounding box coordinates
        tf_node (FrameTransformer): ROS2 node for transforming frames

    Returns:
        tuple[Pose3D, float, float]: Tuple containing the position of the object, its width, and height
    """
    # Get camera intrinsics (Note: Values are swapped to match the camera's rotation)
    camera_matrix = intrinsics_from_camera('/camera/color/camera_info')
    fy, fx = camera_matrix[0, 0], camera_matrix[1, 1]
    cy, cx = camera_matrix[0, 2], camera_matrix[1, 2]
    
    try:
    # Minimize bounding box for median depth calculation
        x1, y1, x2, y2 = map(int, detection["box"])
        x1_new = int(0.6*x1 +0.4*x2)
        x2_new = int(0.4*x1 + 0.6*x2)
        y1_new = int(0.6*y1 +0.4*y2)
        y2_new = int(0.4*y1 + 0.6*y2)
        
        # Calculate camera position from depth image
        depth_img = cv2.imread(os.path.join(IMG_DIR, "head_image_aligned.png"), cv2.IMREAD_ANYDEPTH)
        depth_values = depth_img[y1_new:y2_new, x1_new:x2_new][depth_img[y1_new:y2_new, x1_new:x2_new] > 10]
        if depth_values.size == 0:
            x1_new = int(0.7*x1 +0.3*x2)
            x2_new = int(0.3*x1 + 0.7*x2)
            y1_new = int(0.7*y1 +0.3*y2)
            y2_new = int(0.3*y1 + 0.7*y2)
            depth_values = depth_img[y1_new:y2_new, x1_new:x2_new][depth_img[y1_new:y2_new, x1_new:x2_new] > 10]
        point_head = pixel2coords((x1+x2)//2, (y1+y2)//2, np.median(depth_values), fx, fy, cx, cy)
        
        # Transform position to map frame
        point_head = np.append(point_head, 1.0)
        tf = tf_node.get_tf_matrix("map", "camera_color_optical_frame")
        spin_until_complete(tf_node)
        point_map = point_head @ tf.T
        point_map = point_map[:3]
        print("Object position in map frame:", point_map)
        
        # Calculate width and height of object
        dim_2 = pixel2coords(x2, y2, np.median(depth_values), fx, fy, cx, cy)
        dim_1 = pixel2coords(x1, y1, np.median(depth_values), fx, fy, cx, cy)
        width, height = np.linalg.norm(-(dim_2[1]-dim_1[1])), np.linalg.norm((dim_2[0]-dim_1[0]))
    except Exception as e:
        print(f"Error in get_position_from_head_detection: {traceback.format_exc()}")
    
    return Pose3D(point_map), width, height


def get_cloud_from_gripper_detection(tf_node: FrameTransformer, mask: np.ndarray = None) -> o3d.geometry.PointCloud:
    """
    Get the point cloud of the detected object from the gripper camera.
    This function takes the mask of the detected object and the camera intrinsics,
    and calculates the 3D point cloud of the object in the map frame.

    Args:
        tf_node (FrameTransformer): ROS2 node for transforming frames
        mask (np.ndarray, optional): Mask of detected object. Defaults to None.

    Returns:
        o3d.geometry.PointCloud: Point cloud of the detected object
    """
    # Get camera intrinsics
    camera_matrix = intrinsics_from_camera('/gripper_camera/color/camera_info')
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    # Get depth and rgb image
    rgb_img = o3d.io.read_image(os.path.join(IMG_DIR, "gripper_image_rgb.png"))
    depth_img = o3d.io.read_image(os.path.join(IMG_DIR, "gripper_image_aligned.png"))
    height, width, _ = np.asarray(rgb_img).shape
    
    # Reduce depth image to mask
    if mask is not None:
        depth_img = np.where(mask, np.asarray(depth_img), 0).astype(np.uint16)
    
    # Create point cloud from RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=o3d.geometry.Image(rgb_img),
        depth=o3d.geometry.Image(depth_img),
        depth_scale=1000.0,
        depth_trunc=3.0,  # optional: max range in meters
        convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy),
        np.eye(4),
    )
    
    # Transform point cloud to map frame
    points = np.asarray(pcd.points)
    points_stack = np.hstack([points, np.ones((points.shape[0], 1))])
    tf = tf_node.get_tf_matrix("map", "gripper_camera_color_optical_frame")
    spin_until_complete(tf_node)
    tf_points = points_stack @ tf.T
    tf_points = tf_points[:, :3]
    
    pcd.points = o3d.utility.Vector3dVector(tf_points)
    return pcd


def drawer_handle_matches(detections: list[Detection]) -> list[Match]:
    """
    Match drawer and handle detections based on their bounding boxes and IOA (Intersection Over Area).
    This function takes a list of detections, filters out drawer and handle detections,
    and calculates the matching scores between them.

    Args:
        detections (list[Detection]): List of detections from the drawer detection model

    Returns:
        list[Match]: List of matches between drawer and handle detections
    """
    def matching_score(drawer: Detection, handle: Detection, ioa_weight: float = 10.0) -> tuple[float, float]:
        _, drawer_conf, _ = drawer
        _, _, drawer_bbox = drawer
        *_, handle_bbox = handle

        # calculate overlap
        handle_left, handle_top, handle_right, handle_bottom = handle_bbox
        drawer_left, drawer_top, drawer_right, drawer_bottom = drawer_bbox

        # Calculate the overlap between the bounding boxes
        overlap_left = max(handle_left, drawer_left)
        overlap_top = max(handle_top, drawer_top)
        overlap_right = min(handle_right, drawer_right)
        overlap_bottom = min(handle_bottom, drawer_bottom)

        # Calculate the area of the overlap
        overlap_width = max(0, overlap_right - overlap_left)
        overlap_height = max(0, overlap_bottom - overlap_top)

        intersection_area = overlap_width * overlap_height
        handle_area = (handle_right - handle_left) * (handle_bottom - handle_top)

        ioa = intersection_area / handle_area
        if ioa == 0:
            return ioa, ioa
        else:
            score = ioa_weight * ioa + drawer_conf
            return score, ioa

    drawer_detections = [det for det in detections if "door" in det.name or "drawer" in det.name]
    handle_detections = [det for det in detections if det.name == "handle"]

    matching_scores = np.zeros((len(drawer_detections), len(handle_detections), 2))
    for didx, drawer_detection in enumerate(drawer_detections):
        for hidx, handle_detection in enumerate(handle_detections):
            matching_scores[didx, hidx] = np.array(matching_score(drawer_detection, handle_detection))
    drawer_idxs, handle_idxs = linear_sum_assignment(-matching_scores[..., 0])
    matches = [Match(drawer_detections[drawer_idx], handle_detections[handle_idx])
               for (drawer_idx, handle_idx) in zip(drawer_idxs, handle_idxs)
               if matching_scores[drawer_idx, handle_idx, 1] > 0.9  # ioa
               ]

    for drawer_idx, drawer_detection in enumerate(drawer_detections):
        if drawer_idx not in drawer_idxs:
            matches.append(Match(drawer_detection, None))

    for handle_idx, handle_detection in enumerate(handle_detections):
        if handle_idx not in handle_idxs:
            matches.append(Match(None, handle_detection))

    return matches


def detect_handle(tf_node: FrameTransformer, depth_img: np.ndarray, rgb_img: np.ndarray) -> tuple[Pose3D, str, Pose3D]:
    """
    Detect the handle of a drawer and calculate its pose in the map frame.
    This function takes the depth and RGB images, processes them using the drawer detection model,
    and calculates the handle pose and opening direction, and hinge pose.

    Args:
        tf_node (FrameTransformer): ROS2 node for transforming frames
        depth_img (np.ndarray): Depth image
        rgb_img (np.ndarray): RGB image

    Returns:
        tuple[Pose3D, str, Pose3D]: Tuple containing the handle pose in the map frame, opening direction, and hinge pose in the map frame
    """
    # Get predictions from drawer detection model, match, filter and sort them
    predictions = drawer_predict(rgb_img, config, input_format="rgb", vis_block=False)
    matches = drawer_handle_matches(predictions)
    filtered_matches = [m for m in matches if (m.handle is not None and m.drawer is not None)]
    sorted_matches = sorted(filtered_matches, key=lambda m: ((m.handle.bbox[0]+m.handle.bbox[2])//2 - rgb_img.shape[1]//2)**2 + ((m.handle.bbox[1]+m.handle.bbox[3])//2 - rgb_img.shape[0]//2)**2)
    
    # Get the handle bounding box and center
    handle_detections = [match.handle.bbox for match in sorted_matches]
    handle_bbox = handle_detections[0]
    xmin, ymin, xmax, ymax = [int(v) for v in handle_bbox]
    x_handle, y_handle = int((xmin + xmax) // 2), int((ymin + ymax) // 2)
    
    # Get corresponding drawer bounding box and center
    drawer_detections = [match.drawer.bbox for match in sorted_matches]
    drawer_bbox = drawer_detections[0]
    xmin, ymin, xmax, ymax = [int(v) for v in drawer_bbox]
    x_drawer, y_drawer = int((xmin + xmax) // 2), int((ymin + ymax) // 2)
    
    # Check if handle is in the left, right or center of the image
    pos = x_handle - x_drawer
    if pos > 10:
        open_dir = "left"
        x_hinge = xmin
    elif pos < -10:
        open_dir = "right"
        x_hinge = xmax
    else:
        open_dir = "front"
        x_hinge = x_handle
    y_hinge = y_handle
    
    # Calculate 3D coordinates of the handle and the hinge in the map frame   
    camera_matrix = intrinsics_from_camera('/gripper_camera/color/camera_info')
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    
    depth = depth_img[y_handle, x_handle]
    z = depth / 1000.0
    x = (x_handle - cx) * z / fx
    y = (y_handle - cy) * z / fy
    handle_pose_gripper = np.array((x, y, z, 1.0))
    depth = depth_img[y_hinge, x_hinge]
    z = depth / 1000.0
    x_hinge = (x_hinge - cx) * z / fx
    y_hinge = (y_hinge - cy) * z / fy
    hinge_pose_gripper = np.array((x_hinge, y_hinge, z, 1.0))
    
    tf = tf_node.get_tf_matrix("map", "gripper_camera_color_optical_frame")
    spin_until_complete(tf_node)
    handle_pose_map = handle_pose_gripper @ tf.T
    handle_pose_map = Pose3D(handle_pose_map[:3])
    hinge_pose_map = hinge_pose_gripper @ tf.T
    hinge_pose_map = Pose3D(hinge_pose_map[:3])
    print(f"Handle pose: {handle_pose_map}")
    print(f"Hinge pose: {hinge_pose_map}")

    return handle_pose_map, open_dir, hinge_pose_map
        

def main() -> None:
    _, dict = yolo_detect_object("watering can", "gripper")
    x1, y1, x2, y2 = map(int, dict["box"])
    _, _, _ = sam_detect_object("gripper", (x1+x2)/2, (y1+y2)/2)
    

if __name__ == "__main__":
    main()
