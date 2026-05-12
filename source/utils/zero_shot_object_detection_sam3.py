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
from utils.drawer_detection import predict_door_yolodrawer as door_predict
from utils.object_detetion import BBox, Detection, Match
from utils.recursive_config import Config
from utils.robot_utils.basic_movement import spin_until_complete
from utils.robot_utils.basic_perception import intrinsics_from_camera
from utils.time import convert_time
from utils.vis import normalize_image, draw_boxes
from utils.llm_utils import openai_client, gemini_client
import rclpy
import io
from stretch_package.stretch_visualizer import detection_visualizer
from utils.sam3_detection_client import call_sam3, Sam3Client
from utils.docker_communication import save_files
from utils.files import prep_tmp_path


sys.path.append(os.path.abspath("/home/ws/source/sam2"))
from sam2.build_sam import build_sam2 # type: ignore
from sam2.sam2_image_predictor import SAM2ImagePredictor # type: ignore
from utils.openmask_interface import get_mask_points, get_text_similarity, select_with_clip

# Config and Paths
config = Config()
ending = config["pre_scanned_graphs"]["high_res"]
scan_path = config.get_subpath("ipad_scans")
SCAN_DIR = os.path.join(scan_path, ending)
IMG_DIR = config.get_subpath("images")

# Fixed
_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_SCORE_THRESH = 0.5

if _DEVICE.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

_PROCESSOR = None
_MODEL = None
yolo_model = None
sam2_predictor = None
_gemini_predictor = None


_SAM2_CHECKPOINT = "/home/ws/source/sam2/checkpoints/sam2.1_hiera_large.pt"
_SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Only initialize models when needed to save time and GPU memory, especially since some detections only require one model. 
def _init_owlv2():
    global _PROCESSOR, _MODEL
    if _MODEL is None:
        _PROCESSOR = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        _MODEL = Owlv2ForObjectDetection.from_pretrained(
            "google/owlv2-base-patch16-ensemble",
            torch_dtype=torch.float16 if _DEVICE.type == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        )
        _MODEL.to(_DEVICE)
        _MODEL.eval()

def _init_yolo():
    global yolo_model
    if yolo_model is None:
        yolo_model = YOLOWorld("/home/ws/source/yolov8x-worldv2.pt")
        yolo_model = yolo_model.to(_DEVICE)
        yolo_model.eval()

def _init_sam2():
    global sam2_predictor
    if sam2_predictor is None:
        sam2_predictor = SAM2ImagePredictor(build_sam2(_SAM2_MODEL_CFG, _SAM2_CHECKPOINT, device=_DEVICE.type))

# Adaptable
VIS_BLOCK = False
CLASSES = ["potted plant", "watering can", "herbs", "bottle", "pot", "pan", "cup", "plate", "bowl", "milk carton", "box", "stove", "oven",
           "football", "football plushy", "tennis ball", "image frame", "cat plushy", "shark plushy", "folder", "drawer", "door"]

def _get_gemini_predictor():
    global _gemini_predictor
    if _gemini_predictor is None:
        _gemini_predictor = gemini_client.GeminiLocationPredictor()
    return _gemini_predictor

def parse_detection_output(model_output: dict, img_width: int, img_height: int) -> tuple[bool, dict]:
    """
    Convert Gemini JSON output into (detected, detection_dict) with pixel coords.

    Model output box is [y_min, x_min, y_max, x_max] in 0–1000 normalized coords.
    Converted to pixel coords: (x1, y1, x2, y2).
    """
    try:
        detected = bool(model_output.get("detected", False))
        detection_dict = {}

        if detected:
            dd = model_output.get("detection_dict", {})
            if isinstance(dd, dict) and "box" in dd:
                y_min, x_min, y_max, x_max = dd["box"]

                x1 = int(x_min / 1000 * img_width)
                y1 = int(y_min / 1000 * img_height)
                x2 = int(x_max / 1000 * img_width)
                y2 = int(y_max / 1000 * img_height)

                detection_dict = {
                    "label": dd.get("label", ""),
                    "confidence": float(dd.get("confidence", 0.0)),
                    "box": (x1, y1, x2, y2)
                }
        return detected, detection_dict
    except Exception:
        return False, {}

def draw_and_save_detection(img_path, det, save_path):
    img = cv2.imread(img_path)
    if img is None or not det: return
    x1, y1, x2, y2 = map(int, det["box"])
    cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(img, f'{det["label"]} {det["confidence"]:.2f}',
                (x1, max(0,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    cv2.imwrite(save_path, img)
    

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
        
def save_masks(j: int, image: Image.Image, masks: np.ndarray, scores: np.ndarray, point_coords: np.ndarray=None, input_labels: np.ndarray=None, borders: bool=True, input_box: np.ndarray=None,) -> None:
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

        if input_box is not None:
            x0, y0, x1, y1 = map(int, input_box)
            rect = plt.Rectangle((x0,y0), x1-x0, y1-y0,
                                 edgecolor='yellow', facecolor='none', lw=2)
            ax.add_patch(rect)
            
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
    _init_owlv2()
    for image_file in [f for f in os.listdir(IMG_DIR) if f.startswith("frame")]:
        image = cv2.imread(os.path.join(IMG_DIR, image_file)) 
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = normalize_image(image)
        texts = [f"a photo of a {item}" for item in CLASSES]
        image_pil = Image.fromarray(image)
        
        # Detect objects
        inputs = _PROCESSOR(text=[texts], images=image_pil, return_tensors="pt")
        inputs = {k: v.to(_DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}

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

def owlv2_detect_object(obj: str, camera: str, conf: float=0.25, save_block: bool = True) -> tuple[bool, dict]:
    """
    Detect a specific object in an image using the OWL-ViT model.
    This function loads the OWL-ViT model, sets the class to detect, processes the image, and returns the detection results.

    Args:
        obj (str): Object to detect in the image
        camera (str): Camera with which the image was taken
        conf (float, optional): Confidence threshold. Defaults to 0.2.
        save_block (bool, optional): Whether to save the image with detections. Defaults to False.

    Returns:
        tuple[bool, dict]: Tuple containing a boolean indicating if the object was detected and a dictionary with detection information.
    """
    _init_owlv2()
    detected = False
    detection_dict = {}
    image_path = os.path.join(IMG_DIR, f"{camera}_image_rgb.png")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Image file not found at {image_path}. Skipping image check.")
        return False, {}

    image = np.asarray(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = normalize_image(image)
    texts = [f"a photo of a {obj}"]
    image_pil = Image.fromarray(image)

    # Detect objects
    start_time = time.time()

    inputs = _PROCESSOR(text=[texts], images=image_pil, return_tensors="pt")
    inputs = {k: v.to(_DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}

    outputs = _MODEL(**inputs)
    target_sizes = torch.Tensor([image_pil.size[::-1]])
    results = _PROCESSOR.post_process_object_detection(outputs=outputs, threshold=conf, target_sizes=target_sizes)
    
    # Get detection results
    predictions = results[0]
    scores = predictions["scores"].cpu().detach().numpy()
    labels = predictions["labels"].cpu().detach().numpy()      
    time_elapsed = time.time() - start_time
    detected = len(scores) > 0
    if detected:
        # Pick the highest confidence detection
        print(f"OWL-v2 detected {obj} with scores: {scores}, labels: {labels} in time {time_elapsed:.2f}s")
        idx = np.argmax(scores)
        box = predictions["boxes"][idx]
        label = labels[idx]
        confidence = scores[idx]
        x1, y1, x2, y2 = map(int, box)
        detection_dict = {
            'class_id': int(label),
            'label': obj,
            'confidence': float(confidence),
            'box': (x1, y1, x2, y2)
        }
        if save_block:
            img_draw = image.copy()
            img_draw = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_draw, f'{detection_dict["label"]} {confidence:.2f}', (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            cv2.imwrite(os.path.join(IMG_DIR, f"{camera}_owlv2_detection.png"), img_draw)
        if VIS_BLOCK:
            cv2.imshow("OWL-ViT Detection", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print(f"OWL-v2 did not detect {obj} in time {time_elapsed:.2f}s")
        detected = False
        detection_dict = {}
    return detected, detection_dict
     
def yolo_detect_objects() -> None:
    """
    Detect objects in images using the YOLO-World model.
    """
    _init_yolo()
    yolo_model.set_classes(CLASSES)

    for image_file in [f for f in os.listdir(IMG_DIR) if f.startswith("frame")]:
        image_path = os.path.join(IMG_DIR, image_file)
        
        # Detect objects
        results = yolo_model.predict(image_path, conf=0.2)
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

def yolo_world_detect_object(obj: str, camera: str, conf: float = 0.25, save_block: bool = False) -> tuple[bool, dict]:
    """
    Run YOLO-World detection only.
    """
    _init_yolo()
    image_path = os.path.join(IMG_DIR, f"{camera}_image_rgb.png")
    if not os.path.exists(image_path):
        return False, {}

    # Restrict to target class
    yolo_model.set_classes([obj])
    results = yolo_model.predict(image_path, conf=conf, device=0)  # run on GPU if available
    if not results or len(results[0].boxes) == 0:
        return False, {}

    # Pick highest-confidence box
    box = max(results[0].boxes, key=lambda b: float(b.conf[0]))
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    detection_dict = {
        "class_id": class_id,
        "label": yolo_model.names[class_id],
        "confidence": confidence,
        "box": (x1, y1, x2, y2)
    }

    if save_block:
        img_array = results[0].plot()
        cv2.imwrite(os.path.join(IMG_DIR, f"{camera}_yolo_detection.png"), img_array)

    return True, detection_dict

def gemini_detect_object(obj: str, camera: str, save_block: bool = False) -> tuple[bool, dict]:
    """
    Run Gemini-based detection only.
    """
    image_path = os.path.join(IMG_DIR, f"{camera}_image_rgb.png")
    if not os.path.exists(image_path):
        return False, {}

    try:
        with open(image_path, "rb") as img_file:
            img = Image.open(img_file)
            with io.BytesIO() as jpeg_buffer:
                img.convert("RGB").save(jpeg_buffer, format="JPEG", quality=85)
                image_bytes = jpeg_buffer.getvalue()

        output = _get_gemini_predictor().detect_object_in_image(image_data=image_bytes, object_name=obj)
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        detected, detection_dict = parse_detection_output(output, img_width=w, img_height=h)

        if detected and save_block:
            draw_and_save_detection(image_path, detection_dict,
                                    os.path.join(IMG_DIR, f"{camera}_gemini_detection.png"))

        return detected, detection_dict

    except Exception as e:
        print(f"Warning: Gemini detection error: {e}")
        return False, {}

def openai_detect_object(obj: str, camera: str, save_block: bool = False) -> tuple[bool, dict]:
    """
    Run OpenAI-based detection only.
    """
    image_path = os.path.join(IMG_DIR, f"{camera}_image_rgb.png")
    if not os.path.exists(image_path):
        return False, {}

    try:
        oai = openai_client.oai_client
        output = openai_client.detect_object_in_image_openai(oai, img_path=image_path, object_name=obj, model_name="gpt-4o-mini")
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        detected, detection_dict = parse_detection_output(output, img_width=w, img_height=h)

        if detected and save_block:
            draw_and_save_detection(image_path, detection_dict,
                                    os.path.join(IMG_DIR, f"{camera}_openai_detection.png"))

        return detected, detection_dict

    except Exception as e:
        print(f"Warning: OpenAI detection error: {e}")
        return False, {}

def yolo_detect_object(obj: str, camera: str, conf: float=0.25, save_block: bool = False, use_gemini: bool= True) -> tuple[bool, dict]:
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
    image_path = os.path.join(IMG_DIR, f"{camera}_image_rgb.png")
    detected, detection_dict = yolo_world_detect_object(obj, camera, 0.5, save_block)
    if not detected:
        detected, detection_dict = owlv2_detect_object(obj, camera, conf, save_block)
    if not detected:
        print("YOLO World and OWL-ViT did not detect the object, trying Gemini...")        
        detected, detection_dict = gemini_detect_object(obj, camera, save_block)
    if not detected:
        print("Gemini did not detect the object")
            


    vis = detection_visualizer.DetectionVisualizer()
    #detection_dict_p = json.loads(detection_dict)  

    vis.visualize(image_path, detections=[detection_dict])
    vis.close()
    
    return detected, detection_dict

def yolo_detect_object_old(obj: str, camera: str, conf: float=0.2, save_block: bool = False, use_gemini: bool= True) -> tuple[bool, dict]:
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
    _init_yolo()
    detected = False
    detection_dict = {}
    image_path = os.path.join(IMG_DIR, f"{camera}_image_rgb.png")

    # Always try the full object name first, then try combos if not detected
    obj_words = obj.split()
    combos = []
    if len(obj_words) > 1:
        last_word = obj_words[-1]
        for i in range(len(obj_words) - 1):
            combo = " ".join(obj_words[i:])  # from i to end
            if combo.split()[-1] == last_word:
                combos.append(combo)
        combos = list(dict.fromkeys(combos))  # remove duplicates, preserve order

    # Try full object name first
    print(f"Trying to detect {obj} with YOLO...")
    yolo_model.set_classes([obj])
    results = yolo_model.predict(os.path.join(IMG_DIR, f"{camera}_image_rgb.png"), conf=0.5)
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
        box = max(results[0].boxes[0], key=lambda b: b.conf[0])
        class_id = int(box.cls[0])
        class_label = f"{yolo_model.names[class_id]}"
        confidence = box.conf[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detection_dict = {
            'class_id': class_id,
            'label': class_label,
            'confidence': confidence,
            'box': (x1, y1, x2, y2)
        }
        print(f"YOLO Detection found for full description: {detection_dict}")
    
    if not detected:
        print("YOLO World did not detect the object, trying OWL-ViT...")
        detected, detection_dict = owlv2_detect_object(obj, camera, conf=conf, save_block=save_block)
    
    if not detected:
        print("OWL-ViT did not detect the object, trying Gemini or OpenAI...")
        image_path = os.path.join(IMG_DIR, f"{camera}_image_rgb.png")
        if os.path.exists(image_path):
            if use_gemini:
                print("Using gemini for object detection")
                try:
                    with open(image_path, "rb") as img_file:
                        img = Image.open(img_file)
                        with io.BytesIO() as jpeg_buffer:
                            img.convert("RGB").save(jpeg_buffer, format="JPEG", quality=85)
                            image_bytes = jpeg_buffer.getvalue()
                            if not image_bytes:
                                detected = False
                                detection_dict = {}
                            
                            output = _get_gemini_predictor().detect_object_in_image(image_data=image_bytes, object_name=obj)
                            img = cv2.imread(image_path)
                            h, w = img.shape[:2]
                            detected, detection_dict = parse_detection_output(output, img_width=w, img_height=h)
                            if detected:
                                draw_and_save_detection(image_path, detection_dict, os.path.join(IMG_DIR, f"{camera}_gemini_detection.png")  )
                except:
                    print(f"Warning: Detect Object with Gemini led to some exception")
                    detected = False
                    detection_dict = {}
            else:
                print("Using open ai for object detection")
                try:
                    oai = openai_client.oai_client
                    output = openai_client.detect_object_in_image_openai(oai, img_path=image_path, object_name=obj)
                    print(output)
                    img = cv2.imread(image_path)
                    h, w = img.shape[:2]
                    detected, detection_dict = parse_detection_output(output, img_width=w, img_height=h)
                    if detected:
                        draw_and_save_detection(image_path, detection_dict, os.path.join(IMG_DIR, f"{camera}_openai_detection.png")  )
                    
                except Exception as e:
                    print(f"Warning: Detect Object with OpenAI client led to some exception {e}")
                    detected = False
                    detection_dict = {}

        else:
            print(f"Warning: Image file not found at {image_path}. Skipping image check.")
            image_bytes = None
            detected = False
            detection_dict = {}
            
    if detected:
        start_vis_time = time.time()
        vis = detection_visualizer.DetectionVisualizer()
        #detection_dict_p = json.loads(detection_dict)  

        vis.visualize(image_path, detections=[detection_dict])
        vis.close()
        print(f"YOLO Visualization time: {time.time() - start_vis_time:.2f}s")
    
    return detected, detection_dict
    

def sam_detect_object(camera: str, x: int, y: int, i: int, input_box: dict = None) -> tuple[np.array, np.array, np.array]:
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
    
    _init_sam2()
    image = Image.open(os.path.join(IMG_DIR, f"{camera}_image_rgb.png"))
    start_time = time.time()
    sam2_predictor.set_image(image)
    
    if input_box is not None:
        input_box_np = np.array(input_box["box"])
        input_label = np.array([1])
        pred = sam2_predictor.predict(point_coords=None, point_labels=input_label, box=input_box_np[None,:], multimask_output=False)
        
    else:
    # Get mask, score, and logits of the detected object
        input_point = np.array([[x, y]])
        input_label = np.array([1])
        pred = sam2_predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=False)
    
    print(f"SAM2 segmentation time: {time.time() - start_time:.2f}s")
    masks, scores, logits = pred
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]
    
    save_masks(i, image, masks, scores, input_box=input_box_np, input_labels=input_label, borders=True)
    start_vis_time = time.time()
    vis = detection_visualizer.DetectionVisualizer()
    image_path = os.path.join(IMG_DIR, f"{camera}_image_rgb.png")
    vis.visualize(image_path, detections = [input_box],  masks = [masks[0]], scores=[scores[0]])
    vis.close()
    print(f"SAM2 Visualization time: {time.time() - start_vis_time:.2f}s")
    return masks[0], scores[0], logits[0]

    #if VIS_BLOCK:
        #show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)
        
    # save_masks(i, image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)
    
    # return masks[0], scores[0], logits[0]

def sam_random_detect(camera: str, i: int, num_points: int = 10) -> list[dict]:
    _init_sam2()
    image = Image.open(os.path.join(IMG_DIR, f"{camera}_image_rgb.png")).convert("RGB")
    sam2_predictor.set_image(image)
    w, h = image.size

    results = []
    for _ in range(num_points):
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        input_point = np.array([[x, y]])
        input_label = np.array([1])
        masks, scores, logits = sam2_predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )
        mask = masks[0]
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            continue
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
        crop = image.crop((x0, y0, x1, y1))
        results.append({
            "mask": mask,
            "score": scores[0],
            "logits": logits[0],
            "crop": crop
        })
    return results

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


def drawer_handle_matches(detections: list[Detection], ioa_threshold: float = 0.9) -> list[Match]:
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
        drawer_conf = drawer.conf
        drawer_bbox = drawer.bbox
        handle_bbox = handle.bbox
        
        # calculate overlap
        handle_left, handle_top, handle_right, handle_bottom = handle_bbox.xmin, handle_bbox.ymin, handle_bbox.xmax, handle_bbox.ymax
        drawer_left, drawer_top, drawer_right, drawer_bottom = drawer_bbox.xmin, drawer_bbox.ymin, drawer_bbox.xmax, drawer_bbox.ymax

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

        if handle_area <= 0:
            return 0.0, 0.0
        ioa = intersection_area / handle_area
        if ioa == 0:
            return ioa, ioa
        else:
            score = ioa_weight * ioa + drawer_conf
            return score, ioa

    drawer_detections = [det for det in detections if "door" in det.name or "drawer" in det.name]
    handle_detections = [det for det in detections if "handle" in det.name or "knob" in det.name]
    
    matching_scores = np.zeros((len(drawer_detections), len(handle_detections), 2))
    for didx, drawer_detection in enumerate(drawer_detections):
        for hidx, handle_detection in enumerate(handle_detections):
            matching_scores[didx, hidx] = np.array(matching_score(drawer_detection, handle_detection))
    drawer_idxs, handle_idxs = linear_sum_assignment(-matching_scores[..., 0])
    matches = [Match(drawer_detections[drawer_idx], handle_detections[handle_idx])
               for (drawer_idx, handle_idx) in zip(drawer_idxs, handle_idxs)
               if matching_scores[drawer_idx, handle_idx, 1] > ioa_threshold
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
    
    print(f"Handle pixel: ({x_handle}, {y_handle}), Drawer pixel: ({x_drawer}, {y_drawer})")
    
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

def sam3_detect_object(obj: str, rgb_img: np.ndarray, conf: float=0.25, save_block: bool = False) -> tuple[bool, dict]:
    sam3_client = Sam3Client()
    tmp_path = prep_tmp_path(config)
    save_data = [("image.npy", np.save, rgb_img)]
    image_path, *_ = save_files(save_data, tmp_path)
    print(
        "SAM3 caller image info: "
        f"type={type(rgb_img)}, "
        f"shape={getattr(rgb_img, 'shape', None)}, "
        f"dtype={getattr(rgb_img, 'dtype', None)}"
    )
    predictions = call_sam3(sam3_client,rgb_img, [obj])
    print("SAM3 predictions:", predictions)
    if predictions:
        best_pred = max(predictions, key=lambda p: float(p.conf))
        detection_dict = {
            'label': best_pred.name,
            'confidence': float(best_pred.conf),
            'box': (int(best_pred.bbox.xmin), int(best_pred.bbox.ymin), int(best_pred.bbox.xmax), int(best_pred.bbox.ymax))
        }
        print(f"SAM3 Detection found: {detection_dict}")
        return True, detection_dict
    else:
        print("SAM3 did not detect the object")
        return False, {}

def detect_drawer_handle_sam3(tf_node: FrameTransformer, depth_img: np.ndarray, rgb_img: np.ndarray, prompts: list[str], target_z: float | None = None) -> tuple[Pose3D, str, Pose3D]:
    
    sam3_client = Sam3Client()
    tmp_path = prep_tmp_path(config)

    save_data = [("image.npy", np.save, rgb_img)]
    image_path, *_ = save_files(save_data, tmp_path)
    print(
        "SAM3 caller image info: "
        f"type={type(rgb_img)}, "
        f"shape={getattr(rgb_img, 'shape', None)}, "
        f"dtype={getattr(rgb_img, 'dtype', None)}"
    )
    predictions = call_sam3(sam3_client,rgb_img, prompts)
    print("================== DRAWER HANDLE DETECTION ==================")
    print("Drawer-door detections:", predictions)

    if predictions:
        print("Raw SAM3 detections:")
        for det in predictions:
            bbox = det.bbox
            print(f"  - label={det.name} conf={det.conf:.3f} bbox=({bbox.xmin:.1f},{bbox.ymin:.1f},{bbox.xmax:.1f},{bbox.ymax:.1f})")
    else:
        print("Raw SAM3 detections: []")

    # Save and publish an annotated debug frame so failures can be inspected even without a live ROS image subscriber.
    # debug_img_path = os.path.join(IMG_DIR, "gripper_sam3_debug.png")
    # annotated_debug_path = os.path.join(IMG_DIR, "gripper_sam3_debug_annotated.png")
    # cv2.imwrite(debug_img_path, rgb_img)
    # vis_dets = []
    # for det in predictions:
    #     bbox = det.bbox
    #     vis_dets.append({
    #         "label": det.name,
    #         "confidence": float(det.conf),
    #         "box": [int(bbox.xmin), int(bbox.ymin), int(bbox.xmax), int(bbox.ymax)],
    #     })

    # # Also write a static annotated image to disk for deterministic debugging.
    # annotated = rgb_img.copy()
    # for det in vis_dets:
    #     x0, y0, x1, y1 = det["box"]
    #     color = (0, 255, 0) if "handle" in det["label"] or "knob" in det["label"] else (255, 180, 0)
    #     cv2.rectangle(annotated, (x0, y0), (x1, y1), color, 2)
    #     cv2.putText(
    #         annotated,
    #         f"{det['label']}:{det['confidence']:.2f}",
    #         (x0, max(0, y0 - 8)),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         0.4,
    #         color,
    #         1,
    #         cv2.LINE_AA,
    #     )
    # cv2.imwrite(annotated_debug_path, annotated)

    # vis = detection_visualizer.DetectionVisualizer(topic="annotated_image")
    # vis.visualize(debug_img_path, detections=vis_dets)
    # vis.close()
    
    ioa_threshold = 0.9  # Temporary relaxed threshold for debugging
    print(f"Using drawer-handle IOA threshold: {ioa_threshold}")
    matches = drawer_handle_matches(predictions, ioa_threshold=ioa_threshold)
    filtered_matches = [m for m in matches if (m.handle is not None and m.drawer is not None)]
    
    # Fallback: if IoA matching fails, pair the highest-confidence handle with nearest drawer center.
    if filtered_matches==[]:
        drawer_detections = [det for det in predictions if ("door" in det.name or "drawer" in det.name)]
        handle_detections = [det for det in predictions if ("handle" in det.name or "knob" in det.name)]

        # if drawer_detections and handle_detections:
        #     best_handle = max(handle_detections, key=lambda d: float(d.conf))

        #     hx = 0.5 * (best_handle.bbox.xmin + best_handle.bbox.xmax)
        #     hy = 0.5 * (best_handle.bbox.ymin + best_handle.bbox.ymax)

        #     def center_dist_sq(drawer_det):
        #         dx = 0.5 * (drawer_det.bbox.xmin + drawer_det.bbox.xmax)
        #         dy = 0.5 * (drawer_det.bbox.ymin + drawer_det.bbox.ymax)
        #         return (dx - hx) ** 2 + (dy - hy) ** 2

        #     best_drawer = min(drawer_detections, key=center_dist_sq)
        #     filtered_matches = [Match(best_drawer, best_handle)]
        #     print("Fallback match used: nearest drawer to highest-confidence handle.")

        if drawer_detections and not handle_detections:
            # Handle detections missing —
            # use the center of the highest-confidence drawer as a proxy handle position.
            best_drawer = max(drawer_detections, key=lambda d: float(d.conf))
            filtered_matches = [Match(best_drawer, best_drawer)]
            print("Fallback match used: drawer center as proxy handle position (no handle detected).")

    print("\nFiltered matches:", filtered_matches)
    if not filtered_matches:
        print("No valid handle-drawer matches found.")
        return None, None, None

    if target_z is not None:
        # Pick the match whose handle projects closest to the expected world Z height.
        # This disambiguates stacked drawers that are close in X/Y but differ in Z.
        _cam_mat = intrinsics_from_camera('/gripper_camera/color/camera_info')
        _fx, _fy = _cam_mat[0, 0], _cam_mat[1, 1]
        _cx, _cy = _cam_mat[0, 2], _cam_mat[1, 2]
        _tf = tf_node.get_tf_matrix("map", "gripper_camera_color_optical_frame")
        spin_until_complete(tf_node)

        def _handle_world_z(m):
            """
            Given pixel and depth,
            recover 3D world Z coordinate of handle center and compare to target Z
            """
            
            #find handle center pixel
            hx = int((m.handle.bbox.xmin + m.handle.bbox.xmax) // 2)
            hy = int((m.handle.bbox.ymin + m.handle.bbox.ymax) // 2)
            #check depth value at handle center pixel, return large error if no valid depth
            d = depth_img[hy, hx] / 1000.0
            if d <= 0:
                return float('inf')
            #get world coordinates of handle center pixel
            #x coordinate = 
            px = (hx - _cx) * d / _fx
            py = (hy - _cy) * d / _fy
            #get world Z coordinate of handle center pixel
            world = np.array([px, py, d, 1.0]) @ _tf.T
            return abs(world[2] - target_z)
        
        #pick the match with smallest handle Z error
        best_match = min(filtered_matches, key=_handle_world_z)
        print(f"Target Z={target_z:.3f}m — selected handle world-Z error={_handle_world_z(best_match):.3f}m")
    else:
        sorted_matches = sorted(
            filtered_matches,
            key=lambda m: ((m.handle.bbox.xmin+m.handle.bbox.xmax)//2 - rgb_img.shape[1]//2)**2 +
                          ((m.handle.bbox.ymin+m.handle.bbox.ymax)//2 - rgb_img.shape[0]//2)**2)
        best_match = sorted_matches[0]

    # Get the handle bounding box and center
    # (best_match already selected above)
    hbbox = best_match.handle.bbox
    dbbox = best_match.drawer.bbox
    
    handle_bbox = [int(hbbox.xmin), int(hbbox.ymin), int(hbbox.xmax), int(hbbox.ymax)]
    drawer_bbox = [int(dbbox.xmin),int(dbbox.ymin),int(dbbox.xmax),int(dbbox.ymax)]
    
    hxmin, hymin, hxmax, hymax = handle_bbox
    x_handle, y_handle = (hxmin + hxmax) // 2, (hymin + hymax) // 2

    # Drawer center + edges
    dxmin, dymin, dxmax, dymax = drawer_bbox
    x_drawer, y_drawer = (dxmin + dxmax) // 2, (dymin + dymax) // 2
    
    print(f"Handle pixel: ({x_handle}, {y_handle}), Drawer pixel: ({x_drawer}, {y_drawer})")
    
    # Check if handle is in the left, right or center of the image
    pos = x_handle - x_drawer
    if pos > 10:
        open_dir = "left"
        x_hinge = dxmin
    elif pos < -10:
        open_dir = "right"
        x_hinge = dxmax
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

def detect_door_handle(tf_node: FrameTransformer, depth_img: np.ndarray, rgb_img: np.ndarray) -> tuple[Pose3D, str, Pose3D]:
    """
    Detect the handle of a drawer and calculate its pose in the map frame.
    Handle and hinge are chosen to be closest to xmin or xmax of the cabinet drawer.

    Args:
        tf_node (FrameTransformer): ROS2 node for transforming frames
        depth_img (np.ndarray): Depth image
        rgb_img (np.ndarray): RGB image

    Returns:
        tuple[Pose3D, str, Pose3D]: Handle pose in map frame, opening direction, hinge pose in map frame
    """
    # Get predictions from drawer detection model
    predictions = door_predict(rgb_img, config, input_format="rgb", vis_block=False)
    print("================== DOOR HANDLE DETECTION ==================")
    print("Drawer-door detections:", predictions)
    print("##########################################################")
    matches = drawer_handle_matches(predictions)
    test_prints(matches, rgb_img)
    # Filter matches
    filtered_matches = [m for m in matches if (m.handle is not None and m.drawer is not None)]

    if not filtered_matches:
        print("No valid handle-drawer matches found.")
        return None, None, None

    # Sort by handle center closeness to drawer edge (xmin or xmax)
    sorted_matches = sorted(
        filtered_matches,
        key=lambda m: min(
            abs(((m.handle.bbox[0] + m.handle.bbox[2]) // 2) - int(m.drawer.bbox[0])),
            abs(((m.handle.bbox[0] + m.handle.bbox[2]) // 2) - int(m.drawer.bbox[2]))
        )
    )

    # Pick the best match
    best_match = sorted_matches[0]
    handle_bbox = [int(v) for v in best_match.handle.bbox]
    drawer_bbox = [int(v) for v in best_match.drawer.bbox]

    # Handle center
    hxmin, hymin, hxmax, hymax = handle_bbox
    x_handle, y_handle = (hxmin + hxmax) // 2, (hymin + hymax) // 2

    # Drawer center + edges
    dxmin, dymin, dxmax, dymax = drawer_bbox
    x_drawer, y_drawer = (dxmin + dxmax) // 2, (dymin + dymax) // 2

    print(f"Handle bbox={handle_bbox}, center=({x_handle},{y_handle})")
    print(f"Drawer bbox={drawer_bbox}, center=({x_drawer},{y_drawer})")

    # Decide hinge based on which edge handle is closest to
    if abs(x_handle - dxmin) < abs(x_handle - dxmax):
        x_hinge = dxmin
        open_dir = "left"
    else:
        x_hinge = dxmax
        open_dir = "right"
    y_hinge = y_handle

    print(f"Selected hinge pixel=({x_hinge},{y_hinge}), open_dir={open_dir}")

    # === Project handle + hinge pixels into 3D ===
    camera_matrix = intrinsics_from_camera('/gripper_camera/color/camera_info')
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    # Handle 3D
    depth = depth_img[y_handle, x_handle]
    z = depth / 1000.0
    x = (x_handle - cx) * z / fx
    y = (y_handle - cy) * z / fy
    handle_pose_gripper = np.array((x, y, z, 1.0))

    # Hinge 3D
    depth = depth_img[y_hinge, x_hinge]
    z = depth / 1000.0
    x = (x_hinge - cx) * z / fx
    y = (y_hinge - cy) * z / fy
    hinge_pose_gripper = np.array((x, y, z, 1.0))

    # === Transform into map frame ===
    tf = tf_node.get_tf_matrix("map", "gripper_camera_color_optical_frame")
    spin_until_complete(tf_node)

    handle_pose_map = tf @ handle_pose_gripper
    hinge_pose_map = tf @ hinge_pose_gripper

    handle_pose_map = Pose3D(handle_pose_map[:3])
    hinge_pose_map = Pose3D(hinge_pose_map[:3])

    print(f"Handle pose (map): {handle_pose_map}")
    print(f"Hinge pose (map): {hinge_pose_map}")

    return handle_pose_map, open_dir, hinge_pose_map

def detect_door_handle_sam3(tf_node: FrameTransformer, depth_img: np.ndarray, rgb_img: np.ndarray, prompts: list[str]) -> tuple[Pose3D, str, Pose3D]:
    
    sam3_client = Sam3Client()
    tmp_path = prep_tmp_path(config)

    save_data = [("image.npy", np.save, rgb_img)]
    image_path, *_ = save_files(save_data, tmp_path)
    predictions = call_sam3(sam3_client,rgb_img, prompts)
    print("================== DOOR HANDLE DETECTION ==================")
    print("Drawer-door detections:", predictions)
    
    matches = drawer_handle_matches(predictions)
    # Filter matches
    filtered_matches = [m for m in matches if (m.handle is not None and m.drawer is not None)]

    if not filtered_matches:
        print("No valid handle-drawer matches found.")
        return None, None, None

    # Sort by handle center closeness to drawer edge (xmin or xmax)
    sorted_matches = sorted(
        filtered_matches,
        key=lambda m: min(
            abs(((m.handle.bbox.xmin + m.handle.bbox.xmax) // 2) - int(m.drawer.bbox.xmin)),
            abs(((m.handle.bbox.xmin + m.handle.bbox.xmax) // 2) - int(m.drawer.bbox.xmax))
        )
    )
    print("\nFiltered matches sorted by handle closeness to drawer edges:", sorted_matches)
    test_prints(sorted_matches, rgb_img)

    # Pick the best match
    best_match = sorted_matches[0]
    hbbox = best_match.handle.bbox
    dbbox = best_match.drawer.bbox

    handle_bbox = [int(hbbox.xmin), int(hbbox.ymin), int(hbbox.xmax), int(hbbox.ymax)]
    drawer_bbox = [int(dbbox.xmin),int(dbbox.ymin),int(dbbox.xmax),int(dbbox.ymax)]

    # Handle center
    hxmin, hymin, hxmax, hymax = handle_bbox
    x_handle, y_handle = (hxmin + hxmax) // 2, (hymin + hymax) // 2

    # Drawer center + edges
    dxmin, dymin, dxmax, dymax = drawer_bbox
    x_drawer, y_drawer = (dxmin + dxmax) // 2, (dymin + dymax) // 2

    print(f"Handle bbox={handle_bbox}, center=({x_handle},{y_handle})")
    print(f"Drawer bbox={drawer_bbox}, center=({x_drawer},{y_drawer})")

    # Decide hinge based on which edge handle is closest to
    if abs(x_handle - dxmin) < abs(x_handle - dxmax):
        x_hinge = dxmin
        open_dir = "left"
    else:
        x_hinge = dxmax
        open_dir = "right"
    y_hinge = y_handle

    print(f"Selected hinge pixel=({x_hinge},{y_hinge}), open_dir={open_dir}")

    # === Project handle + hinge pixels into 3D ===
    camera_matrix = intrinsics_from_camera('/gripper_camera/color/camera_info')
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    # Handle 3D
    depth = depth_img[y_handle, x_handle]
    z = depth / 1000.0
    x = (x_handle - cx) * z / fx
    y = (y_handle - cy) * z / fy
    handle_pose_gripper = np.array((x, y, z, 1.0))

    # Hinge 3D
    depth = depth_img[y_hinge, x_hinge]
    z = depth / 1000.0
    x = (x_hinge - cx) * z / fx
    y = (y_hinge - cy) * z / fy
    hinge_pose_gripper = np.array((x, y, z, 1.0))

    # === Transform into map frame ===
    tf = tf_node.get_tf_matrix("map", "gripper_camera_color_optical_frame")
    spin_until_complete(tf_node)

    handle_pose_map = tf @ handle_pose_gripper
    hinge_pose_map = tf @ hinge_pose_gripper

    handle_pose_map = Pose3D(handle_pose_map[:3])
    hinge_pose_map = Pose3D(hinge_pose_map[:3])

    print(f"Handle pose (map): {handle_pose_map}")
    print(f"Hinge pose (map): {hinge_pose_map}")

    return handle_pose_map, open_dir, hinge_pose_map

def test_prints(matches, rgb_img, save_path="handle_door_match_current.png"):
    if not matches:
        print("No matches found.")
        return None

    # Already sorted outside, so just take the best
    print("Matches found: ", matches)
    best_match = matches[0]

    # Handle
    hbbox = best_match.handle.bbox
    hxmin, hymin, hxmax, hymax = map(int, (
        hbbox.xmin,
        hbbox.ymin,
        hbbox.xmax,
        hbbox.ymax
    ))
    x_handle, y_handle = (hxmin + hxmax) // 2, (hymin + hymax) // 2
    handle_conf = getattr(best_match.handle, "conf", None)

    # Drawer
    dbbox = best_match.drawer.bbox
    dxmin, dymin, dxmax, dymax = map(int, (
        dbbox.xmin,
        dbbox.ymin,
        dbbox.xmax,
        dbbox.ymax
    ))
    x_drawer, y_drawer = (dxmin + dxmax) // 2, (dymin + dymax) // 2
    drawer_conf = getattr(best_match.drawer, "conf", None)

    # Print info
    print("=== Selected Match (handle closest to drawer edge) ===")
    if handle_conf is not None:
        print(f"Handle | conf={handle_conf:.2f} | bbox=({hxmin},{hymin},{hxmax},{hymax}) | center=({x_handle},{y_handle})")
    else:
        print(f"Handle | bbox=({hxmin},{hymin},{hxmax},{hymax}) | center=({x_handle},{y_handle})")

    if drawer_conf is not None:
        print(f"Drawer | conf={drawer_conf:.2f} | bbox=({dxmin},{dymin},{dxmax},{dymax}) | center=({x_drawer},{y_drawer})")
    else:
        print(f"Drawer | bbox=({dxmin},{dymin},{dxmax},{dymax}) | center=({x_drawer},{y_drawer})")

    # Visualize
    vis = rgb_img.copy()

    # Draw handle (green)
    cv2.rectangle(vis, (hxmin, hymin), (hxmax, hymax), (0, 255, 0), 2)
    cv2.circle(vis, (x_handle, y_handle), 5, (0, 255, 0), -1)
    handle_label = f"Handle ({handle_conf:.2f})" if handle_conf is not None else "Handle"
    cv2.putText(vis, handle_label, (hxmin, max(0, hymin - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw drawer (blue)
    cv2.rectangle(vis, (dxmin, dymin), (dxmax, dymax), (255, 0, 0), 2)
    cv2.circle(vis, (x_drawer, y_drawer), 5, (255, 0, 0), -1)
    drawer_label = f"Drawer ({drawer_conf:.2f})" if drawer_conf is not None else "Drawer"
    cv2.putText(vis, drawer_label, (dxmin, max(0, dymin - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    IMG_DIR = config.get_subpath("images")
    vis_path = os.path.join(IMG_DIR, "handle_door_match_current.png")
    cv2.imwrite(vis_path, vis)
    print(f"Visualization saved to {vis_path}")

# def save_head_rgb_image():
#     rclpy.init()
#     node = rclpy.create_node('save_head_rgb_image')
#     bridge = CvBridge()
#     msg = rclpy.task.spin_until_future_complete(
#         node,
#         rclpy.task.Future(lambda: None),
#         lambda: node.create_subscription(RosImage, '/camera/color/image_raw', lambda m: setattr(node, 'img_msg', m), 1)
#     )
#     # Wait for image
#     while not hasattr(node, 'img_msg'):
#         rclpy.spin_once(node)
#     img_msg = node.img_msg
#     cv_img = bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
#     cv2.imwrite(os.path.join(IMG_DIR, "head_aligned_rgb.png"), cv_img)
#     node.destroy_node()
#     rclpy.shutdown()
def main() -> None:
    rclpy.init()

    camera = "phone"

    # save_head_rgb_image() kitchen tissue roll
    #detected, dict = owlv2_detect_object("binder", camera, save_block=True)
    detected, dict = yolo_detect_object("bathroom cleaner", camera)
    if detected:
        _, _, _ = sam_detect_object(camera, 0, 0, 0, input_box=dict)
    
    # detected = False
    # if detected == False:
    #     oai = openai_client.oai_client
    #     #result = openai_client.check_image_response_for_object(oai, os.path.join(IMG_DIR, "head_image_rgb.png"), "purple folder")
    #     result = "yes, the first word is yes"
    #     if result and isinstance(result, str) and result.strip().lower().startswith("yes"):
    #         print("The first word is 'yes'.")
    #         sam_results = sam_random_detect("head", 0, num_points=50)
    #         top_clip_results = select_with_clip(sam_results, "pringles", image_path = os.path.join(IMG_DIR, "head_image_rgb.png"), top_k=10, VIS_BLOCK=True, img_dir=IMG_DIR)
    #     else:
    #         print("The first word is not 'yes'.")
    # else:    
    #     x1, y1, x2, y2 = map(int, dict["box"])
        # _, _, _ = sam_detect_object("head", (x1+x2)/2, (y1+y2)/2, 0)
    
    
    

if __name__ == "__main__":
    main()
