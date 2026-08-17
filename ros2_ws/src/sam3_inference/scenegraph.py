import glob
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sam3
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import draw_box_on_image, normalize_bbox, plot_results
import torch

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
print(f"#######################: {sam3_root}")

class BBox:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

    def __repr__(self):
        return f'BBox(xmin={self.xmin}, ymin={self.ymin}, xmax={self.xmax}, ymax={self.ymax})'
    
    def __reduce__(self):
        return (BBox, (self.xmin, self.ymin, self.xmax, self.ymax))

class Detection:
    def __init__(self, file, name, conf, bbox):
        self.file = file
        self.name = name
        self.conf = conf
        self.bbox = bbox

    def __repr__(self):
        return f"Detection(file='{self.file}', name='{self.name}', conf={self.conf}, bbox={self.bbox})"
    
    def __reduce__(self):
        return (Detection, (self.file, self.name, self.conf, self.bbox))

# ------------------------
# SAM3 Utilities
# ------------------------
def convert_to_detection_format(image_path, name, boxes, scores):
    detections = []
    for i in range(len(boxes)):
        bbox = boxes[i]
        conf = scores[i]
        detections.append(
            Detection(
                file=image_path[:-4],
                name=name,
                conf=float(conf.item()),
                bbox=BBox(float(bbox[0].item()), float(bbox[1].item()), float(bbox[2].item()), float(bbox[3].item()))
            )
        )
    return detections

def load_sam3_model():
    bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(bpe_path=bpe_path)
    return model

def run_sam3_on_image(model, image_path, prompts=["cabinet door", "drawer", "door", "pillow"]):
    image = Image.open(image_path).convert("RGB")
    processor = Sam3Processor(model, confidence_threshold=0.6)
    state = processor.set_image(image)  # must return a valid state object
    if state is None:
        raise ValueError("processor.set_image returned None; make sure image is a PIL Image.")

    all_detections = []
    for prompt in prompts:
        result = processor.set_text_prompt(state=state, prompt=prompt)
        all_detections += convert_to_detection_format(image_path, prompt, result['boxes'], result['scores'])
    
    return all_detections, len(all_detections)

# ------------------------
# Main Function
# ------------------------
def register_drawers(dir_path):
    """
    Registers drawers using SAM3 detections for all frames in the directory.

    :param dir_path: Path to folder with frames.
    :return: List of tuples in the format [(list_of_detections, count), ...]
    """

    print(f"DIRECTORY PATH: {dir_path}")
    DETECTIONS = []

    # Load precomputed detections if they exist
    pkl_path = os.path.join(dir_path, 'detections_sam3.pkl')
    # if os.path.exists(pkl_path):
    #     with open(pkl_path, 'rb') as f:
    #         DETECTIONS = pickle.load(f)
    # else:
    model = load_sam3_model()
    frame_files = sorted(glob.glob(os.path.join(dir_path, 'frame_*.jpg')))
    for frame_path in frame_files:
        detections_list, count = run_sam3_on_image(model, frame_path)
        DETECTIONS.append((detections_list, count))
        # Save detections for next time
    with open(pkl_path, 'wb') as f:
        pickle.dump(DETECTIONS, f)

    txt_path = os.path.join(dir_path, 'detections_sam3.txt')
    with open(txt_path, 'w') as f:
        f.write(f"Total frames processed: {len(DETECTIONS)}\n")
        f.write(f"DETECTIONS: {DETECTIONS}\n")
            
    print('##############################')
    print(f"Total frames processed: {len(DETECTIONS)}")
    print('DETECTIONS: ', DETECTIONS)
    print('##############################')

    return DETECTIONS

# ------------------------
# Example Usage
# ------------------------
# folder = "/home/ws/data/ipad_scans/2026_04_02"
folder = "/home/ws/data/ipad_scans/2026_04_04"
all_detections = register_drawers(folder)

print(f"All Detections: {all_detections}")