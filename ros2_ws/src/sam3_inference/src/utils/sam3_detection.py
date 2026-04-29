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

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

class BBox:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        
    def __repr__(self):
        return f'BBox(xmin={self.xmin}, ymin={self.ymin}, xmax={self.xmax}, ymax={self.ymax})'

class Detection:
    def __init__(self, name, conf, bbox):
        self.name = name
        self.conf = conf
        self.bbox = bbox
        
    def __repr__(self):
        return f"Detection(name='{self.name}', conf={self.conf}, bbox={self.bbox})"
   
def convert_to_detection_format(name, boxes, scores):
    """
    Converts the SAM output (boxes and scores) into a list of Detection objects with the provided class name.
    """
    detections = []
    for i in range(len(boxes)):
        bbox = boxes[i]  # Bounding box as [xmin, ymin, xmax, ymax]
        conf = scores[i]  # Confidence score
        detection = Detection(
            name=name,
            conf=conf.item(),  # Convert tensor to float
            bbox=BBox(bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item())  # Convert tensor to float
        )
        detections.append(detection)
    return detections
 
def load_model():
    bpe_path = f"{sam3_root}/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    model = build_sam3_image_model(bpe_path=bpe_path)
    print("sam3 model loaded successfully")
    return model

def sam3_inference(model, image, promt_1, promt_2, input_format: str = "rgb", vis_block: bool = False):
    # model = load_model()
    print("Inferencing with SAM3...")
    # assert image.shape[-1] == 3
    if input_format == "bgr":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # image = Image.fromarray(image)    
    image = Image.open(image).convert("RGB")
    processor = Sam3Processor(model, confidence_threshold=0.6)
    inference_state = processor.set_image(image)
    all_detections = {}
    
    # processor.reset_all_prompts(inference_state)
    inference_knob = processor.set_text_prompt(state=inference_state, prompt=promt_1)
    plot_and_save(image, inference_knob, image_name="sam3_pred_knob.png")
    all_detections = convert_to_detection_format('handle', inference_knob['boxes'], inference_knob['scores'])
    
    inference_door = processor.set_text_prompt(state=inference_state, prompt=promt_2)
    plot_and_save(image, inference_door, image_name="sam3_pred_door.png")
    all_detections += convert_to_detection_format('cabinet drawer', inference_door['boxes'], inference_door['scores'])
    
    print("All Detections:", all_detections)
    return inference_knob, inference_door

# def sam3_inference(model, image_np, prompt_1, prompt_2):

#     # Convert BGR → RGB
#     rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
#     pil_image = PILImage.fromarray(rgb)

#     processor = Sam3Processor(model, confidence_threshold=0.6)
#     inference_state = processor.set_image(pil_image)

#     detections = []

#     for prompt, label in [(prompt_1, "handle"),
#                           (prompt_2, "door")]:

#         result = processor.set_text_prompt(
#             state=inference_state,
#             prompt=prompt
#         )

#         boxes = result["boxes"]
#         scores = result["scores"]

#         for box, score in zip(boxes, scores):
#             detections.append({
#                 "class": label,
#                 "score": float(score),
#                 "bbox": box.tolist()
#             })

#     return detections


def plot_and_save(image, inference_state, image_name="sam3_pred.png"):
    print("KNOB Boxes.", inference_state['boxes'])
    print("Scores", inference_state['scores'])
    IMG_DIR = "/home/ws/data/images"
    vis_path = os.path.join(IMG_DIR, image_name)
    print(f"Detections saved to {vis_path}")
    plot_results(image, inference_state, save_path=vis_path)


def main():
    model = load_model()
    # image_path = f"{sam3_root}/assets/images/gripper_cam_3.png"
    # prompt_1 = "knob"
    # prompt_2 = "door"
    # sam3_inference(model, image_path, prompt_1, prompt_2)

def test():
    model = load_model()
    send = "Model loaded successfully"
    return send
    
if __name__ == "__main__":
    main()
    

# replace yolodrawer with sam3
# load the model at the start and pass the model for sam3_inference
# in inference use 2 text prompts and get boxes for both, then check the close mathes between them for door and handle(knob)
# How to do: do I add them in a list with name list or do it separately?