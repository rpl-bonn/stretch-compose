import os
import cv2
import torch
from PIL import Image

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results



class BBox:
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin = float(xmin)
        self.ymin = float(ymin)
        self.xmax = float(xmax)
        self.ymax = float(ymax)

    def __repr__(self):
        return (f"BBox(xmin={self.xmin}, ymin={self.ymin}, xmax={self.xmax}, ymax={self.ymax})")
    
    def __reduce__(self):
        return (BBox, (self.xmin, self.ymin, self.xmax, self.ymax))

class Detection:
    def __init__(self, name, conf, bbox: BBox, file: str):
        self.file = file
        self.name = name
        self.conf = float(conf)
        self.bbox = bbox

    def __repr__(self):
        return (f"Detection(file='{self.file}', name='{self.name}', conf={self.conf}, bbox={self.bbox})")
    
    def __reduce__(self):
        return (Detection, (self.file, self.name, self.conf, self.bbox))

class Sam3SceneInference:

    def __init__(self, confidence_threshold=0.6, device="cuda"):
        """
        Initializes SAM3 model once.
        Should be created once in your ROS service constructor.
        """

        # Detect device
        self.device = device if torch.cuda.is_available() else "cpu"

        self.sam3_root = os.path.join(
            os.path.dirname(sam3.__file__), ".."
        )

        self.model = self.load_model()
        self.model.to(self.device)
        self.model.eval()

        self.processor = Sam3Processor(
            self.model,
            confidence_threshold=confidence_threshold
        )

        print("SAM3 processor UP & Running successfully")

    def load_model(self):
        bpe_path = f"{self.sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
        model = build_sam3_image_model(bpe_path=bpe_path)
        print("SAM3 model loaded successfully")
        return model
        
    def infer(self, image, prompts, file, input_format="bgr", visualize=False):
        """
        prompts: list of dicts
            Example:
            [
                {"prompt": "handle", "label": "handle"},
                {"prompt": "door", "label": "cabinet drawer"}
            ]
        """

        if input_format == "bgr":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        else:
            image = Image.fromarray(image)

        inference_state = self.processor.set_image(image)

        all_detections = []

        for text_prompt in prompts:

            result = self.processor.set_text_prompt(
                state=inference_state,
                prompt=text_prompt
            )

            all_detections += self._convert_to_detection_format(
                text_prompt,
                result["boxes"],
                result["scores"],
                file
            )

            if visualize:
                self._plot(image, result, f"sam3_{text_prompt}_1.png")

        return all_detections


    def _convert_to_detection_format(self, name, boxes, scores, file):
        detections = []

        for i in range(len(boxes)):
            bbox = boxes[i]
            conf = scores[i]

            detection = Detection(
                file=file,
                name=name,
                conf=conf.item(),
                bbox=BBox(
                    bbox[0].item(),
                    bbox[1].item(),
                    bbox[2].item(),
                    bbox[3].item()
                )
            )

            detections.append(detection)

        return detections

    def _plot(self, image, inference_state, image_name):
        IMG_DIR = "/home/ws/data/images"
        vis_path = os.path.join(IMG_DIR, 'sam3', image_name)

        print(f"Saving visualization to {vis_path}")
        plot_results(image, inference_state, save_path=vis_path)
