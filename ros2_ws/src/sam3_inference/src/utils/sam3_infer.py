import os
import inspect
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
        return (
            f"BBox(xmin={self.xmin}, ymin={self.ymin}, "
            f"xmax={self.xmax}, ymax={self.ymax})"
        )


class Detection:
    def __init__(self, name, conf, bbox: BBox):
        self.name = name
        self.conf = float(conf)
        self.bbox = bbox

    def __repr__(self):
        return (
            f"Detection(name='{self.name}', "
            f"conf={self.conf}, bbox={self.bbox})"
        )


class Sam3Inference:

    def __init__(self, confidence_threshold=0.6, device="cuda"):
        """
        Initializes SAM3 model once.
        Should be created once in your ROS service constructor.
        """

        # Detect device
        self.device = device if torch.cuda.is_available() else "cpu"

        self.sam3_root = os.path.dirname(sam3.__file__)
        print(f"Using sam3 module from: {sam3.__file__}")

        self.model = self.load_model()
        self.model.to(self.device)
        self.model.eval()

        self.processor = Sam3Processor(
            self.model,
            confidence_threshold=confidence_threshold
        )

        print("SAM3 processor UP & Running successfully")

    def load_model(self):
        bpe_path = self._resolve_bpe_path()
        checkpoint_path = self._resolve_checkpoint_path()
        model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=checkpoint_path,
            load_from_HF=checkpoint_path is None,
        )
        print("SAM3 model loaded successfully")
        return model

    def _resolve_checkpoint_path(self):
        # Support multiple env names to match different launch setups.
        env_keys = ("SAM3_CKPT_PATH", "SAM3_CHECKPOINT_PATH", "SAM3_MODEL_PATH")
        for key in env_keys:
            candidate = os.environ.get(key)
            if candidate and os.path.exists(candidate):
                print(f"Using SAM3 checkpoint from {key}: {candidate}")
                return candidate

        return None

    def _resolve_bpe_path(self):
        env_path = os.environ.get("SAM3_BPE_PATH")
        candidates = []
        if env_path:
            candidates.append(env_path)

        candidates.extend([
            os.path.join(self.sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz"),
            "/home/ws/src/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            "/home/ws/src/sam3_og/assets/bpe_simple_vocab_16e6.txt.gz",
        ])

        for path in candidates:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            "SAM3 BPE vocabulary file not found. Checked: " + ", ".join(candidates)
        )
        
    def infer(self, image, prompts, input_format="bgr", visualize=False):
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
                result["scores"]
            )

            if visualize:
                self._plot(image, result, f"sam3_{text_prompt}_1.png")

        return all_detections


    def _convert_to_detection_format(self, name, boxes, scores):
        detections = []

        for i in range(len(boxes)):
            bbox = boxes[i]
            conf = scores[i]

            detection = Detection(
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
        # Different sam3 builds expose different plot_results signatures.
        sig = inspect.signature(plot_results)
        # if "save_path" in sig.parameters:
        plot_results(image, inference_state, save_path=vis_path)
        # else:
        #     plot_results(image, inference_state)

