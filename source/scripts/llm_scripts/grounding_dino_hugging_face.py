import argparse
import json
import os
import cv2
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from PIL import Image

IMG = "/home/ws/data/images/head_image_rgb_2.png"
QUERY = "purple folder"
JSON_OUT = "/home/ws/data/images/grounding_dino.json"
IMG_OUT = "/home/ws/data/images/head_image_rgb_dino.png"

# Load model and processor once (global)
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").to(device)

def run_grounding_dino(image_path: str, query: str, json_path: str = "output.json", image_out: str = "output.png") -> dict:
    """Run HuggingFace GroundingDINO and save detections + annotated image."""

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Preprocess
    inputs = processor(images=image, text=query, return_tensors="pt").to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Postprocess
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        target_sizes=[image.size[::-1]],  # (height, width)
    )
    print(f"\n Raw model results: {results} \n")
    detections = []
    if results:
        for box, label, score in zip(results[0]["boxes"], results[0]["text_labels"], results[0]["scores"]):
            if score < 0.2:
                continue
            x1, y1, x2, y2 = map(int, box.tolist())
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "label": label,
                "score": float(score)
            })

    # Save JSON
    with open(json_path, "w") as f:
        json.dump({"detections": detections}, f, indent=2)

    # Draw bboxes
    img_cv = cv2.imread(image_path)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img_cv, f"{label} ({det['score']:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(image_out, img_cv)

    print(f"Detections JSON saved to {os.path.abspath(json_path)}")
    print(f"Annotated image saved to {os.path.abspath(image_out)}")

    return {"detections": detections}

def main():
    result = run_grounding_dino(IMG, QUERY, JSON_OUT, IMG_OUT)
    print(result)

if __name__ == "__main__":
    main()
