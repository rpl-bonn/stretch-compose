import base64
import json
import os
import cv2
from inference.models.grounding_dino import GroundingDINO

# === CONFIG ===
API_KEY = "OgGwzg8N26RquWMskQ4x"  # <- replace with your Roboflow API key
IMG = "/home/ws/data/images/head_image_rgb_2.png"
QUERY = ["purple folder"]   # list of queries
JSON_OUT = "/home/ws/data/images/grounding_dino.json"
IMG_OUT = "/home/ws/data/images/head_image_rgb_dino.png"

# === SETUP MODEL ===
model = GroundingDINO(api_key=API_KEY)

def run_grounding_dino(image_path: str, queries: list, json_path: str, img_out: str):
    # Run inference directly with file path
    results = model.infer(
        image=image_path,
        text=queries,
        box_threshold=0.5,
        text_threshold=0.5,
    )
    print(f"\n Raw model results: {results} \n")
    
    # Convert to dict
    detections = results.model_dump()

    # Save JSON
    with open(json_path, "w") as f:
        json.dump(detections, f, indent=2)
    print(f"Detections JSON saved to {os.path.abspath(json_path)}")

    # Draw bboxes
    img = cv2.imread(image_path)
    for det in detections.get("predictions", []):
        x1, y1, x2, y2 = int(det["x_min"]), int(det["y_min"]), int(det["x_max"]), int(det["y_max"])
        label = det["class"]
        score = det["confidence"]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f"{label} ({score:.2f})", (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite(img_out, img)
    print(f"Annotated image saved to {os.path.abspath(img_out)}")

    return detections


if __name__ == "__main__":
    output = run_grounding_dino(IMG, QUERY, JSON_OUT, IMG_OUT)
    print(output)
