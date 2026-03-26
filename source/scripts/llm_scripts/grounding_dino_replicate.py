from utils.environment import set_key
from utils.recursive_config import Config
import base64
import json
import os
from utils.recursive_config import Config
import time
config = Config()
import replicate
import argparse
import cv2

IMG = "/home/ws/data/images/head_image_rgb.png"
QUERY = "pringles can"
JSON_OUT = "/home/ws/data/images/grounding_dino.json"
IMG_OUT = "/home/ws/data/images/head_image_rgb_dino.png"

def run_grounding_dino(image_path: str, query: str, json_path: str = "output.json", image_out: str = "output.png") -> dict:
    """Run GroundingDINO on an image and save detections."""
    # Encode image
    with open(image_path, "rb") as file:
        data = base64.b64encode(file.read()).decode("utf-8")
        image = f"data:application/octet-stream;base64,{data}"

    # Prepare request
    input_data = {
        "image": image,
        "query": query,
        "box_threshold": 0.2,
        "text_threshold": 0.2
    }

    # Run model
    output = replicate.run(
        "adirik/grounding-dino:efd10a8ddc57ea28773327e881ce95e20cc1d734c589f7dd01d2036921ed78aa",
        input=input_data
    )

    # Save results
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    # print(f"Detections saved to {os.path.abspath(output_path)}")
    # return output
    img = cv2.imread(image_path)

    # Draw bounding boxes
    if "detections" in output:
        for det in output["detections"]:
            x1, y1, x2, y2 = map(int, det["bbox"])
            label = det["label"]

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Save annotated image
    cv2.imwrite(image_out, img)

    print(f"Detections JSON saved to {os.path.abspath(json_path)}")
    print(f"Annotated image saved to {os.path.abspath(image_out)}")

    return output

def main():
    result = run_grounding_dino(IMG, QUERY, JSON_OUT, IMG_OUT)
    print(result)
        
if __name__ == "__main__":
    main()