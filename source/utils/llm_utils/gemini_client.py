import os
import json
import time
import base64

from typing import Optional
from PIL import Image
import io
from google import genai
from google.genai import types as genai_types
from utils.environment import set_key
from utils.recursive_config import Config

def encode_image(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def load_json(json_path):
    with open(json_path, 'r') as file:
        json_data = json.load(file)
    return json.dumps(json_data)

class GeminiLocationPredictor:
    def __init__(self, model_name: str = "gemini-3.1-pro-preview"):
        api_key = set_key(Config(), "gemini")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        print(f"GeminiLocationPredictor will use model: {self.model_name}")

    def ask_for_shelf_with_room_json(self, json_string: str, object_name: str, object_not_found_location: str="trash can") -> Optional[dict]:
        try:
            json_string = load_json(json_string)

            system_msg = (
                "You will be given: "
                "1) a JSON with furniture (id, label, centroid, dimensions, room), "
                "2) an object name. "
                "Task: predict the 3 most likely furniture pieces where the object is most likely located. Return exactly 3 predictions. "
                "IMPORTANT: The 'id' in your response MUST be the exact same id from the provided furniture JSON. Do NOT invent new IDs. "
                "For each prediction, include id (exact string from input), label, room, probability, and a short spatial relation like 'on top of', 'inside'. "
                "Return ONLY a valid JSON in this format:\n"
                "{\n"
                '  "<object_name>": [\n'
                "    {\"id\": <id>, \"label\": <label>, \"room\": <room>, \"probability\": <float>, \"relation\": <short string>},\n"
                "    ...3 items total...\n"
                "  ]\n"
                "}"
            )

            user_msg = object_name
            if object_not_found_location:
                user_msg = f"Find {object_name}. The object was not found at this location: {object_not_found_location}. So exclude {object_not_found_location} from your predictions."

            start_time = time.time()
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[system_msg, json_string, user_msg],
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            end_time = time.time()

            print(f"Time taken just for Gemini API response: {end_time - start_time} seconds")
            print(f"Raw API response text:\n{response.text}")

            data = json.loads(response.text)
            object_name = list(data.keys())[0]

            locations = []
            for entry in data[object_name]:
                locations.append({
                    "furniture_id": entry["id"],
                    "furniture_name": entry["label"],
                    "relation": entry["relation"].split("(")[0].strip(),
                    "probability": entry["probability"],
                    "room": entry["room"].title()
                })

            result = {
                "item": object_name,
                "locations": locations
            }

            return result

        except Exception as e:
            print(f"An error occurred: {e}")
            return None
            
    def ask_for_rooms_with_scene_json(self, json_string: str) -> Optional[dict]:
        json_string = load_json(json_string)
        try:
            # 1. Define the system instruction for this specific task
            system_and_user_msg = (
                "The user will give you a json containing furniture (label, center position, dimensions) in the environment. "
                "1. Cluster the furniture from the json into 3 clusters using not just k-means on the x-y-center-coordinates, but also furniture labels. A room or location can contain 1 or multiple pieces of furniture. "
                "2. In each cluster, give a realistic room_name for a household setting, and list all furniture as members with id, label, centroid, and dimensions. "
                "3. Return the result in json format (all lowercase) as a single list of rooms."
            )
            
            # 2. Create a new model instance with this specific system instruction
            # model = GenerativeModel(self.model_name, system_instruction={"parts": [{"text": system_and_user_msg}]})

            # 3. Make the API call, using the new, task-specific model instance
            start_time = time.time()
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[system_and_user_msg, json_string],
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            end_time = time.time()

            print(f"Time taken just for Gemini API response: {end_time - start_time} seconds")
            print(f"Raw API response text:\n{response.text}")

            data = json.loads(response.text)
            return data
            
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def check_image_response_for_object(self, img_path: str, object_name: str) -> None:
        try:
            with open(img_path, "rb") as f:
                image_bytes = f.read()
            system_msg = (
                "You will be given an image of a shelf or cabinet. or a table or kitchen counter "
                "The user will ask about a specific object, including its attributes (e.g., color, type). "
                "Carefully check if the object exists in the image, paying attention to the attributes. "
                "If the object is present, confirm its existence and describe its location and relevant attributes (e.g., color, label). "
                "If the object is not present or the attributes do not match, clearly state that it is not found."
                "If you can then reliably estimate the approximate location (top left, top right, bottom left, bottom right, center) and size i.e. small, medium, large in the image."
            )
            user_msg = f"Is there a '{object_name}' in the image? Pay attention to attributes like color or type (e.g., blue bottle, mustard bottle, ketchup bottle). Begin first word of answer with yes or no."

            start_time = time.time()
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    genai_types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                    user_msg,
                ],
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_msg,
                    temperature=0.5,
                ),
            )
            end_time = time.time()

            print(f"Time taken for object detection API response: {end_time - start_time} seconds")
            print(f"Raw API response text:\n{response.text}")
            
            print(response.text)
            return response.text
            
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    # def detect_object_in_image(self, image_data: bytes, object_name: str) -> Optional[dict]:
    #     try:
    #         system_instruction = (
    #         "You are an object detection assistant. "
    #         "Your only task is to identify the requested object in the image "
    #         "and return its normalized bounding box coordinates considering top-left as origin. "
    #         "You must ONLY respond with a valid JSON object."
    #         )

    #         model = GenerativeModel(
    #             self.model_name,
    #             system_instruction={"parts": [{"text": system_instruction}]}
    #         )

    #         prompt = (
    #             f"Detect the object '{object_name}' in the image. "
    #             "The box coordinates must be normalized integers from 0 to 1000, "
    #             "in the order [y_min, x_min, y_max, x_max]. "
    #             "Return a JSON with the following format:\n"
    #             "{\n"
    #             '  "detected": <true|false>,\n'
    #             '  "detection_dict": {\n'
    #             f'    "label": "{object_name}",\n'  # force label to match query
    #             '    "confidence": <float>,\n'
    #             '    "box": [<y_min>, <x_min>, <y_max>, <x_max>]\n'
    #             "  }\n"
    #             "}\n"
    #             "If multiple proposals exist, keep only the one with the highest confidence. "
    #             "If not found, set detected=false and detection_dict={}. "
    #         )

    #         start_time = time.time()
    #         response = model.generate_content(
    #             contents=[
    #                 {"role": "user", "parts": [
    #                     {"text": prompt},
    #                     {"inline_data": {
    #                         "mime_type": "image/jpeg",
    #                         "data": image_data
    #                     }}
    #                 ]}
    #             ],
    #             generation_config=GenerationConfig(response_mime_type="application/json")
    #         )
    #         end_time = time.time()

    #         print(f"Time taken for object detection API response: {end_time - start_time} seconds")
    #         print(f"Raw API response text:\n{response.text}")

    #         data = json.loads(response.text)
    #         return data

    #     except Exception as e:
    #         print(f"An error occurred: {e}")
    #         return None


class GeminiERDetector:
    """Object detector using Gemini Robotics-ER 1.5"""

    def __init__(self, model_name: str = "gemini-robotics-er-1.5-preview"):
        api_key = set_key(Config(), "gemini_ER")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        print(f"GeminiERDetector will use model: {self.model_name}")

    def ask_for_shelf_with_room_json(self, json_string: str, object_name: str, object_not_found_location: str = "") -> Optional[dict]:
        try:
            # If json_string is a file path, read its content
            if os.path.isfile(json_string):
                with open(json_string, "r") as f:
                    json_string = f.read()

            system_msg = (
                "You will be given: "
                "1) a JSON with furniture (id, label, centroid, dimensions, room), "
                "2) an object name. "
                "Task: predict the 3 most likely furniture pieces where the object is most likely located. Return exactly 3 predictions. "
                "IMPORTANT: The 'id' in your response MUST be the exact same id from the provided furniture JSON. Do NOT invent new IDs. "
                "For each prediction, include id (exact string from input), label, room, probability, and a short spatial relation like 'on top of', 'inside'. "
                "Return ONLY a valid JSON in this format:\n"
                "{\n"
                '  "<object_name>": [\n'
                "    {\"id\": <id>, \"label\": <label>, \"room\": <room>, \"probability\": <float>, \"relation\": <short string>},\n"
                "    ...3 items total...\n"
                "  ]\n"
                "}"
            )

            user_msg = object_name
            if object_not_found_location:
                user_msg = f"Find {object_name}. The object was not found at this location: {object_not_found_location}. So exclude {object_not_found_location} from your predictions."

            start_time = time.time()
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[system_msg, json_string, user_msg],
                config=genai_types.GenerateContentConfig(
                    temperature=0.5,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                ),
            )
            end_time = time.time()

            print(f"Time taken for ER 1.5 shelf prediction: {end_time - start_time:.2f}s")
            print(f"Raw ER 1.5 response:\n{response.text}")

            raw = response.text.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            data = json.loads(raw)
            object_name = list(data.keys())[0]

            locations = []
            for entry in data[object_name]:
                locations.append({
                    "furniture_id": entry["id"],
                    "furniture_name": entry["label"],
                    "relation": entry["relation"].split("(")[0].strip(),
                    "probability": entry["probability"],
                    "room": entry["room"].title()
                })

            return {"item": object_name, "locations": locations}

        except Exception as e:
            print(f"GeminiERDetector ask_for_shelf error: {e}")
            return None

    def detect_object_in_image(self, image_data: bytes, object_name: str) -> Optional[dict]:
        """
        Detect a single object using Robotics-ER 1.5 bounding box API.
        Returns the same format as GeminiLocationPredictor.detect_object_in_image:
        {"detected": bool, "detection_dict": {"label": str, "confidence": float, "box": [y_min, x_min, y_max, x_max]}}
        """
        try:
            prompt = (
                f"Return bounding boxes as a JSON array with labels. "
                f"Only detect the object '{object_name}'. "
                f"Limit to 1 object — the best match. "
                f"The format should be as follows: "
                f'[{{"box_2d": [ymin, xmin, ymax, xmax], "label": "{object_name}"}}] '
                f"normalized to 0-1000. The values in box_2d must only be integers."
            )

            start_time = time.time()
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    genai_types.Part.from_bytes(data=image_data, mime_type="image/jpeg"),
                    prompt,
                ],
                config=genai_types.GenerateContentConfig(
                    temperature=0.5,
                    thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                ),
            )
            end_time = time.time()

            print(f"Time taken for ER 1.5 detection: {end_time - start_time:.2f}s")
            print(f"Raw ER 1.5 response:\n{response.text}")

            detections = json.loads(response.text)

            if detections and isinstance(detections, list) and len(detections) > 0:
                best = detections[0]
                return {
                    "detected": True,
                    "detection_dict": {
                        "label": best.get("label", object_name),
                        "confidence": best.get("confidence", 1.0),
                        "box": best["box_2d"],
                    },
                }
            return {"detected": False, "detection_dict": {}}

        except Exception as e:
            print(f"GeminiERDetector error: {e}")
            return None

# Uncomment and run this file to create rooms.json using Gemini based on scene.json

# if __name__ == "__main__":
    # The rest of your __main__ block remains the same
    # ... (code for mock data and calling the methods) ...
    # mock_furniture_data_1 = [
    #     {"id": "F_1", "label": "bookshelf", "centroid": [1, 2, 3], "dimensions": [0.5, 1.5, 2], "room": "living room"},
    #     {"id": "F_2", "label": "coffee table", "centroid": [4, 5, 6], "dimensions": [1, 0.5, 1], "room": "living room"},
    #     {"id": "F_3", "label": "nightstand", "centroid": [7, 8, 9], "dimensions": [0.4, 0.4, 0.6], "room": "bedroom"},
    #     {"id": "F_4", "label": "kitchen cabinet", "centroid": [10, 11, 12], "dimensions": [0.6, 2, 1], "room": "kitchen"},
    #     {"id": "F_5", "label": "office desk", "centroid": [13, 14, 15], "dimensions": [1.2, 0.8, 0.7], "room": "study"},
    #     {"id": "F_6", "label": "drawer", "centroid": [16, 17, 18], "dimensions": [0.5, 0.5, 0.8], "room": "bedroom"},
    #     {"id": "F_7", "label": "refrigerator", "centroid": [19, 20, 21], "dimensions": [0.8, 0.8, 1.8], "room": "kitchen"},
    # ]
    # mock_json_string_1 = json.dumps(mock_furniture_data_1)
    
    # mock_scene_json_2 = json.dumps([
    #     {"id": "F_8", "label": "sofa", "centroid": [1, 1, 0], "dimensions": [2.5, 1, 1]},
    #     {"id": "F_9", "label": "armchair", "centroid": [1.5, 3, 0], "dimensions": [1, 1, 1]},
    #     {"id": "F_10", "label": "coffee table", "centroid": [2, 2, 0], "dimensions": [1, 0.5, 0.5]},
    #     {"id": "F_11", "label": "dining table", "centroid": [5, 6, 0], "dimensions": [2, 1, 1]},
    #     {"id": "F_12", "label": "chair", "centroid": [5, 7, 0], "dimensions": [0.5, 0.5, 1]},
    #     {"id": "F_13", "label": "chair", "centroid": [5, 5, 0], "dimensions": [0.5, 0.5, 1]},
    #     {"id": "F_14", "label": "bed", "centroid": [10, 1, 0], "dimensions": [2, 1.5, 1]},
    #     {"id": "F_15", "label": "nightstand", "centroid": [11, 1.5, 0], "dimensions": [0.5, 0.5, 0.5]},
    # ])

    # image_path = "/home/ws/data/images/old/head_image_rgb.png"
    # if os.path.exists(image_path):
    #     with open(image_path, "rb") as img_file:
    #         img = Image.open(img_file)
    #         with io.BytesIO() as jpeg_buffer:
    #             img.convert("RGB").save(jpeg_buffer, format="JPEG", quality=85)
    #             image_bytes = jpeg_buffer.getvalue()
    # else:
    #     print(f"Warning: Image file not found at {image_path}. Skipping image check.")
    #     image_bytes = None

    # try:
    #     predictor = GeminiLocationPredictor()

    #     # # Call methods as before
    #     # result_keys = predictor.ask_for_shelf_with_room_json(mock_json_string_1, "keys")
    #     # if result_keys:
    #     #     print("\n--- Predictions for 'keys' ---")
    #     #     print(json.dumps(result_keys, indent=2))
    #     #     print("---")
            
            
    #     config = Config()
    #     ending  = '2025_09_15'
        
    #     scene_json_path = os.path.join(config.get_subpath("scene_graph"), ending, "scene.json")
    #     print(scene_json_path)
    #     img_path = os.path.join(config.get_subpath("images"), ending)
    #     write_path = os.path.join(config.get_subpath("scene_graph"), ending, "locations", "rooms.json")
    #     location_path = os.path.join(config.get_subpath("scene_graph"), ending,  "shelf_locations.json")
    #     #ask_for_shelf_with_image(client, img_path)
    #     #ask_for_shelf_content(client, img_path)
    #     #rooms = ask_for_rooms_with_scene_json(client, json_path)
    #     #rooms_json_path = os.path.join(config.get_subpath("scene_graph"), ending, "rooms_2.json")
    #     result_rooms = predictor.ask_for_rooms_with_scene_json(scene_json_path)
    #     if result_rooms:
    #         print("\n--- Clustered Rooms ---")
    #         print(json.dumps(result_rooms, indent=2))
    #         print("---")

    #     with open(write_path, 'w') as f:
    #         json.dump(result_rooms, f, indent=4)
    #     # if image_bytes:
    #     #     print("\n--- Simulating object check in image ---")
    #     #     result_image_check = predictor.detect_object_in_image(image_bytes, "bottle")
    #     #     if result_image_check:
    #     #         print(json.dumps(result_image_check, indent=2))
    #     #     print("---")


    # except ValueError as e:
    #     print(f"Initialization Error: {e}")
    # except Exception as e:
    #     print(f"An unexpected error occurred during execution: {e}")