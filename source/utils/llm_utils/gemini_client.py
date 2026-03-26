import os
import json
import time
from google.generativeai import GenerativeModel, configure
from google.generativeai.types import GenerationConfig
from typing import Optional
from PIL import Image
import io

class GeminiLocationPredictor:
    def __init__(self, model_name: str = "gemini-2.5-pro"):
        if not os.getenv("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] =  "AIzaSyDn6S5FWDub8lTDg5YS8sAaXJ7i2Mt6u40"
            #raise ValueError("GOOGLE_API_KEY environment variable is not set.")
        configure(api_key=os.getenv("GOOGLE_API_KEY"))

        self.model_name = model_name
        print(f"GeminiLocationPredictor will use model: {self.model_name}")

    def ask_for_shelf_with_room_json(self, json_string: str, object_name: str, object_not_found_location: str="") -> Optional[dict]:
        try:
            # 1. Define the system instruction for this specific task
            system_msg = (
                "You will be given: "
                "1) a JSON with furniture (id, label, centroid, dimensions, room), "
                "2) an object name. "
                "Task: predict the 3 most likely furniture for the object. "
                "For each prediction, include id as a str, label, room, probability, and a short spatial only relation like on top of, inside. "
                "Return ONLY a valid JSON in this format:\n"
                "{\n"
                '  "<object_name>": [\n'
                "    {\"id\": <id>, \"label\": <label>, \"room\": <room>, \"probability\": <float>, \"relation\": <short string>},\n"
                "    ...3 items total...\n"
                "  ]\n"
                "}"
            )
            
            # 2. Create a new model instance with this specific system instruction
            model = GenerativeModel(self.model_name, system_instruction={"parts": [{"text": system_msg}]})

            # 3. Define the user message
            user_msg = object_name
            if object_not_found_location:
                user_msg = f"Find {object_name}. The object was not found at this location: {object_not_found_location}. So exclude {object_not_found_location} from your predictions."

            start_time = time.time()
            response = model.generate_content(
                contents=[
                    {"role": "user", "parts": [{"text": json_string}, {"text": user_msg}]},
                ],
                generation_config=GenerationConfig(response_mime_type="application/json")
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
        try:
            # 1. Define the system instruction for this specific task
            system_and_user_msg = (
                "The user will give you a json containing furniture (label, center position, dimensions) in the environment. "
                "1. Cluster the furniture from the json into 3 clusters using not just k-means on the x-y-center-coordinates, but also furniture labels. A room or location can contain 1 or multiple pieces of furniture. "
                "2. In each cluster, give a realistic room_name for a household setting, and list all furniture as members with id, label, centroid, and dimensions. "
                "3. Return the result in json format (all lowercase) as a single list of rooms."
            )
            
            # 2. Create a new model instance with this specific system instruction
            model = GenerativeModel(self.model_name, system_instruction={"parts": [{"text": system_and_user_msg}]})

            # 3. Make the API call, using the new, task-specific model instance
            start_time = time.time()
            response = model.generate_content(
                contents=[
                    {"role": "user", "parts": [{"text": json_string}]},
                ],
                generation_config=GenerationConfig(response_mime_type="application/json", max_output_tokens=4096)
            )
            end_time = time.time()

            print(f"Time taken just for Gemini API response: {end_time - start_time} seconds")
            print(f"Raw API response text:\n{response.text}")

            data = json.loads(response.text)
            return data
            
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def check_object_in_image(self, image_data: bytes, object_name: str) -> Optional[dict]:
        try:
            # 1. Define the system instruction for this specific task
            system_instruction = (
                "You are an object detection assistant. "
                "Your only task is to identify objects in images and return their bounding boxes. "
                "You must ONLY respond with a valid JSON object."
            )
            
            # 2. Create a new model instance with this instruction
            model = GenerativeModel(self.model_name, system_instruction={"parts": [{"text": system_instruction}]})
            
            # 3. Define the user prompt for this specific image and object
            prompt = (
                f"Is the object '{object_name}' present in the image? "
                "Return a JSON object with a 'found' key (boolean). If the object is found, "
                "include a 'bounding_boxes' key with a list of objects, each containing "
                "the 'x1', 'y1', 'x2', and 'y2' coordinates of the bounding box. "
                "If not found, 'found' should be false and 'bounding_boxes' an empty list. For multiple proposals for an object sort them by your confidence and include the "
            )

            start_time = time.time()
            response = model.generate_content(
                contents=[
                    {"role": "user", "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": image_data}}
                    ]},
                ],
                generation_config=GenerationConfig(response_mime_type="application/json")
            )
            end_time = time.time()

            print(f"Time taken for object detection API response: {end_time - start_time} seconds")
            print(f"Raw API response text:\n{response.text}")
            
            data = json.loads(response.text)
            return data
            
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def detect_object_in_image(self, image_data: bytes, object_name: str) -> Optional[dict]:
        try:
            system_instruction = (
            "You are an object detection assistant. "
            "Your only task is to identify the requested object in the image "
            "and return its normalized bounding box coordinates considering top-left as origin. "
            "You must ONLY respond with a valid JSON object."
            )

            model = GenerativeModel(
                self.model_name,
                system_instruction={"parts": [{"text": system_instruction}]}
            )

            prompt = (
                f"Detect the object '{object_name}' in the image. "
                "The box coordinates must be normalized integers from 0 to 1000, "
                "in the order [y_min, x_min, y_max, x_max]. "
                "Return a JSON with the following format:\n"
                "{\n"
                '  "detected": <true|false>,\n'
                '  "detection_dict": {\n'
                f'    "label": "{object_name}",\n'  # force label to match query
                '    "confidence": <float>,\n'
                '    "box": [<y_min>, <x_min>, <y_max>, <x_max>]\n'
                "  }\n"
                "}\n"
                "If multiple proposals exist, keep only the one with the highest confidence. "
                "If not found, set detected=false and detection_dict={}. "
            )

            start_time = time.time()
            response = model.generate_content(
                contents=[
                    {"role": "user", "parts": [
                        {"text": prompt},
                        {"inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_data
                        }}
                    ]}
                ],
                generation_config=GenerationConfig(response_mime_type="application/json")
            )
            end_time = time.time()

            print(f"Time taken for object detection API response: {end_time - start_time} seconds")
            print(f"Raw API response text:\n{response.text}")

            data = json.loads(response.text)
            return data

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

if __name__ == "__main__":
    # The rest of your __main__ block remains the same
    # ... (code for mock data and calling the methods) ...
    mock_furniture_data_1 = [
        {"id": "F_1", "label": "bookshelf", "centroid": [1, 2, 3], "dimensions": [0.5, 1.5, 2], "room": "living room"},
        {"id": "F_2", "label": "coffee table", "centroid": [4, 5, 6], "dimensions": [1, 0.5, 1], "room": "living room"},
        {"id": "F_3", "label": "nightstand", "centroid": [7, 8, 9], "dimensions": [0.4, 0.4, 0.6], "room": "bedroom"},
        {"id": "F_4", "label": "kitchen cabinet", "centroid": [10, 11, 12], "dimensions": [0.6, 2, 1], "room": "kitchen"},
        {"id": "F_5", "label": "office desk", "centroid": [13, 14, 15], "dimensions": [1.2, 0.8, 0.7], "room": "study"},
        {"id": "F_6", "label": "drawer", "centroid": [16, 17, 18], "dimensions": [0.5, 0.5, 0.8], "room": "bedroom"},
        {"id": "F_7", "label": "refrigerator", "centroid": [19, 20, 21], "dimensions": [0.8, 0.8, 1.8], "room": "kitchen"},
    ]
    mock_json_string_1 = json.dumps(mock_furniture_data_1)
    
    mock_scene_json_2 = json.dumps([
        {"id": "F_8", "label": "sofa", "centroid": [1, 1, 0], "dimensions": [2.5, 1, 1]},
        {"id": "F_9", "label": "armchair", "centroid": [1.5, 3, 0], "dimensions": [1, 1, 1]},
        {"id": "F_10", "label": "coffee table", "centroid": [2, 2, 0], "dimensions": [1, 0.5, 0.5]},
        {"id": "F_11", "label": "dining table", "centroid": [5, 6, 0], "dimensions": [2, 1, 1]},
        {"id": "F_12", "label": "chair", "centroid": [5, 7, 0], "dimensions": [0.5, 0.5, 1]},
        {"id": "F_13", "label": "chair", "centroid": [5, 5, 0], "dimensions": [0.5, 0.5, 1]},
        {"id": "F_14", "label": "bed", "centroid": [10, 1, 0], "dimensions": [2, 1.5, 1]},
        {"id": "F_15", "label": "nightstand", "centroid": [11, 1.5, 0], "dimensions": [0.5, 0.5, 0.5]},
    ])

    image_path = "/home/ws/data/images/old/head_image_rgb.png"
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            img = Image.open(img_file)
            with io.BytesIO() as jpeg_buffer:
                img.convert("RGB").save(jpeg_buffer, format="JPEG", quality=85)
                image_bytes = jpeg_buffer.getvalue()
    else:
        print(f"Warning: Image file not found at {image_path}. Skipping image check.")
        image_bytes = None

    try:
        predictor = GeminiLocationPredictor()

        # # Call methods as before
        # result_keys = predictor.ask_for_shelf_with_room_json(mock_json_string_1, "keys")
        # if result_keys:
        #     print("\n--- Predictions for 'keys' ---")
        #     print(json.dumps(result_keys, indent=2))
        #     print("---")
            
        # result_rooms = predictor.ask_for_rooms_with_scene_json(mock_scene_json_2)
        # if result_rooms:
        #     print("\n--- Clustered Rooms ---")
        #     print(json.dumps(result_rooms, indent=2))
        #     print("---")

        if image_bytes:
            print("\n--- Simulating object check in image ---")
            result_image_check = predictor.detect_object_in_image(image_bytes, "bottle")
            if result_image_check:
                print(json.dumps(result_image_check, indent=2))
            print("---")


    except ValueError as e:
        print(f"Initialization Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during execution: {e}")