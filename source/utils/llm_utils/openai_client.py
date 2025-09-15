from openai import OpenAI
from utils.environment import set_key
from utils.recursive_config import Config
import base64
import json
import os
from utils.recursive_config import Config
import time
config = Config()

oai_client = OpenAI(api_key=set_key(config, "openai"))

def encode_image(img_path):
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_json(json_path):
    with open(json_path, 'r') as file:
        json_data = json.load(file)
    return json.dumps(json_data)

def save_json(json_path, json_data):
    with open(json_path, 'w') as file:
        json.dump(json_data, file, indent=4)
    

def extend_json(client: OpenAI, json_path: str, write_path: str) -> None:
    json_string = load_json(json_path)
    system_msg = "You will get a json containing furniture (label, center position, dimensions, movable). \
                  Complete and return the json with a short 'description' field for each object. \
                  Also replace the center and dimension fields with a 'bounding box' field containing the coordinates of all 8 corners."
                  
    response = client.chat.completions.create(
        model="o1",
        messages=[
            {
                "role": "system",
                "content": system_msg,
            },
            {
                "role": "user",
                "content": 
                [
                    {
                        "type": "text",
                        "text": json_string,  
                    },
                ],
            },
        ],
        response_format={
            'type': 'json_object'
        },
        max_completion_tokens=4096,
    )
    print(response)
    print(response.choices[0].message.content)
    save_json(write_path, response.choices[0].message.content)

def ask_for_shelf_with_room_json(client: OpenAI, json_path: str, object_name: str, object_not_found_location: str="", model_name: str="gpt-4o-mini") -> None:
    json_string = load_json(json_path)
    # system_msg = "The user will give you 2 things. \
    #               1. A json containing a list of furniture (label, center position, dimensions) in the environment, clustered into rooms by an llm. \
    #               2. An object he or she wants to search for in the environment. \
    #               3. For each object, name the 3 most likely furniture with estimated probabilities from the json with IDs, where it can be found. Also consider the room and name a short relation towards the furniture. \
    #               4. Return the result in json format (all lowercase)"
    
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
    if object_not_found_location is None or object_not_found_location == "":
        user_msg = object_name
    else:
        user_msg = f" Find {object_name}. The object was not found at this location: {object_not_found_location}. So exclude {object_not_found_location} from your predictions."
    start_time = time.time()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_msg,
            },
            {
                "role": "user",
                "content": 
                [
                    {
                        "type": "text",
                        "text": json_string,  
                    },
                    {
                        "type": "text",
                        "text": user_msg,
                    },
                ],
            },
        ],
        max_completion_tokens=2048,
    )
    end_time = time.time()
    print(f"Time taken just for openai api response: {end_time - start_time} seconds")
    print(response)
    print(response.choices[0].message.content)
    data = json.loads(response.choices[0].message.content)
    object_name = list(data.keys())[0]
    # build target schema
    locations = []
    for entry in data[object_name]:
        locations.append({
            "furniture_id": entry["id"],
            "furniture_name": entry["label"],
            "relation": entry["relation"].split("(")[0].strip(),  # drop likelihood notes
            "probability": entry["probability"],
            "room": entry["room"].title()
        })

    result = {
        "item": object_name,
        "locations": locations
    }

    # save as object_name.json (replace spaces with underscores if you like)
    return result
    #print(json.dumps(result, indent=4))
    
    
def ask_for_rooms_with_scene_json(client: OpenAI, json_path: str) -> None:
    json_string = load_json(json_path)
    system_msg = "The user will give you 1 thing. \
                  A json containing furniture (label, center position, dimensions) in the environment. \
                  1. Cluster the furniture from the json into 3 clusters using not just k-means on the x-y-center-coordinates, but also furniture labels. A room or location can contain 1 or multiple pieces of furniture. \
                  2. In each cluster, give room_name, and list all furniture as members with id, use id as the key with each id have the label from scene.json, its centroid and dimensions. \
                  3. Return the result in json format (all lowercase)"
    user_msg = "Ensure rooms are named realistically for a household setting, so that it can help future llm queries for semantic search locations."
    
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {
                "role": "system",
                "content": system_msg,
            },
            {
                "role": "user",
                "content": 
                [
                    {
                        "type": "text",
                        "text": json_string,  
                    },
                    {
                        "type": "text",
                        "text": user_msg,
                    },
                ],
            },
        ],
        max_completion_tokens=4096,
    )
    print(response)
    print(response.choices[0].message.content)
    data = json.loads(response.choices[0].message.content)
    return data

def ask_for_shelf_with_json(client: OpenAI, json_path: str) -> None:
    json_string = load_json(json_path)
    system_msg = "The user will give you 2 things. \
                  A json containing furniture (label, center position, dimensions) in the environment. \
                  A list of objects he/she wants to search for in the environment. \
                  1. Cluster the furniture from the json into 3 clusters using k-means on the x-y-center-coordinates. \
                  2. Now include the furniture labels and give each cluster a room name. \
                  3. For each object, name the 3 most likely furniture from the json with IDs, where it can be found. \
                  Also consider the room and name the relation towards the furniture. \
                  4. Return the result in json format (all lowercase)"
    user_msg = "ketchup bottle"
    
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {
                "role": "system",
                "content": system_msg,
            },
            {
                "role": "user",
                "content": 
                [
                    {
                        "type": "text",
                        "text": json_string,  
                    },
                    {
                        "type": "text",
                        "text": user_msg,
                    },
                ],
            },
        ],
        max_completion_tokens=4096,
    )
    print(response)
    print(response.choices[0].message.content)
    data = json.loads(response.choices[0].message.content)
    object_name = list(data["search_results"].keys())[0]

    # build target schema
    locations = []
    for entry in data["search_results"][object_name]:
        locations.append({
            "furniture_id": entry["id"],
            "furniture_name": entry["label"],
            "relation": entry["relation"].split("(")[0].strip(),  # drop likelihood notes
            "room": entry["room"].title()
        })

    result = {
        "item": object_name,
        "locations": locations
    }

    # save as object_name.json (replace spaces with underscores if you like)
    filename = f"{object_name.replace(' ', '_')}.json"
    with open(filename, "w") as f:
        json.dump(result, f, indent=4)

    print(f"Saved to {filename}")
    print(json.dumps(result, indent=4))
    
    

def ask_for_shelf_with_image(client: OpenAI, img_path: str) -> None:
    base64_image1 = encode_image(os.path.join(img_path, "shelf1.png"))
    base64_image2 = encode_image(os.path.join(img_path, "shelf2.png"))
    system_msg = "The user will give you 2 images containing each a shelf/cabinet and a list of items he/she wants to find.\
                  First propose the 2 most likely regions/rooms for each shelf in a household setting. \
                  Then, name the image (1, 2) for each item in which it most likely can be found with a short reason. \
                  After that, also name a shelf compartment or door/drawer where the item might most likely be with a short reason."
    user_msg = "water bottle, cat plushy, cup, book, image frame, herbs, plant, fork, keys"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": system_msg,
            },
            {
                "role": "user",
                "content": 
                [
                    {
                        "type": "image_url",
                        "image_url": {"url":  f"data:image/png;base64,{base64_image1}",
                                      "detail": "low",},   
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url":  f"data:image/png;base64,{base64_image2}",
                                      "detail": "low",},
                        
                    },
                    {
                        "type": "text",
                        "text": user_msg,
                    },
                ],
            },
        ],
        max_tokens=500,
    )
    print(response)
    print(response.choices[0].message.content)
    return response.choices[0].message.content

def ask_for_shelf_content(client: OpenAI, img_path: str) -> None:
    base64_image1 = encode_image(os.path.join(img_path, "navigation_image.png"))
    system_msg = "The user will give you an fisheye image of a shelf or cabinet. Have a close look at it. Describe how many compartments, doors, and drawers it has. \
                  The user will then name a list of items. Tell, if the item can be seen in the shelf/cabinet and describe where in the image it is. \
                  Then, describe all items you can see inside the open compartments and on top of it."
    user_msg = "water bottle, cat plushy, book, image frame, plant, folder, keys, shark plushy, watering can"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": system_msg,
            },
            {
                "role": "user",
                "content": 
                [
                    {
                        "type": "image_url",
                        "image_url": {"url":  f"data:image/png;base64,{base64_image1}",
                                      "detail": "low",},   
                    },
                    {
                        "type": "text",
                        "text": user_msg,
                    },
                    
                ],
            },
        ],
        max_tokens=500,
    )
    print(response)
    print(response.choices[0].message.content)


    

def main(config: Config):
    client = OpenAI(api_key=set_key(config, "openai"))
    ending = config["pre_scanned_graphs"]["high_res"]
    
    json_path = os.path.join(config.get_subpath("scene_graph"), ending, "rooms.json")
    print(json_path)
    img_path = os.path.join(config.get_subpath("images"), ending)
    write_path = os.path.join(config.get_subpath("scene_graph"), ending, "description_scene_graph.json")
    location_path = os.path.join(config.get_subpath("scene_graph"), ending,  "shelf_locations.json")
    #ask_for_shelf_with_image(client, img_path)
    #ask_for_shelf_content(client, img_path)
    #rooms = ask_for_rooms_with_scene_json(client, json_path)
    #rooms_json_path = os.path.join(config.get_subpath("scene_graph"), ending, "rooms_2.json")
    object_name = "teddy bear"
    start_time = time.time()
    object_location = ask_for_shelf_with_room_json(client, json_path, object_name=object_name, model_name ="gpt-4o-mini")
    end_time = time.time()
    
    object_location_json_path = os.path.join(config.get_subpath("scene_graph"), ending, "locations", f"{object_name}.json")
    
    with open(object_location_json_path, 'w') as f:
        json.dump(object_location, f, indent=4)
    print(f"Time taken with i/o operations: {end_time - start_time} seconds")
    print(object_location)
    # with open(rooms_json_path, 'w') as f:
    #     json.dump(rooms, f, indent=4)
    # print(rooms)
    #new_json = extend_json(client, json_path, write_path)
    

if __name__ == "__main__":
    main(Config())
