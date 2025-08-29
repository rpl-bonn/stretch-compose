from openai import OpenAI
from utils.environment import set_key
from utils.recursive_config import Config

import json
import os

NO_ROOMS = 2

config = Config()
client = OpenAI(
        base_url = "https://integrate.api.nvidia.com/v1",
        api_key = set_key(config, "deepseek")
    )

def ask_deepseek_for_locations(item: str, hint: str = "", no_proposals: int = 3):
    # Open scene.json
    json_path = os.path.join(config.get_subpath("scene_graph"), config["pre_scanned_graphs"]["high_res"], "scene.json")
    with open(json_path, 'r') as file:
        json_data = json.load(file)
    json_string = json.dumps(json_data)

    system_msg = f"The user will give you two things. \
                   A json containing furniture in the environment with its label, centroid, and bounding box dimensions.\
                   An item he/she wants to search for in the environment.\
                1. Cluster the furniture into 2 clusters using k-means on the x-y-center-coordinates and the labels. Give each cluster a room name. \
                2. Name the {no_proposals} most likely places (furniture name and relation to furniture) with descending likelihood order, where the item can be found. Also consider the room. \
                3. Return the result as a json in the following format: \
                    {{\"item\": \"item_name\", \"locations\":[{{\"furniture_id\", \"furniture_name\", \"relation\", \"room\"}}]}}"

    completion = client.chat.completions.create(
        model="deepseek-ai/deepseek-r1",
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
                        "text": item,
                    },
                    {
                        "type": "text",
                        "text": hint,
                    }
                ],
            },
        ],
        response_format={
            'type': 'json_object'
        },
        temperature=0.6,
        top_p=0.7,
        max_tokens=4096,
        stream=False
    )

    print("Reasoning:\n", completion.choices[0].message.content.split("```json")[0])
    #print("Response:\n", completion.choices[0].message.content.split("</think>\n\n")[1])

    # Save result json
    result = completion.choices[0].message.content.split("```json")[0]
    json_data = json.loads(result)
    json_data["chain-of-thought"] = completion.choices[0].message.content.split("```json")[0]
    os.makedirs(os.path.join(config.get_subpath("scene_graph"), config["pre_scanned_graphs"]["high_res"], "locations"), exist_ok=True)
    result_path = os.path.join(config.get_subpath("scene_graph"), config["pre_scanned_graphs"]["high_res"], "locations", f"{item.replace(' ', '_')}.json")
    with open(result_path, 'w') as file:
        json.dump(json_data, file, indent=4)
        
        
def ask_deepseek_for_object(query: str):
    # Open graph.json
    json_path = os.path.join(config.get_subpath("scene_graph"), config["pre_scanned_graphs"]["high_res"], "graph.json")
    with open(json_path, 'r') as file:
        json_data = json.load(file)
    json_string1 = json.dumps(json_data)
    json_path = os.path.join(config.get_subpath("scene_graph"), config["pre_scanned_graphs"]["high_res"], "scene.json")
    with open(json_path, 'r') as file:
        json_data = json.load(file)
    json_string2= json.dumps(json_data)

    system_msg = f"The user will give you three things. \
                   A json containing furniture, drawers, and objects in the environment.\
                   A json containing only the furniture with its label, centroid, and bounding box dimensions.\
                   A query the user wants to fulfill.\
                1. Look at the objects in node_labels that are not in immovable_labels.\
                2. Cluster the furniture into 2 clusters using k-means on the x-y-center-coordinates and the labels. Give each cluster a room name. \
                3. Look at the query and choose the most likely object, the user wants to have. \
                4. If there are multiple objects of the same type, look at the connected furnitures and consider the room, where the object is located. \
                5. Return the result as a json in the following format: \
                    {{\"item_id\": \"id\", \"item_name\": \"label\"}}"
    completion = client.chat.completions.create(
        model="deepseek-ai/deepseek-r1",
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
                        "text": json_string1,
                    },
                    {
                        "type": "text",
                        "text": json_string2,
                    },
                    {
                        "type": "text",
                        "text": query,
                    },
                ],
            },
        ],
        response_format={
            'type': 'json_object'
        },
        temperature=0.6,
        top_p=0.7,
        max_tokens=4096,
        stream=False
    )

    print("Reasoning:\n", completion.choices[0].message.content.split("```json")[0])
    print("Response:\n", completion.choices[0].message.content.split("</think>\n\n")[1])
    
    # Save result json
    result = completion.choices[0].message.content.split("```json")[1].split("```")[0]
    json_data = json.loads(result)
    print(json_data["item_id"])
    print(json_data["item_name"])
    
    return json_data["item_id"], json_data["item_name"]

def main(config: Config):
    q = "I am hungry."
    ask_deepseek_for_object(q)
    #ask_deepseek_for_locations("bowl")


if __name__ == "__main__":
   main(Config())