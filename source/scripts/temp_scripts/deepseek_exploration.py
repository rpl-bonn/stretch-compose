from openai import OpenAI
from utils.environment import _load_environment_file
from utils.recursive_config import Config

import json
import os

NO_ROOMS = 2

config = Config()

def set_deepseek_key(config: Config) -> None:
    api_data = _load_environment_file(config)["api"]
    deepseek_key = api_data["deepseek"]["key"]
    return deepseek_key


def ask_deepseek(item: str):
    client = OpenAI(
        base_url = "https://integrate.api.nvidia.com/v1",
        api_key = set_deepseek_key(config)
    )

    # Open scene_graph.json
    json_path = os.path.join(config.get_subpath("scene_graph"), config["pre_scanned_graphs"]["high_res"], "scene_graph.json")
    with open(json_path, 'r') as file:
        json_data = json.load(file)
    json_string = json.dumps(json_data)

    system_msg = f"The user will give you two things. \
                   A json containing furniture in the environment with its label, centroid, and bounding box dimensions.\
                   An item he/she wants to search for in the environment.\
                1. Cluster the furniture into {NO_ROOMS} clusters using k-means on the x-y-center-coordinates and the labels. \
                   Give each cluster a room name. \
                2. Name the 3 most likely places (combination of furniture and relation to furniture) with descending likelihood order, where the item can be found. \
                   Also consider the room. \
                3. Return the result as a json in the following format: \
                   {{\"item\", {{\"furniture_id\", \"furniture_name\", \"relation\", \"room\"}}}}"

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
    result_path = os.path.join(config.get_subpath("scene_graph"), config["pre_scanned_graphs"]["high_res"], f"{item.replace(' ', '_')}.json")
    with open(result_path, 'w') as file:
        json.dump(json_data, file, indent=4)


if __name__ == "__main__":
   ask_deepseek("image frame")