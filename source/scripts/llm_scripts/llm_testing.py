import json
from openai import OpenAI
import os
import time
from utils.environment import _load_environment_file
from utils.recursive_config import Config
from utils.time import convert_time

NO_ROOMS = 2

config = Config()
USER1 = "I need to water my plant"
USER2 = "I would love to read my book"
USER3 = "I can't find the cat plushy anymore"
USER4 = "I want to drink some water"

json_path = os.path.join(config.get_subpath("scene_graph"), config["pre_scanned_graphs"]["high_res"], "scene_graph.json")
with open(json_path, 'r') as file:
    json_data = json.load(file)
JSON1 = json.dumps(json_data)

json_path = os.path.join(config.get_subpath("scene_graph"), config["pre_scanned_graphs"]["high_res"], "scene_graph_2.json")
with open(json_path, 'r') as file:
    json_data = json.load(file)
JSON2 = json.dumps(json_data)

# both environments, include room clustering
SYSTEM1 = f"Instructions: The user gives you a query and a scene graph with furniture of the environment in json format. \
            1. Identify which objects are necessary to fulfill the task (without looking at the scene graph!) \
            2. Cluster the given scene_graph into 2 rooms using the labels and centroid coordinates of the furniture (give each cluster a room name) \
            3. Find the 3 most likely furniture places in the scene graph for each object determined in Step 1 (also consider the room)\
            4. Return the result as a json (all lowercase) for each object in the following format: \
            {{\"object\", [\"furniture_id\", \"furniture_name\", \"relation_to_furniture\", \"room\"]}}"

# zurich environment, object already given in the scene                    
SYSTEM2 = f"You are a spatial reasoning agent. You are given a scene graph of the environemnt in json format. \
            Based on the scene graph, answer the given question. Ignore ingoing/outgoing and just focus on objects and their centroids.\
            Note that the coordinate system is right handed, with z axis pointing up. \
            After reasoning, you will return the exact object and its ID in the scene graph, that is most RELEVANT to the given question. \
            Return the result as a json (all lowercase) in the following format: \
            {{0, \"object_type\"}}, {{1, \"object_id\"}}, {{2, \"summary of reasoning\"}}"

# bonn environment, object not given in the scene            
SYSTEM3 = f"You are a semantic mapping agent. You are given a scene graph of the environment in json format and a user query. \
            Your task is to identify, which (different types of) items the user needs to fulfill the query. \
            Then, your task is to find the most likely place (furniture) in the scene graph for each item, where it might be located. \
            Return the result as a json (all lowercase) in the following format: \
            {{\"item\", \"furniture_id\", \"furniture_name\", \"relation_to_furniture\", \"summary of reasoning\"}}"


def set_openai_key(config: Config) -> None:
    api_data = _load_environment_file(config)["api"]
    return api_data["openai"]["key"]

def set_deepseek_key(config: Config) -> None:
    api_data = _load_environment_file(config)["api"]
    return api_data["deepseek"]["key"]

def openai_o3(user, json, system):
    client = OpenAI(api_key = set_openai_key(config))
    timer_start = time.time_ns()
    completion = client.chat.completions.create(
        model="o3-mini",
        #reasoning_effort="high",
        messages=[
            {
                "role": "developer",
                "content": system,
            },
            {
                "role": "user",
                "content": 
                [
                    {
                        "type": "text",
                        "text": json,  
                    },
                    {
                        "type": "text",
                        "text": user,
                    },
                ],
            },
        ],
        #max_completion_tokens=4096,
    )
    timer_end = time.time_ns()
    seconds = round((timer_end - timer_start) / 1e9, 2)
    print(f"o3 timing: {seconds}s")
    print("Result:\n", completion.choices[0].message.content)
    return seconds
    
def openai_o1(user, json, system):
    client = OpenAI(api_key = set_openai_key(config))
    timer_start = time.time_ns()
    completion = client.chat.completions.create(
        model="o1",
        #reasoning_effort="high",
        messages=[
            {
                "role": "developer",
                "content": system,
            },
            {
                "role": "user",
                "content": 
                [
                    {
                        "type": "text",
                        "text": json,  
                    },
                    {
                        "type": "text",
                        "text": user,
                    },
                ],
            },
        ],
        #max_completion_tokens=4096,
    )
    timer_end = time.time_ns()
    seconds = round((timer_end - timer_start) / 1e9, 2)
    print(f"o1 timing: {seconds}s")
    print("Result:\n", completion.choices[0].message.content)
    return seconds

def deepseek(user, json, system):
    client = OpenAI(base_url = "https://integrate.api.nvidia.com/v1", api_key = set_deepseek_key(config))
    timer_start = time.time_ns()
    completion = client.chat.completions.create(
        model="deepseek-ai/deepseek-r1",
        messages=[
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": 
                [
                    {
                        "type": "text",
                        "text": json, 
                    },
                    {
                        "type": "text",
                        "text": user,
                    },
                ],
            },
        ],
        temperature=0.6,
        top_p=0.7,
        max_tokens=4096,
        stream=False
    )
    timer_end = time.time_ns()
    seconds = round((timer_end - timer_start) / 1e9, 2)
    print(f"R1 timing: {seconds}s")
    print("Result:\n", completion.choices[0].message.content.split("</think>\n")[1])
    return seconds


if __name__ == "__main__":
    users = [USER1, USER2, USER3, USER4] # 
    jsons = [JSON1, JSON2]
    systems = [SYSTEM1, SYSTEM2, SYSTEM3]
    dt = 0
    o1t = 0
    o3t = 0
    cnt = 0
    # # 1st environment (json)
    # for u in users:
    #     j = jsons[0]
    #     s = systems[2]
    #     print("\n1st:", u)
    #     dt += deepseek(u, j, s)
    #     o1t += openai_o1(u, j, s)
    #     o3t += openai_o3(u, j, s)
    #     cnt += 1
    # # 2nd environment (json)
    # for u in users:
    #     j = jsons[1]
    #     s = systems[1]
    #     print("\n2nd: ", u)
    #     dt += deepseek(u, j, s)
    #     o1t += openai_o1(u, j, s)
    #     o3t += openai_o3(u, j, s)
    #     cnt += 1
    # Both environments (json)
    for u in users:
        for j in jsons:
            s = systems[0]
            print("\nBoth: ", u)
            dt += deepseek(u, j, s)
            o1t += openai_o1(u, j, s)
            o3t += openai_o3(u, j, s)
            cnt += 1
    print(f"\n\nAverage timing for Deepseek: {round(dt/cnt, 2)}s")
    print(f"Average timing for OpenAI o1: {round(o1t/cnt, 2)}s")
    print(f"Average timing for OpenAI o3: {round(o3t/cnt, 2)}s")
