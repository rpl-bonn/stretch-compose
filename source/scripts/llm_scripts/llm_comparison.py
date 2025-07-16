import json
from openai import OpenAI
import os
import time

from utils.environment import _load_environment_file
from utils.recursive_config import Config
from utils.time import convert_time


config = Config()
# User messages
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

# System messages
# bonn environment, object not given in the scene            
SYSTEM1 = f"You are a semantic mapping agent. You are given a scene graph of the environment in json format and a user query. \
            Your task is to identify, which item the user is searching for. \
            Then, your task is to find the most likely furniture in the scene graph for this item, where it might be located. \
            Return the result as a json (all lowercase) in the following format: \
            {{\"item\", \"furniture_id\", \"furniture_name\", \"relation_to_furniture\", \"summary of reasoning\"}}"
            
# zurich environment, object already given in the scene                    
SYSTEM2 = f"You are a spatial reasoning agent. You are given a scene graph of the environemnt in json format. \
            Based on the scene graph, answer the given question. Ignore ingoing/outgoing and just focus on objects and their centroids.\
            Note that the coordinate system is right handed, with z axis pointing up. \
            After reasoning, you will return the exact object and its ID in the scene graph, that is most RELEVANT to the given question. \
            Return the result as a json (all lowercase) in the following format: \
            {{0, \"object_type\"}}, {{1, \"object_id\"}}, {{2, \"summary of reasoning\"}}"


def set_openai_key(config: Config) -> None:
    api_data = _load_environment_file(config)["api"]
    return api_data["openai"]["key"]

def set_deepseek_key(config: Config) -> None:
    api_data = _load_environment_file(config)["api"]
    return api_data["deepseek"]["key"]

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
        response_format={
            'type': 'json_object'
        },
        temperature=0.6,
        top_p=0.7,
        max_tokens=4096,
        stream=True
    )
    timer_end1 = time.time_ns()
    s1 = round((timer_end1 - timer_start) / 1e9, 2)
    print(f"R1 1st time: {s1}s")
    token_cnt = 0
    response = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            response += chunk.choices[0].delta.content
            token_cnt += 1
            print(chunk.choices[0].delta.content, end="")
    timer_end2 = time.time_ns()
    s2 = round((timer_end2 - timer_start) / 1e9, 2)
    print(f"R1 2nd time: {s2}s")
    print(f"Tokens: {token_cnt}")
    print(response.split("</think>\n\n")[1])
    #print("Result:\n", completion.choices[0].message.content.split("</think>\n")[1])
    return s1, s2, token_cnt
  
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
        response_format={
            'type': 'json_object'
        },
        max_completion_tokens=4096,
    )
    timer_end = time.time_ns()
    seconds = round((timer_end - timer_start) / 1e9, 2)
    print(f"o1 timing: {seconds}s")
    tokens = completion.usage.completion_tokens
    print(tokens)
    print("Result:\n", completion.choices[0].message.content)
    return seconds, tokens

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
        response_format={
            'type': 'json_object'
        },
        max_completion_tokens=4096,
    )
    timer_end = time.time_ns()
    seconds = round((timer_end - timer_start) / 1e9, 2)
    print(f"o3 timing: {seconds}s")
    tokens = completion.usage.completion_tokens
    print(tokens)
    print("Result:\n", completion.choices[0].message.content)
    return seconds, tokens

if __name__ == "__main__":
    users = [USER1, USER2, USER3, USER4]
    jsons = [JSON1, JSON2]
    systems = [SYSTEM1, SYSTEM2]
    t1 = 0
    t2 = 0
    tkn = 0
    cnt = 0
    # 1st environment (json)
    for i in range(10):
        for u in users:
            j = jsons[0]
            s = systems[0]
            print("\nBonn environment:", u)
            #t1d, t2d, tknd = deepseek(u, j, s)
            #t1d, tknd = openai_o1(u, j, s)
            t1d, tknd = openai_o3(u, j, s)
            t1 += t1d
            #t2 += t2d
            tkn += tknd
            cnt += 1
    # 2nd environment (json)
    for i in range(3):
        for u in users:
            j = jsons[1]
            s = systems[1]
            print("\nZurich environment: ", u)
            t1d, t2d, tknd = deepseek(u, j, s)
            #t1d, tknd = openai_o1(u, j, s)
            #t1d, tknd = openai_o3(u, j, s)
            t1 += t1d
            t2 += t2d
            tkn += tknd
            cnt += 1

    print(f"Average time 1: {t1}, {round(t1/cnt, 2)}s")
    print(f"Average time 2: {t2}, {round(t2/cnt, 2)}s")
    print(f"Average tokens: {tkn}, {round(tkn/cnt, 2)}")
