# simulate_real_household_histories.py
import random
from datetime import datetime, timedelta
from pathlib import Path

from furniture_setup import init_furniture_scene, get_furniture_ids
from object_priors import sample_objects, SEMANTIC_PRIORS
from user_profiles import get_profile
from household_io import write_gt_day, write_seen_day
from st_graphs import update_object_st_graph, update_household_profile

# -----------------------------
# Simulation Parameters
# -----------------------------
NUM_DAYS = 60
PROFILE_NAME = "messy_single"  # change to run different households
OUTPUT_ROOT = Path(f"/home/rohit/workspace/ros2/stretch-compose/data/output_household/{PROFILE_NAME}")
START_DATE = datetime(2025, 1, 1)


RANDOM_SEED = 42

# -----------------------------
# Helper Functions
# -----------------------------
def maybe_perturb_profile(profile: dict, prob: float = 0.15, strength: float = 0.3):
    """
    With probability `prob`, perturb the profile parameters randomly by ±strength.
    """
    if random.random() > prob:
        return profile
    perturbed = {}
    for k, v in profile.items():
        delta = v * strength
        perturbed[k] = min(max(v + random.uniform(-delta, delta), 0.0), 1.0)
    return perturbed

def assign_initial_objects(objects, priors, furniture_ids):
    """
    Assign sampled objects to furniture using semantic priors.
    """
    placements = {}
    oid_counter = 100
    objs_json = {}
    node_ids = []
    node_labels = []
    connections = {}

    for obj in objects:
        oid_counter += 1
        oid = oid_counter
        choices = priors.get(obj, list(furniture_ids.keys()))
        furn_choice = random.choice(choices)
        fid = furniture_ids[furn_choice]

        # Minimal object JSON
        data = {
            "id": oid,
            "label": obj,
            "centroid": [random.random(), random.random(), 0.5],
            "dimensions": [0.1, 0.1, 0.1],
            "pose": [[1,0,0,0],[0,1,0,0],[0,0,1,0.5],[0,0,0,1]],
            "drawer": -1,
            "confidence": round(random.uniform(0.6, 0.95), 3)
        }
        objs_json[oid] = data
        node_ids.append(oid)
        node_labels.append(obj)
        connections[str(oid)] = fid
        placements[oid] = fid
    return objs_json, node_ids, node_labels, connections, placements

# -----------------------------
# Main Simulation
# -----------------------------
def simulate():
    random.seed(RANDOM_SEED)
    last_seen_days = {}

    # Base furniture
    base_scene = init_furniture_scene(OUTPUT_ROOT / "base_scene.json")
    furniture_ids = get_furniture_ids()

    # Sample objects
    sampled_objects = sample_objects()
    objs_json, node_ids, node_labels, connections, placements = assign_initial_objects(
        sampled_objects, SEMANTIC_PRIORS, furniture_ids
    )

    # Graph bootstrap
    graph = {
        "node_ids": list(furniture_ids.values()) + node_ids,
        "node_labels": list(furniture_ids.keys()) + node_labels,
        "connections": connections,
        "immovable_ids": furniture_ids,
        "immovable_labels": list(furniture_ids.keys())
    }

    # Profile
    profile_base = get_profile(PROFILE_NAME, random_seed=RANDOM_SEED)

    # Seen starts as GT on day 0
    prev_seen_graph = dict(graph)

    # Simulation days
    for d in range(NUM_DAYS):
        date = (START_DATE + timedelta(days=d)).strftime("%Y_%m_%d")

        # Perturb profile some days
        profile_today = maybe_perturb_profile(profile_base)

        # GT folder
        gt_day = OUTPUT_ROOT / "gt" / date
        seen_day = OUTPUT_ROOT / "seen" / date

        # --- Simulate relocations ---
        coverage_gt = []
        relocation_events = {}

        for oid, obj in objs_json.items():
            if random.random() < profile_today["reloc_prob"]:
                old_fid = connections[str(oid)]
                choices = SEMANTIC_PRIORS.get(obj["label"], list(furniture_ids.keys()))
                new_fid = furniture_ids[random.choice(choices)]
                connections[str(oid)] = new_fid
                relocation_events[oid] = {"from": old_fid, "to": new_fid}
                coverage_gt.append({
                    "id": oid,
                    "label": obj["label"],
                    "status": "relocated",
                    "furniture_id": new_fid,
                    "drawer_id": obj.get("drawer", -1),
                    "centroid": obj["centroid"],
                    "confidence": obj["confidence"]
                })
            else:
                coverage_gt.append({
                    "id": oid,
                    "label": obj["label"],
                    "status": "observed",
                    "furniture_id": connections[str(oid)],
                    "drawer_id": obj.get("drawer", -1),
                    "centroid": obj["centroid"],
                    "confidence": obj["confidence"]
                })

        # --- Write GT ---
        write_gt_day(base_scene, graph, objs_json, coverage_gt, gt_day)

        for oid, obj in objs_json.items():
            update_object_st_graph(oid, obj["label"], date,
                                   furniture_id=connections[str(oid)],
                                   drawer_id=obj.get("drawer", -1),
                                   centroid=obj["centroid"],
                                   confidence=obj["confidence"],
                                   base_dir=gt_day,
                                   is_seen=False)

        update_household_profile(date, relocation_events, gt_day, is_seen=False)

        # --- Seen: only partial observation ---
        observed_today = random.sample(list(objs_json.keys()),
                                       k=int(len(objs_json) * 0.6))  # 60% seen

        last_seen_days = write_seen_day(prev_seen_graph, graph, observed_today, objs_json,
                       base_scene, seen_day, last_seen_days, d)

        for oid in observed_today:
            obj = objs_json[oid]
            update_object_st_graph(oid, obj["label"], date,
                                   furniture_id=connections[str(oid)],
                                   drawer_id=obj.get("drawer", -1),
                                   centroid=obj["centroid"],
                                   confidence=obj["confidence"],
                                   base_dir=seen_day,
                                   is_seen=True)

        update_household_profile(date, {oid: relocation_events.get(oid, {}) for oid in observed_today},
                                 seen_day, is_seen=True)

        prev_seen_graph = graph  # carry forward

if __name__ == "__main__":
    simulate()
