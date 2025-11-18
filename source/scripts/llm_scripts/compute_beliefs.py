import json
import math
from pathlib import Path
from collections import Counter, defaultdict
# Parameters
DECAY = 0.92        # exponential decay for history
HISTORY_BLEND_K = 5 # how many days before α=1

ROOT_DIR = Path("/home/rohit/workspace/ros2/stretch-compose/data/output_household/messy_single/seen/2025_03_01/").resolve()
GRAPH_FP = ROOT_DIR / "graph.json"
COVERAGE_FP = ROOT_DIR / "coverage.json"
SCENE_FP = ROOT_DIR / "scene.json"
PROFILE_FP = ROOT_DIR / "household_profile.json"
ST_DIR = ROOT_DIR / "st_graphs"

# ---------------- Household Beliefs ----------------
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

def compute_household_belief(graph_fp: Path, scene_fp: Path, profile_fp: Path):
    """
    Compute household-level belief distribution, entropy, and type, with normalization.
    """
    with open(graph_fp) as f:
        graph = json.load(f)
    with open(scene_fp) as f:
        scene = json.load(f)
    with open(profile_fp) as f:
        profile = json.load(f)

    # Aggregate relocations from household profile
    reloc_counts = Counter()
    for day in profile.get("history", []):
        for _, move in day.get("relocations", {}).items():
            if "to" in move:
                reloc_counts[move["to"]] += 1

    total_relocations = sum(reloc_counts.values())
    total_days = len(profile.get("history", []))

    if total_relocations == 0 or total_days == 0:
        return {"belief": {}, "entropy": 0.0, "type": "tidy/unimodal", "room_entropies": {}}

    # Calculate household belief distribution and raw entropy
    dist = {fid: c / total_relocations for fid, c in reloc_counts.items()}
    raw_entropy = -sum(p * math.log(p + 1e-9) for p in dist.values())

    # Normalize entropy by the number of days
    normalized_entropy = raw_entropy / total_days
    
    print("Household relocation distribution:", dist)
    print("Raw household entropy:", raw_entropy)
    print("Normalized household entropy:", normalized_entropy)

    # Classify household
    if normalized_entropy < 0.05:
        htype = "tidy/unimodal"
    elif normalized_entropy < 0.15:
        htype = "habitual/bimodal"
    else:
        htype = "messy/multimodal"

    print("Household type:", htype)
    # Assign furniture to correct rooms
    room_furniture = {
        "living_room": ["sofa", "tv_stand", "bookshelf_kallax", "couchtisch", "round_table", "small_table", "chair"],
        "bedroom": ["bed", "nightstand_left", "nightstand_right", "wardrobe", "dressing_table"],
        "hallway": ["shoe_rack"]
    }

    rooms = defaultdict(list)
    for furn in scene.get("furniture", []):
        for room_name, furn_list in room_furniture.items():
            if furn["label"] in furn_list:
                rooms[room_name].append(furn["id"])
                break

    # Calculate and normalize room entropies
    room_entropies = {}
    for rname, rnodes in rooms.items():
        mass = [dist.get(fid, 0) for fid in rnodes]
        Z = sum(mass)
        if Z > 0:
            norm = [m / Z for m in mass]
            raw_rent = -sum(p * math.log(p + 1e-9) for p in norm)
            normalized_rent = raw_rent / total_days
        else:
            normalized_rent = 0
        room_entropies[rname] = normalized_rent

    print("Normalized room entropies:", room_entropies)

    return {
        "belief": dist,
        "entropy": normalized_entropy,
        "type": htype,
        "room_entropies": room_entropies
    }


# ---------------- Object Beliefs ----------------
def compute_object_belief(obj_name: str,
                          graph_fp: Path,
                          coverage_fp: Path,
                          st_dir: Path,
                          household_belief: dict,
                          day_index: int):
    """
    Compute posterior belief for a given object.
    """
    with open(graph_fp) as f:
        graph = json.load(f)
    with open(coverage_fp) as f:
        coverage = json.load(f)

    # if object not in graph: unseen → call LLM
    if obj_name not in graph["node_labels"]:
        return {
            "object": obj_name,
            "posterior": {},
            "object_entropy": None,
            "household_entropy": household_belief["entropy"],
            "household_type": household_belief["type"],
            "should_call_llm": True,
            "note": "Object not in graph, unseen"
        }

    idx = graph["node_labels"].index(obj_name)
    oid = graph["node_ids"][idx]
    st_path = Path(st_dir) / f"object_{oid}.json"
    if not st_path.exists():
        return {
            "object": obj_name,
            "posterior": {},
            "object_entropy": None,
            "household_entropy": household_belief["entropy"],
            "household_type": household_belief["type"],
            "should_call_llm": True,
            "note": "No spatio-temporal history"
        }

    with open(st_path) as f:
        st_graph = json.load(f)

    # collect counts with decay
    counts = Counter()
    history = st_graph.get("history", [])
    for i, entry in enumerate(history):
        fid = entry["furniture_id"]
        delta = max(0, day_index - i)  # crude offset by index
        weight = (DECAY ** delta) * entry.get("confidence", 1.0)
        counts[fid] += weight

    total = sum(counts.values())
    posterior = {fid: c / total for fid, c in counts.items()} if total > 0 else {}
    obj_entropy = -sum(p * math.log(p + 1e-9) for p in posterior.values())

    # recency penalty from coverage
    recency_penalty = 1.0
    for entry in coverage:
        if entry["id"] == oid:
            days_since = entry.get("last_seen_since_days", 0)
            recency_penalty = DECAY ** days_since
            break

    # blend with household belief
    alpha = min(1.0, len(history) / HISTORY_BLEND_K)
    blended = {}
    for fid in set(list(posterior.keys()) + list(household_belief["belief"].keys())):
        po = posterior.get(fid, 0)
        ph = household_belief["belief"].get(fid, 0)
        blended[fid] = recency_penalty * (alpha * po + (1 - alpha) * ph)

    Z = sum(blended.values())
    if Z > 0:
        blended = {fid: p / Z for fid, p in blended.items()}

    should_call_llm = (len(blended) == 0 or obj_entropy > 1.5)

    return {
        "object": obj_name,
        "posterior": blended,
        "object_entropy": obj_entropy,
        "household_entropy": household_belief["entropy"],
        "household_type": household_belief["type"],
        "room_entropies": household_belief["room_entropies"],
        "should_call_llm": should_call_llm
    }


# ---------------- Batch ----------------
def compute_all_objects_beliefs(graph_fp: Path,
                                coverage_fp: Path,
                                scene_fp: Path,
                                profile_fp: Path,
                                st_dir: Path,
                                day_index: int):
    """
    Compute beliefs for all objects in the graph.
    """
    household_belief = compute_household_belief(graph_fp, scene_fp, profile_fp)
    results = {}
    with open(graph_fp) as f:
        graph = json.load(f)
    for obj_name in graph["node_labels"]:
        res = compute_object_belief(obj_name, graph_fp, coverage_fp, st_dir,
                                    household_belief, day_index)
        results[obj_name] = res
        print(json.dumps(res, indent=2))
    return results


# ---------------- CLI ----------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--object", type=str, help="Object name to query (else all)")
    parser.add_argument("--graph", type=Path, default=GRAPH_FP)
    parser.add_argument("--coverage", type=Path, default=COVERAGE_FP)
    parser.add_argument("--scene", type=Path, default=SCENE_FP)
    parser.add_argument("--profile", type=Path, default=PROFILE_FP)
    parser.add_argument("--st", type=Path, default=ST_DIR)
    parser.add_argument("--day_index", type=int, default=60)
    args = parser.parse_args()

    if args.object:
        household_belief = compute_household_belief(args.graph, args.scene, args.profile)
        res = compute_object_belief(args.object, args.graph, args.coverage,
                                    args.st, household_belief, args.day_index)
        print(json.dumps(res, indent=2))
    else:
        compute_all_objects_beliefs(args.graph, args.coverage, args.scene,
                                    args.profile, args.st, args.day_index)
