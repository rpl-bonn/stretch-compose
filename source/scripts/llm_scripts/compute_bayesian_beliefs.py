import json
import math
from collections import defaultdict, Counter
from pathlib import Path

# Parameters
P_HIT = 0.8
P_MISS = 0.2
DECAY = 0.95 # Adjusted for better log-odds behavior
MAX_LO = 5.0
HISTORY_BLEND_K = 10 # Parameter for history blending

ROOT_DIR = Path("/home/rohit/workspace/ros2/stretch-compose/data/output_household/tidy_single/gt/2025_03_01/").resolve()
GRAPH_FP = ROOT_DIR / "graph.json"
COVERAGE_FP = ROOT_DIR / "coverage.json"
SCENE_FP = ROOT_DIR / "scene.json"
PROFILE_FP = ROOT_DIR / "household_profile.json"
ST_DIR = ROOT_DIR / "st_graphs"

import json
import math
from collections import defaultdict, Counter
from pathlib import Path

# Parameters
P_HIT = 0.8
P_MISS = 0.2
DECAY = 0.95 # Adjusted for better log-odds behavior
MAX_LO = 5.0

def logodds(p):
    return math.log(p / (1 - p + 1e-9))

def sigmoid(lo):
    return 1 / (1 + math.exp(-lo))

def compute_entropy(dist: dict):
    """Normalized Shannon entropy"""
    probs = [v for v in dist.values() if v > 0]
    if not probs:
        return 0.0
    
    Z = sum(probs)
    if Z == 0:
        return 0.0

    probs = [p / Z for p in probs]
    H = -sum(p * math.log(p + 1e-9) for p in probs)
    return H / math.log(len(probs) + 1e-9)

def likelihoods(state_H, trans_H):
    """
    Likelihoods of observing entropies given each household type.
    Using Gaussian-like bumps in entropy space.
    """
    def gaussian(x, mu, sigma):
        return math.exp(-0.5 * ((x - mu) / sigma) ** 2)

    return {
        "tidy/unimodal": gaussian(state_H, 0.1, 0.1) * gaussian(trans_H, 0.1, 0.1),
        "habitual/bimodal": gaussian(state_H, 0.4, 0.15) * gaussian(trans_H, 0.3, 0.15),
        "messy/multimodal": gaussian(state_H, 0.8, 0.1) * gaussian(trans_H, 0.7, 0.1),
    }

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

def compute_bayesian_household_belief(graph_fp: Path, scene_fp: Path, profile_fp: Path):
    """
    Compute household-level belief using a Bayesian approach, conditioning on history length and object coverage.
    """
    with open(graph_fp) as f:
        graph = json.load(f)
    with open(scene_fp) as f:
        scene = json.load(f)
    with open(profile_fp) as f:
        profile = json.load(f)

    # --- Step 1: Calculate Normalized Entropy based on True Relocations ---
    reloc_counts = Counter()
    seen_objects = set()
    for day in profile.get("history", []):
        for oid, move in day.get("relocations", {}).items():
            seen_objects.add(oid)
            if "to" in move and "from" in move and move["to"] != move["from"]:
                reloc_counts[move["to"]] += 1

    total_relocations = sum(reloc_counts.values())
    total_days = len(profile.get("history", []))

    if total_relocations == 0 or total_days == 0:
        normalized_entropy = 0.0
    else:
        dist = {fid: c / total_relocations for fid, c in reloc_counts.items()}
        raw_entropy = -sum(p * math.log(p + 1e-9) for p in dist.values())
        normalized_entropy = raw_entropy / total_days

    # --- Step 2: Calculate Confidence based on History and Coverage ---
    total_movable_objects = len(graph["node_ids"]) - len(graph["immovable_ids"])
    # A simple confidence factor based on the proportion of objects covered and history length
    # A longer history and higher coverage lead to higher confidence
    history_confidence = min(1.0, total_days / 21) # Max confidence at 21 days
    coverage_confidence = min(1.0, len(seen_objects) / total_movable_objects)
    confidence_factor = (history_confidence + coverage_confidence) / 2

    # --- Step 3: Define Priors and Likelihood Functions ---
    priors = {
        "tidy/unimodal": 1/3,
        "habitual/bimodal": 1/3,
        "messy/multimodal": 1/3
    }
    
    # Likelihood functions (simplified, based on normalized entropy ranges)
    # A Gaussian or similar distribution could be used for a more complex model
    def get_likelihood(htype, entropy):
        if htype == "tidy/unimodal":
            if entropy < 0.05: return 0.95
            elif entropy < 0.15: return 0.5
            else: return 0.05
        elif htype == "habitual/bimodal":
            if entropy < 0.05: return 0.2
            elif entropy < 0.15: return 0.8
            else: return 0.2
        elif htype == "messy/multimodal":
            if entropy < 0.05: return 0.05
            elif entropy < 0.15: return 0.4
            else: return 0.95
        return 0

    # --- Step 4: Calculate Posterior Belief and Final Classification ---
    raw_posterior = {}
    for htype, prior in priors.items():
        likelihood = get_likelihood(htype, normalized_entropy)
        raw_posterior[htype] = prior * likelihood

    total_posterior = sum(raw_posterior.values())
    
    # The final belief is a blend of the uniform prior and the calculated posterior
    final_belief = {
        htype: ((1 - confidence_factor) * prior) + (confidence_factor * (raw_posterior.get(htype, 0) / total_posterior))
        for htype, prior in priors.items()
    }
    
    Z = sum(final_belief.values())
    final_belief = {k: v / Z for k, v in final_belief.items()}

    final_type = max(final_belief, key=final_belief.get)

    # --- Step 5: Calculate and Normalize Room Entropies (for consistency) ---
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
    room_entropies = {}
    if total_relocations > 0:
        dist = {fid: c / total_relocations for fid, c in reloc_counts.items()}
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

    # --- Step 6: Return the Final Result ---
    return {
        "normalized_entropy": normalized_entropy,
        "posterior_belief": final_belief,
        "type": final_type,
        "room_entropies": room_entropies,
        "total_days": total_days,
        "unique_objects_seen": len(seen_objects),
        "total_movable_objects": total_movable_objects
    }

# Execute the function with the provided file paths and print the output
# Example usage with corrected code and data paths
graph_fp = GRAPH_FP
scene_fp = SCENE_FP
profile_fp = PROFILE_FP

result = compute_bayesian_household_belief(graph_fp, scene_fp, profile_fp)
print(json.dumps(result, indent=4))