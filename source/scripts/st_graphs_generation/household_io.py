# household_io.py
import json
from pathlib import Path
from typing import Dict, List, Any

# -------------------------
# Basic writers
# -------------------------
def write_scene(scene: Dict[str, Any], out_dir: Path):
    """
    Write scene.json (furniture only).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "scene.json", "w") as f:
        json.dump(scene, f, indent=4)

def write_graph(node_ids: List[int], node_labels: List[str],
                connections: Dict[str, int],
                immovable_ids: Dict[str, int],
                immovable_labels: List[str],
                out_dir: Path):
    """
    Write graph.json.
    """
    graph = {
        "node_ids": node_ids,
        "node_labels": node_labels,
        "connections": connections,
        "immovable_ids": immovable_ids,
        "immovable_labels": immovable_labels
    }
    with open(out_dir / "graph.json", "w") as f:
        json.dump(graph, f, indent=4)

def write_objects(objects: Dict[int, Dict[str, Any]], out_dir: Path):
    """
    Write per-object JSON files into objects/ folder.
    """
    obj_dir = out_dir / "objects"
    obj_dir.mkdir(parents=True, exist_ok=True)
    for oid, data in objects.items():
        with open(obj_dir / f"{oid}.json", "w") as f:
            json.dump(data, f, indent=4)

def write_coverage(entries: List[Dict[str, Any]], out_dir: Path):
    """
    Write coverage.json with observed/relocated/added objects for the day.
    """
    with open(out_dir / "coverage.json", "w") as f:
        json.dump(entries, f, indent=4)

# -------------------------
# High-level wrappers
# -------------------------
def write_gt_day(scene: Dict[str, Any],
                 graph: Dict[str, Any],
                 objects: Dict[int, Dict[str, Any]],
                 coverage: List[Dict[str, Any]],
                 out_dir: Path,
                 day_index: int = 0):
    """
    Write full ground-truth day: scene, graph, objects, coverage.
    """
    write_scene(scene, out_dir)
    write_graph(graph["node_ids"], graph["node_labels"],
                graph["connections"], graph["immovable_ids"],
                graph["immovable_labels"], out_dir)
    write_objects(objects, out_dir)
    write_coverage(coverage, out_dir)

def write_seen_day(prev_seen_graph: Dict[str, Any],
                   gt_graph: Dict[str, Any],
                   observed: List[int],
                   gt_objects: Dict[int, Dict[str, Any]],
                   scene: Dict[str, Any],
                   out_dir: Path,
                   last_seen_days: Dict[str, int],
                   day_index: int):
    """
    Write seen day: update only observed objects in graph and objects folder.
    Furniture (scene.json) is unchanged.
    prev_seen_graph: last seen graph.json (dict)
    gt_graph: today's ground truth graph.json (dict)
    observed: list of object IDs seen today
    gt_objects: today's ground truth objects (dict of oid->json)
    scene: unchanged furniture from gt
    """
    # Start from previous seen graph
    seen_graph = dict(prev_seen_graph)
    seen_objects = {}
    coverage = []
    
    print("Day index:", day_index)
    # keep track of last seen days in graph

    # For each object in ground truth
    for oid, obj in gt_objects.items():
        str_oid = str(oid)

        if oid in observed:
            # Update connections in graph
            seen_graph["connections"][str_oid] = gt_graph["connections"][str_oid]
            if oid == 101:
                print("Object ", oid, "seen today")
            # Add object JSON
            seen_objects[oid] = obj
            last_seen_days[str_oid] = day_index

            coverage.append({
                "id": oid,
                "label": obj["label"],
                "last_seen_since_days": 0,
                "furniture_id": gt_graph["connections"][str_oid],
                "drawer_id": obj.get("drawer", -1),
                "centroid": obj["centroid"],
                "confidence": obj["confidence"]
            })
        else:
            # Keep whatever was last known in seen graph
            if str_oid in seen_graph["connections"]:
                days_since =  (day_index - last_seen_days[str_oid]
                          if str_oid in last_seen_days else -1)
                if oid == 101:
                    print("Object", oid, "not seen, last seen days:", days_since)
                coverage.append({
                    "id": oid,
                    "label": obj["label"],
                    "last_seen_since_days": days_since,
                    "furniture_id": seen_graph["connections"][str_oid],
                    "drawer_id": obj.get("drawer", -1)
                })

    # Write outputs
    write_scene(scene, out_dir)  # furniture stays same
    write_graph(seen_graph["node_ids"], seen_graph["node_labels"],
                seen_graph["connections"], seen_graph["immovable_ids"],
                seen_graph["immovable_labels"], out_dir)
    write_objects(seen_objects, out_dir)
    write_coverage(coverage, out_dir)
    
    return last_seen_days
