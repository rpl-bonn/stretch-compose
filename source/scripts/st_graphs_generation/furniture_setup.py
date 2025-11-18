# furniture_setup.py
import json
from pathlib import Path

# IKEA-like dimensions in meters (W × D × H)
FURNITURE = [
    {"id": 1, "label": "sofa", "dimensions": [2.2, 0.9, 0.85]},
    {"id": 2, "label": "tv_stand", "dimensions": [1.8, 0.4, 0.5]},
    {"id": 3, "label": "bookshelf_kallax", "dimensions": [1.47, 0.39, 1.47], "drawers": 2, "doors": 2},
    {"id": 4, "label": "couchtisch", "dimensions": [0.9, 0.55, 0.45], "drawers": 1},
    {"id": 5, "label": "round_table", "dimensions": [1.05, 1.05, 0.74]},
    {"id": 6, "label": "shoe_rack", "dimensions": [1.07, 0.3, 0.89]},
    {"id": 7, "label": "small_table", "dimensions": [0.55, 0.55, 0.45]},
    {"id": 8, "label": "chair", "dimensions": [0.42, 0.5, 0.9]},
    {"id": 9, "label": "bed", "dimensions": [2.0, 1.6, 0.45]},
    {"id": 10, "label": "nightstand_left", "dimensions": [0.4, 0.55, 0.55], "drawers": 1},
    {"id": 11, "label": "nightstand_right", "dimensions": [0.4, 0.55, 0.55], "drawers": 1},
    {"id": 12, "label": "wardrobe", "dimensions": [1.0, 0.6, 2.0]},
    {"id": 13, "label": "dressing_table", "dimensions": [1.2, 0.5, 0.8], "drawers": 4}
]

def init_furniture_scene(output_path: Path):
    """
    Initialize scene.json with all furniture.
    """
    scene = {"furniture": []}
    for f in FURNITURE:
        entry = {
            "id": f["id"],
            "label": f["label"],
            "centroid": [0.0, 0.0, f["dimensions"][2] / 2],  # placeholder centroids
            "dimensions": f["dimensions"],
            "pose": [
                [1, 0, 0, 0.0],
                [0, 1, 0, 0.0],
                [0, 0, 1, f["dimensions"][2] / 2],
                [0, 0, 0, 1]
            ]
        }
        scene["furniture"].append(entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scene, f, indent=4)
    return scene

def get_furniture_ids():
    """
    Return mapping from furniture labels to IDs.
    """
    return {f["label"]: f["id"] for f in FURNITURE}
