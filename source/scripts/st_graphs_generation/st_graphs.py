# st_graphs.py
import json
from pathlib import Path
from typing import Dict, Any

def _load_json(path: Path):
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None

def update_object_st_graph(obj_id: int,
                           obj_label: str,
                           date: str,
                           furniture_id: int,
                           drawer_id: int,
                           centroid: Any,
                           confidence: float,
                           base_dir: Path,
                           is_seen: bool):
    """
    Update spatio-temporal graph for a single object.
    base_dir: root path (gt/ or seen/)
    is_seen: whether this update is for seen or gt
    """
    root = base_dir.parent
    st_dir = root / "st_graphs"
    st_dir.mkdir(parents=True, exist_ok=True)
    st_dir_daily = base_dir / "st_graphs"
    st_dir_daily.mkdir(parents=True, exist_ok=True)
    fname = st_dir / f"object_{obj_id}.json"

    data = _load_json(fname)
    if data is None:
        data = {
            "id": obj_id,
            "label": obj_label,
            "history": []  # list of daily placements
        }

    entry = {
        "date": date,
        "furniture_id": furniture_id,
        "drawer_id": drawer_id,
        "centroid": centroid,
        "confidence": confidence,
        "source": "seen" if is_seen else "gt"
    }
    data["history"].append(entry)

    with open(base_dir / "st_graphs" / f"object_{obj_id}.json", "w") as f:
        json.dump(data, f, indent=4)

    with open(fname, "w") as f:
        json.dump(data, f, indent=4)

def update_household_profile(date: str,
                             relocations: Dict[int, Dict[str, Any]],
                             base_dir: Path,
                             is_seen: bool):
    """
    Update household-level spatio-temporal profile.
    relocations: dict of oid -> {"from": fid, "to": fid}
    """
    root = base_dir.parent
    fname = root / "household_profile.json"
    data = _load_json(fname)
    if data is None:
        data = {
            "history": []  # list of daily relocation summaries
        }

    entry = {
        "date": date,
        "relocations": relocations,
        "source": "seen" if is_seen else "gt"
    }
    data["history"].append(entry)
    
    with open(base_dir / "household_profile.json", "w") as f:
        json.dump(data, f, indent=4)

    with open(fname, "w") as f:
        json.dump(data, f, indent=4)
