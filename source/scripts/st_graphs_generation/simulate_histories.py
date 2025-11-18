#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulated_histories_v3.py

Extends v2 with two root outputs, each per day:
  gt/YYYY_MM_DD/{objects/, graph.json, scene.json, coverage.json}
  seen/YYYY_MM_DD/{objects/, graph.json, scene.json, coverage.json}

- gt/: full omniscient world (relocation + misplacement logic). History evolves gt -> gt.
- seen/: robot's belief graph. History evolves seen(t-1) -> seen(t) by applying only the
         subset of gt changes actually detected that day (open > drawer-opened >> drawer-closed).

- Removes 'locations/'.
- Objects JSON have: id, label, centroid, dimensions, pose (4x4), drawer, confidence
- graph.json: node_ids, node_labels, connections {obj_id -> furniture_id}
- coverage.json:
    * gt:     operation vs previous gt:       added | relocated | drawer | stationary
    * seen:   operation vs previous seen:     added | relocated | drawer | observed

Immovables (always present): cabinet, bookshelf, table_couch, table_near_door, trash_can, armchair.
We never place items in trash_can (forbidden in priors), but it stays as an immovable node.

All indentation uses 2 spaces.
"""

import os
import re
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# =====================
# CONFIG
# =====================
DATA_ROOT = "/home/ws/data/scene_graph/"                 # parent containing base YYYY_MM_DD folder(s)
OUTPUT_ROOT_GT = "/home/ws/data/gt_scene_graph/"
OUTPUT_ROOT_SEEN = "/home/ws/data/scene_graph/"
NUM_DAYS = 5
START_FROM_MOST_RECENT = True
RANDOM_SEED = 7

# New objects (HARDCODED, WITH CAPS) -> [x, y, z] dims (m)
NEW_OBJECTS = {
  "BOARD_GAME": [0.30, 0.25, 0.10],
  "BINDER_FOLDER": [0.31, 0.27, 0.05],
  "POWER_DRILL": [0.23, 0.08, 0.18],
  "APARTMENT_KEYS": [0.05, 0.03, 0.01],
  "MOBILE_PHONE": [0.16, 0.08, 0.01],
  "WATCH": [0.05, 0.05, 0.01],
  "SPECTACLE_BOX": [0.17, 0.06, 0.05]
}
NEW_OBJECT_DAILY_PROB = 0.15
RELOCATION_DAILY_PROB = 0.55

# Misplacement behavior
MISPLACE_TO_ADJACENT_DRAWER_PROB = 0.25
PERSONAL_SMALL_ITEMS = {"spectacle box", "mobile phone", "watch"}

# Detection (for seen)
OPEN_DET_PROB = 0.75
DRAWER_OPEN_DET_PROB = 0.90
DRAWER_CLOSED_DET_PROB = 0.03
FRACTION_DRAWERS_OPENED = 0.40  # fraction of all drawers opened that day (random sample)

# Immovable nodes (stable synthetic ids)
IMMOVABLE_LABELS_ORDERED = [
  "trash_can",
  "cabinet",
  "bookshelf",
  "armchair",
  "table_couch",
  "table_near_door"
]
# immovable_ids = {
#   "trash_can": 1000,
#   "cabinet": 1001,
#   "bookshelf": 1002,
#   "armchair": 1003,
#   "table_couch": 1004,
#   "table_near_door": 1005
# }

# Drawer -> furniture type lookup (from drawers/*.json 'furniture' field)
FURNITURE_MAP = {
  1: {"type": "cabinet"},
  2: {"type": "bookshelf"}
}

# Allowed open surfaces for priors (excluding trash_can to forbid placements there)
OPEN_LOCATIONS = ["cabinet", "bookshelf", "armchair", "table_couch", "table_near_door"]

# Semantic priors: object -> [(location_tag, weight)]
# location_tag in {"armchair","table_couch","table_near_door","cabinet_open","bookshelf_open",
#                  "cabinet_drawer","bookshelf_drawer"}
SEMANTIC_PRIORS = {
  "bottle": [("cabinet_open", 3), ("table_couch", 2), ("bookshelf_open", 1)],
  "yellow storage box": [("bookshelf_open", 3), ("cabinet_open", 2)],
  "blue plastic cup": [("cabinet_open", 3), ("table_couch", 3), ("bookshelf_open", 1)],
  "sheep plushy": [("armchair", 3), ("table_couch", 1), ("bookshelf_open", 1)],
  "silver watering can": [("bookshelf_open", 3), ("cabinet_open", 2)],
  "cat plushy": [("armchair", 3), ("bookshelf_open", 1), ("table_couch", 1)],
  "cooking pot": [("cabinet_drawer", 3), ("cabinet_open", 2)],
  "saucepan": [("cabinet_drawer", 3), ("cabinet_open", 2)],
  "book": [("bookshelf_open", 3), ("bookshelf_drawer", 2)],
  "card game": [("bookshelf_drawer", 3), ("bookshelf_open", 2), ("table_couch", 1)],
  "spoon": [("cabinet_drawer", 3)],
  "fork": [("cabinet_drawer", 3)],
  "knife": [("cabinet_drawer", 3)],
  "spatula": [("cabinet_drawer", 3)],
  "tennis ball": [("bookshelf_drawer", 2), ("armchair", 1), ("table_couch", 1)],
  "football": [("armchair", 2), ("table_couch", 1), ("bookshelf_drawer", 1)],
  "green watering can": [("bookshelf_open", 3), ("cabinet_open", 1)],
  "board_game": [("bookshelf_open", 3), ("bookshelf_drawer", 2), ("table_couch", 1)],
  "binder_folder": [("bookshelf_open", 3), ("bookshelf_drawer", 1)],
  "power_drill": [("bookshelf_drawer", 3)],
  "scissors": [("cabinet_drawer", 3)],
  "wrench": [("cabinet_drawer", 3)],
  "screwdrivers": [("cabinet_drawer", 3)],
  "hammer": [("cabinet_drawer", 3)],
  "spectacle box": [("bookshelf_drawer", 3), ("armchair", 1), ("table_couch", 1)],
  "mobile phone": [("bookshelf_drawer", 2), ("armchair", 2), ("table_couch", 2), ("table_near_door", 2)],
  "watch": [("bookshelf_drawer", 2), ("armchair", 2), ("table_couch", 2), ("table_near_door", 2)],
  "apartment keys": [("bookshelf_drawer", 2), ("table_near_door", 3), ("table_couch", 2)]
}

random.seed(RANDOM_SEED)

# =====================
# Basic I/O helpers
# =====================

def _is_date_folder(name: str) -> bool:
  return bool(re.fullmatch(r"\d{4}_\d{2}_\d{2}", name))

def _latest_folder(root: Path) -> Optional[Path]:
  cands = [p for p in root.iterdir() if p.is_dir() and _is_date_folder(p.name)]
  if not cands:
    return None
  cands.sort(key=lambda p: datetime.strptime(p.name, "%Y_%m_%d"))
  return cands[-1]

def _load_json(p: Path) -> dict:
  with open(p, "r") as f:
    return json.load(f)

def _maybe_load_json(p: Path) -> Optional[dict]:
  try:
    with open(p, "r") as f:
      return json.load(f)
  except Exception:
    return None

def _write_json(p: Path, data: dict):
  p.parent.mkdir(parents=True, exist_ok=True)
  with open(p, "w") as f:
    json.dump(data, f, indent=2)

# =====================
# Data loaders
# =====================
import numpy as np

def load_immovable_ids(graph_fp: Path, scene_fp: Path) -> Dict[str, int]:
    """
    Extract immovable IDs from graph.json, with centroids from scene.json.
    Distinguish tables based on distance to the armchair when possible.
    Always return both table_couch and table_near_door roles.
    """
    data = _maybe_load_json(graph_fp)
    scene = _maybe_load_json(scene_fp)
    if not data or not scene:
        return {}

    ids = data.get("node_ids", [])
    labels = data.get("node_labels", [])
    immovable = {}
    table_ids = []

    for i, lbl in zip(ids, labels):
        l = lbl.lower()
        if l in {"cabinet", "bookshelf", "armchair", "couch", "trash_can"}:
            immovable[l] = i
        elif l == "table":
            table_ids.append(i)

    # helper to get centroid
    def centroid_for(node_id: int):
        for node in scene.get("immovables", []):
            if node.get("id") == node_id:
                return node.get("centroid")
        return None

    # armchair position
    armchair_id = immovable.get("armchair") or immovable.get("couch")
    armchair_c = centroid_for(armchair_id) if armchair_id else None

    if len(table_ids) == 1:
        immovable["table_couch"] = table_ids[0]
        immovable["table_near_door"] = table_ids[0]  # alias to same table

    elif len(table_ids) >= 2:
        if armchair_c:
            import numpy as np
            dists = []
            for tid in table_ids:
                tc = centroid_for(tid)
                if tc is not None:
                    dists.append((tid, np.linalg.norm(np.array(tc) - np.array(armchair_c))))
            if dists:
                dists.sort(key=lambda x: x[1])
                immovable["table_couch"] = dists[0][0]
                immovable["table_near_door"] = dists[-1][0]
        # fallback if no centroids found
        if "table_couch" not in immovable:
            immovable["table_couch"] = table_ids[0]
        if "table_near_door" not in immovable:
            immovable["table_near_door"] = table_ids[-1]

    # final fallback if no tables at all
    if "table_couch" not in immovable:
        immovable["table_couch"] = max(ids) + 1
    if "table_near_door" not in immovable:
        immovable["table_near_door"] = immovable["table_couch"]

    print("Resolved immovables:", immovable)
    return immovable



def load_drawers(drawers_dir: Path) -> List[dict]:
  out = []
  if not drawers_dir.exists():
    return out
  for fp in sorted(drawers_dir.glob("*.json")):
    d = _maybe_load_json(fp)
    if d:
      d["_path"] = str(fp)
      out.append(d)
  return out

def load_objects(objects_dir: Path) -> Dict[int, dict]:
  out = {}
  if not objects_dir.exists():
    return out
  for fp in sorted(objects_dir.glob("*.json")):
    d = _maybe_load_json(fp)
    if d and "id" in d:
      out[int(d["id"])] = d
  return out

def load_graph(graph_fp: Path) -> dict:
  return _maybe_load_json(graph_fp) or {"node_ids": [], "node_labels": [], "connections": {}}

# =====================
# Geometry / sampling
# =====================
def dims_fit(obj_dims: List[float], box_dims: List[float]) -> bool:
  o = sorted(obj_dims)
  b = sorted(box_dims)
  return all(oi <= bi for oi, bi in zip(o, b))

def sample_centroid_open(hint=None) -> List[float]:
  if hint:
    hx, hy, hz = hint.get("x", 0.0), hint.get("y", 0.0), hint.get("z", 0.75)
  else:
    hx, hy, hz = 0.0, 0.0, 0.75
  return [
    round(random.uniform(hx - 1.0, hx + 1.0), 3),
    round(random.uniform(hy - 1.0, hy + 1.0), 3),
    round(random.uniform(hz - 0.15, hz + 0.15), 3),
  ]

def sample_centroid_in_drawer(drawer: dict, margin: float = 0.02) -> List[float]:
  cx, cy, cz = drawer.get("centroid", [0, 0, 0])
  dx, dy, dz = drawer.get("dimensions", [0.3, 0.4, 0.1])
  dx_i = max(dx - 2 * margin, 0.01)
  dy_i = max(dy - 2 * margin, 0.01)
  dz_i = max(dz - 2 * margin, 0.01)
  return [
    round(random.uniform(cx - dx_i / 2, cx + dx_i / 2), 3),
    round(random.uniform(cy - dy_i / 2, cy + dy_i / 2), 3),
    round(random.uniform(cz - dz_i / 2, cz + dz_i / 2), 3),
  ]

def pose_from_centroid(c: List[float]) -> List[List[float]]:
  x, y, z = c
  return [
    [1.0, 0.0, 0.0, x],
    [0.0, 1.0, 0.0, y],
    [0.0, 0.0, 1.0, z],
    [0.0, 0.0, 0.0, 1.0],
  ]

# =====================
# Semantics / priors
# =====================
def _normalize_ws(ws: List[float]) -> List[float]:
  s = float(sum(ws))
  if s <= 0:
    return [1.0 / len(ws)] * len(ws)
  return [w / s for w in ws]

def choose_semantic_location(label: str, drawers: List[dict]) -> Tuple[str, Optional[dict]]:
  key = label.lower()
  priors = SEMANTIC_PRIORS.get(
    key,
    [("table_couch", 2), ("bookshelf_open", 2), ("cabinet_open", 1), ("armchair", 1)]
  )

  def drawers_of(kind: str) -> List[dict]:
    out = []
    for d in drawers:
      furn_id = d.get("furniture", None)
      furn_type = FURNITURE_MAP.get(furn_id, {}).get("type", None) if furn_id is not None else None
      if kind == "cabinet_drawer" and furn_type == "cabinet":
        out.append(d)
      elif kind == "bookshelf_drawer" and furn_type == "bookshelf":
        out.append(d)
    return out

  tags, ws = zip(*priors)
  tag = random.choices(tags, weights=_normalize_ws(list(map(float, ws))), k=1)[0]

  if tag in {"armchair", "table_couch", "table_near_door"}:
    return tag, None
  if tag == "cabinet_open":
    return "cabinet", None
  if tag == "bookshelf_open":
    return "bookshelf", None
  if tag in {"cabinet_drawer", "bookshelf_drawer"}:
    pool = drawers_of(tag)
    if not pool:
      return ("cabinet" if tag.startswith("cabinet") else "bookshelf"), None
    return tag, random.choice(pool)

  return "table_couch", None

def same_type_drawer_pool(drawers: List[dict], drawer_choice: dict) -> List[dict]:
  fid = drawer_choice.get("furniture", None)
  ftype = FURNITURE_MAP.get(fid, {}).get("type", None) if fid is not None else None
  if ftype is None:
    return [drawer_choice]
  return [d for d in drawers if FURNITURE_MAP.get(d.get("furniture", None), {}).get("type", None) == ftype]

def maybe_misplace_drawer(label: str, drawer_choice: dict, pool_same_type: List[dict]) -> dict:
  if label.lower() not in PERSONAL_SMALL_ITEMS:
    return drawer_choice
  if random.random() >= MISPLACE_TO_ADJACENT_DRAWER_PROB:
    return drawer_choice
  cands = [d for d in pool_same_type if d.get("id") != drawer_choice.get("id")]
  return random.choice(cands) if cands else drawer_choice

# =====================
# Graph helpers
# =====================
def ensure_immovables(node_ids: List[int], node_labels: List[str], immovable_ids: Dict[str, int]) -> None:
  """
  Ensure all immovable IDs from immovable_ids are present in node_ids/node_labels.
  """
  for lab, iid in immovable_ids.items():
    if iid not in node_ids:
      node_ids.append(iid)
      #node_labels.append("table" if lab.startswith("table") else lab)
      print(f"  [info] added missing immovable {lab} ({iid}) to graph.")

def furniture_id_for_open(open_class: str, immovable_ids: Dict[str, int]) -> int:
  """
  Map an open_class string to a furniture ID from immovable_ids.
  Fallback = table_couch if nothing else fits.
  """
  return immovable_ids.get(open_class, immovable_ids.get("table_couch"))
# =====================
# Detection for 'seen'
# =====================
def pick_opened_drawers(drawers: List[dict]) -> set:
  ids = [int(d.get("id")) for d in drawers if "id" in d]
  if not ids:
    return set()
  n = max(0, int(len(ids) * FRACTION_DRAWERS_OPENED))
  random.shuffle(ids)
  return set(ids[:n])

def detect(furn_id: int, drawer_id: int, opened_drawers: set) -> Tuple[bool, float]:
  if drawer_id != -1:
    p = DRAWER_OPEN_DET_PROB if drawer_id in opened_drawers else DRAWER_CLOSED_DET_PROB
  else:
    p = OPEN_DET_PROB
  seen = (random.random() < p)
  conf = round(random.uniform(0.82, 0.98), 6) if seen else 0.0
  return seen, conf

# =====================
# Main simulator
# =====================
def simulate():
  # Base day
  root = Path(DATA_ROOT)
  base = _latest_folder(root) if START_FROM_MOST_RECENT else _latest_folder(root)
  
  if base is None:
    raise FileNotFoundError("No dated folders found under DATA_ROOT")
  
  base_scene = _maybe_load_json(base / "scene.json") or {}

  base_graph_fp = base / "graph.json"
  base_scene_fp = base / "scene.json"
  immovable_ids = load_immovable_ids(base_graph_fp, base_scene_fp)
  # Inputs from base
  drawers = load_drawers(base / "drawers")
  objects = load_objects(base / "objects")
  base_graph = load_graph(base / "graph.json")

  

  
  # Seed objects and placements from base folder
  placements = {}
  for oid, obj in objects.items():
      fid = base_graph["connections"].get(str(oid))
      did = int(obj.get("drawer", -1))
      if fid is None:
          # fallback: assign to cabinet if unknown
          fid = immovable_ids.get("cabinet", -1)
      placements[oid] = (fid, did)

  # Graph bootstrap (shared immovables)
  node_ids = list(base_graph.get("node_ids", []))
  node_labels = list(base_graph.get("node_labels", []))
  ensure_immovables(node_ids, node_labels, immovable_ids)

  
  # State for evolving gt and seen
  next_id = (max(objects.keys()) + 1) if objects else 1
  introduced_caps = set([str(o.get("label", "")).upper() for o in objects.values()])

  # Seed seen(0) as the base graph state (robot believes the last full graph)
  seen_placements = dict(placements)  # deep-enough for tuples
  seen_objects = dict(objects)         # shallow—fields will be overwritten day-by-day

  start_date = datetime.strptime(base.name, "%Y_%m_%d") + timedelta(days=1)

  for day_idx in range(NUM_DAYS):
    date = start_date + timedelta(days=day_idx)
    # Paths
    gt_day = Path(OUTPUT_ROOT_GT) / date.strftime("%Y_%m_%d")
    seen_day = Path(OUTPUT_ROOT_SEEN) / date.strftime("%Y_%m_%d")
    (gt_day / "objects").mkdir(parents=True, exist_ok=True)
    (seen_day / "objects").mkdir(parents=True, exist_ok=True)

    # 1) ---- Evolve GT (world truth): new + relocations with misplacement ----
    prev_gt_conn = {str(oid): placements[oid][0] for oid in placements}
    prev_gt_draw = {str(oid): placements[oid][1] for oid in placements}

    # possibly introduce new objects
    for NAME_CAPS, dims in NEW_OBJECTS.items():
      if NAME_CAPS in introduced_caps:
        continue
      if random.random() < NEW_OBJECT_DAILY_PROB:
        label = NAME_CAPS.replace("_", " ").lower()
        oid = next_id; next_id += 1
        # sample semantic location
        tag, drawer_choice = choose_semantic_location(label, drawers)
        if drawer_choice and dims_fit(dims, drawer_choice.get("dimensions", [0.3, 0.4, 0.1])):
          pool = same_type_drawer_pool(drawers, drawer_choice)
          drawer_final = maybe_misplace_drawer(label, drawer_choice, pool)
          did = int(drawer_final["id"])
          ftype = FURNITURE_MAP.get(drawer_final.get("furniture"), {}).get("type")
          fid = immovable_ids["bookshelf"] if ftype == "bookshelf" else immovable_ids["cabinet"]
          placements[oid] = (fid, did)
        else:
          # open placement
          if tag in {"cabinet_open", "cabinet"}:
            fid = immovable_ids["cabinet"]
          elif tag in {"bookshelf_open", "bookshelf"}:
            fid = immovable_ids["bookshelf"]
          elif tag in OPEN_LOCATIONS:
            fid = immovable_ids[tag]
          else:
            fid = immovable_ids["table_couch"]
          placements[oid] = (fid, -1)
        # register object entity
        objects[oid] = {"id": oid, "label": label, "dimensions": dims}
        introduced_caps.add(NAME_CAPS)
        if oid not in node_ids:
          node_ids.append(oid); node_labels.append(label)

    # relocations for all existing objects (with priors)
    for oid, obj in list(objects.items()):
      dims = obj.get("dimensions", [0.1, 0.1, 0.1])
      lab = str(obj.get("label", "")).lower()
      if (oid not in placements) or (random.random() < RELOCATION_DAILY_PROB):
        tag, drawer_choice = choose_semantic_location(lab, drawers)
        if drawer_choice and dims_fit(dims, drawer_choice.get("dimensions", [0.3, 0.4, 0.1])):
          pool = same_type_drawer_pool(drawers, drawer_choice)
          drawer_final = maybe_misplace_drawer(lab, drawer_choice, pool)
          did = int(drawer_final["id"])
          ftype = FURNITURE_MAP.get(drawer_final.get("furniture"), {}).get("type")
          fid = immovable_ids["bookshelf"] if ftype == "bookshelf" else immovable_ids["cabinet"]
          placements[oid] = (fid, did)
        else:
          if tag in {"cabinet_open", "cabinet"}:
            placements[oid] = (immovable_ids["cabinet"], -1)
          elif tag in {"bookshelf_open", "bookshelf"}:
            placements[oid] = (immovable_ids["bookshelf"], -1)
          elif tag in OPEN_LOCATIONS:
            placements[oid] = (immovable_ids[tag], -1)
          else:
            placements[oid] = (immovable_ids["table_couch"], -1)

    # 2) ---- Write GT day (objects, graph, scene, coverage) ----
    gt_connections = {}
    gt_coverage = []
    for oid, obj in objects.items():
      fid, did = placements[oid]
      # sample centroid consistent with placement
      if did != -1:
        dr = next((d for d in drawers if int(d.get("id", -1)) == did), None)
        centroid = sample_centroid_in_drawer(dr if dr else {})
      else:
        hint = {"x": 1.0, "y": 0.0, "z": 0.75} if fid == immovable_ids["table_near_door"] else None
        centroid = sample_centroid_open(hint)
      pose = pose_from_centroid(centroid)
      # confidence in gt is not a detector score; set to 1.0 (omniscient)
      obj_out = {
        "id": oid,
        "label": str(obj.get("label", "")).lower(),
        "centroid": centroid,
        "dimensions": obj.get("dimensions", [0.1, 0.1, 0.1]),
        "pose": pose,
        "drawer": int(did),
        "confidence": 1.0
      }
      _write_json(gt_day / "objects" / f"{oid}.json", obj_out)
      gt_connections[str(oid)] = int(fid)

      # coverage op vs previous gt
      prev_fid = prev_gt_conn.get(str(oid))
      prev_did = prev_gt_draw.get(str(oid))
      if prev_fid is None:
        op = "added"
      elif prev_fid != fid or prev_did != did:
        op = "relocated" if did == -1 else "drawer"
      else:
        op = "stationary"
      gt_coverage.append({
        "label": obj_out["label"],
        "id": oid,
        "operation": op,
        "location": (f"drawer_{did}" if did != -1 else f"furniture_{fid}"),
        "furniture_id": int(fid),
        "drawer_id": (int(did) if did != -1 else -1),
        "centroid": centroid,
        "confidence": 1.0
      })

    gt_graph = {
      "node_ids": node_ids,
      "node_labels": node_labels,
      "connections": gt_connections
    }
    _write_json(gt_day / "graph.json", gt_graph)
    _write_json(gt_day / "scene.json", base_scene)
    _write_json(gt_day / "coverage.json", {"date": gt_day.name, "observations": gt_coverage})

    # 3) ---- Evolve SEEN: start from yesterday's seen; apply only detected GT changes ----
    # opened drawers today:
    opened = pick_opened_drawers(drawers)
    # map for comparing vs prev seen
    prev_seen_conns = {str(oid): seen_placements[oid][0] for oid in seen_placements}
    prev_seen_draws = {str(oid): seen_placements[oid][1] for oid in seen_placements}

    # Determine GT deltas (changes) vs previous GT
    changed_or_new = set()
    for oid in objects.keys():
      prev_fid = prev_gt_conn.get(str(oid))
      prev_did = prev_gt_draw.get(str(oid))
      fid, did = placements[oid]
      if prev_fid is None or prev_fid != fid or prev_did != did:
        changed_or_new.add(oid)

    # Apply only detected changes to seen_placements; others keep old seen state
    seen_coverage = []
    for oid in objects.keys():
      fid_gt, did_gt = placements[oid]
      # if new object in gt and never existed in seen, it can enter seen only if detected
      if oid not in seen_placements:
        seen_flag, conf = detect(fid_gt, did_gt, opened)
        if seen_flag:
          # Add to seen with GT placement
          seen_placements[oid] = (fid_gt, did_gt)
          seen_objects[oid] = {"id": oid, "label": str(objects[oid]["label"]).lower(),
                               "dimensions": objects[oid].get("dimensions", [0.1, 0.1, 0.1])}
          op = "added"
          seen_coverage.append({
            "label": seen_objects[oid]["label"], "id": oid, "operation": op,
            "location": (f"drawer_{did_gt}" if did_gt != -1 else f"furniture_{fid_gt}"),
            "furniture_id": int(fid_gt),
            "drawer_id": int(did_gt),
            "centroid": None,  # will be filled when writing object
            "confidence": conf
          })
        # if not detected, remain absent from seen
        continue

      # existing in seen: decide if today's GT change (if any) is detected
      fid_prev_seen, did_prev_seen = seen_placements[oid]
      if oid in changed_or_new:
        seen_flag, conf = detect(fid_gt, did_gt, opened)
        if seen_flag:
          # update seen to gt placement
          seen_placements[oid] = (fid_gt, did_gt)
          op = "relocated" if did_gt == -1 else "drawer"
        else:
          # not detected -> keep old seen placement
          op = "observed"  # we still might observe it at its old location; will set conf below
          fid_gt, did_gt = fid_prev_seen, did_prev_seen  # for writing object below
          conf = 0.0
      else:
        # no gt change -> if we detect it, mark observed; else leave as-is (but still persist node)
        seen_flag, conf = detect(fid_prev_seen, did_prev_seen, opened)
        op = "observed" if seen_flag else "observed"  # classification label kept simple

      # record coverage; centroid set later when object file is written
      seen_coverage.append({
        "label": str(objects[oid]["label"]).lower(),
        "id": oid,
        "operation": op,
        "location": (f"drawer_{did_gt}" if did_gt != -1 else f"furniture_{fid_gt}"),
        "furniture_id": int(fid_gt),
        "drawer_id": int(did_gt),
        "centroid": None,
        "confidence": conf
      })

    # 4) ---- Write SEEN day (objects, graph, scene, coverage) ----
    seen_connections = {}
    # We write ALL previously-known seen objects (persist belief), but their placement is whatever seen_placements has now.
    for oid, (fid, did) in seen_placements.items():
      # centroid consistent with seen placement
      if did != -1:
        dr = next((d for d in drawers if int(d.get("id", -1)) == did), None)
        centroid = sample_centroid_in_drawer(dr if dr else {})
      else:
        hint = {"x": 1.0, "y": 0.0, "z": 0.75} if fid == immovable_ids["table_near_door"] else None
        centroid = sample_centroid_open(hint)
      pose = pose_from_centroid(centroid)
      # confidence in seen object json: keep last nonzero detection confidence if any, else 0.0
      # Find the coverage entry for this oid today to set confidence; else keep prior
      todays = next((o for o in reversed(seen_coverage) if o["id"] == oid), None)
      det_conf = todays["confidence"] if todays else 0.0
      prior_fp = seen_day.parent / (seen_day.name if False else "")  # no prior file to read; keep 0 if not detected
      obj_dims = seen_objects[oid].get("dimensions", [0.1, 0.1, 0.1])

      obj_out = {
        "id": oid,
        "label": str(seen_objects[oid]["label"]).lower(),
        "centroid": centroid,
        "dimensions": obj_dims,
        "pose": pose,
        "drawer": int(did),
        "confidence": float(det_conf)
      }
      _write_json(seen_day / "objects" / f"{oid}.json", obj_out)
      seen_connections[str(oid)] = int(fid)

      # back-fill centroid in coverage entries for this oid
      for rec in seen_coverage:
        if rec["id"] == oid and rec["centroid"] is None:
          rec["centroid"] = centroid

    seen_graph = {
      "node_ids": node_ids,
      "node_labels": node_labels,
      "connections": seen_connections
    }
    _write_json(seen_day / "graph.json", seen_graph)
    _write_json(seen_day / "scene.json", base_scene)
    _write_json(seen_day / "coverage.json", {
      "date": seen_day.name,
      "observations": seen_coverage,
      "opened_drawers": sorted(list(pick_opened_drawers(drawers))),  # logged for ablations; not the same as used above
      "params": {
        "open_det_prob": OPEN_DET_PROB,
        "drawer_open_det_prob": DRAWER_OPEN_DET_PROB,
        "drawer_closed_det_prob": DRAWER_CLOSED_DET_PROB,
        "fraction_drawers_opened": FRACTION_DRAWERS_OPENED
      }
    })

  print(f"Done. Wrote {NUM_DAYS} day(s) under {OUTPUT_ROOT_GT}/ and {OUTPUT_ROOT_SEEN}/")

# =====================
# Entry
# =====================
if __name__ == "__main__":
  random.seed(RANDOM_SEED)
  simulate()
