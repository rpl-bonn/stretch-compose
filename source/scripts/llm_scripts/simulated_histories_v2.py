
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Simulate future scene snapshots with semantic priors and geometric checks.
# - Reads the most recent dated folder (YYYY_MM_DD) under DATA_ROOT.
# - Uses drawers/*.json to get drawer capacities and furniture mapping.
# - Uses objects/*.json and graph.json to determine current objects and locations.
# - Adds new objects with small probability and keeps them thereafter.
# - Respects semantic priors for location selection (open places vs. specific drawer types).
# - Places centroids plausibly on open surfaces or inside drawers if they fit.
# - Adds an immovable 'table_near_door' node to the graph the first day it appears.
#
# Folder output: simulated_future/YYYY_MM_DD/{objects,locations}/, graph.json, scene.json

import os
import re
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

# =====================
# CONFIG (edit here)
# =====================
DATA_ROOT = "/home/ws/data/scene_graph"                      # root containing dated folders like 2025_08_21
OUTPUT_ROOT = "/home/ws/data/simulated_future"     # where to write simulated future days
NUM_DAYS = 5                         # number of future days to generate
START_FROM_MOST_RECENT = True        # auto-pick latest folder under DATA_ROOT

# New immovable item to add to graph once (kept across days)
ADD_TABLE_NEAR_DOOR = True
TABLE_NEAR_DOOR_NODE = {
  "id": "table_near_door",
  "type": "furniture",
  "label": "table_near_door",
  "pose_hint": {"x": 1.0, "y": 0.0, "z": 0.75}
}

# NEW OBJECTS (HARDCODED, WITH CAPS) -> [x, y, z] dims (m)
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

# Map drawer JSON field "furniture" (int) -> {"type": "cabinet"|"bookshelf"|...}
FURNITURE_MAP = {
  1: {"type": "cabinet"},
  2: {"type": "bookshelf"}
}

# Allowed top-level open locations
OPEN_LOCATIONS = ["trash_can", "cabinet", "bookshelf", "armchair", "table", "table_near_door"]

# Semantic priors: object label (lowercased) -> list of (location_tag, weight)
# location_tag in: "trash_can", "armchair", "table", "table_near_door",
# "cabinet_open", "bookshelf_open", "cabinet_drawer", "bookshelf_drawer"
SEMANTIC_PRIORS = {
  "bottle": [("cabinet_open", 3), ("table", 2), ("bookshelf_open", 1)],
  "yellow storage box": [("bookshelf_open", 3), ("cabinet_open", 2)],
  "blue plastic cup": [("cabinet_open", 3), ("table", 3), ("bookshelf_open", 1)],
  "sheep plushy": [("armchair", 3), ("table", 1), ("bookshelf_open", 1)],
  "silver watering can": [("bookshelf_open", 3), ("cabinet_open", 2)],
  "cat plushy": [("armchair", 3), ("bookshelf_open", 1), ("table", 1)],
  "cooking pot": [("cabinet_drawer", 3), ("cabinet_open", 2)],
  "saucepan": [("cabinet_drawer", 3), ("cabinet_open", 2)],
  "book": [("bookshelf_open", 3), ("bookshelf_drawer", 2)],
  "card game": [("bookshelf_drawer", 3), ("bookshelf_open", 2), ("table", 1)],
  "spoon": [("cabinet_drawer", 3)],
  "fork": [("cabinet_drawer", 3)],
  "knife": [("cabinet_drawer", 3)],
  "spatula": [("cabinet_drawer", 3)],
  "tennis ball": [("bookshelf_drawer", 2), ("armchair", 1), ("table", 1)],
  "football": [("armchair", 2), ("table", 1), ("bookshelf_drawer", 1)],
  "green watering can": [("bookshelf_open", 3), ("cabinet_open", 1)],
  "board_game": [("bookshelf_open", 3), ("bookshelf_drawer", 2), ("table", 1)],
  "binder_folder": [("bookshelf_open", 3), ("bookshelf_drawer", 1)],
  "power_drill": [("bookshelf_drawer", 3)],
  "scissors": [("cabinet_drawer", 3)],
  "wrench": [("cabinet_drawer", 3)],
  "screwdrivers": [("cabinet_drawer", 3)],
  "hammer": [("cabinet_drawer", 3)],
  "spectacle box": [("bookshelf_drawer", 3), ("armchair", 1), ("table", 1)],
  "mobile phone": [("bookshelf_drawer", 2), ("armchair", 2), ("table", 2), ("table_near_door", 2)],
  "watch": [("bookshelf_drawer", 2), ("armchair", 2), ("table", 2), ("table_near_door", 2)],
  "apartment keys": [("bookshelf_drawer", 2), ("table_near_door", 3), ("table", 2)]
}

MISPLACE_TO_ADJACENT_DRAWER_PROB = 0.25
PERSONAL_SMALL_ITEMS = {"spectacle box", "mobile phone", "watch"}

def _is_date_folder(name: str) -> bool:
  return bool(re.fullmatch(r"\d{4}_\d{2}_\d{2}", name))

def _latest_folder(root: Path) -> Path | None:
  dated = [p for p in root.iterdir() if p.is_dir() and _is_date_folder(p.name)]
  if not dated:
    return None
  dated.sort(key=lambda p: datetime.strptime(p.name, "%Y_%m_%d"))
  return dated[-1]

def _load_json(p: Path) -> dict:
  with open(p, "r") as f:
    return json.load(f)

def _write_json(p: Path, data: dict):
  p.parent.mkdir(parents=True, exist_ok=True)
  with open(p, "w") as f:
    json.dump(data, f, indent=2)

def load_drawers(drawers_dir: Path) -> list[dict]:
  drawers = []
  if not drawers_dir.exists():
    return drawers
  for fp in sorted(drawers_dir.glob("*.json")):
    try:
      d = _load_json(fp)
      d["_path"] = str(fp)
      drawers.append(d)
    except Exception:
      continue
  return drawers

def load_objects(objects_dir: Path) -> dict[int, dict]:
  objs = {}
  if not objects_dir.exists():
    return objs
  for fp in sorted(objects_dir.glob("*.json")):
    try:
      d = _load_json(fp)
      objs[int(d["id"])] = d
    except Exception:
      continue
  return objs

def load_graph(graph_fp: Path) -> dict:
  if graph_fp.exists():
    return _load_json(graph_fp)
  return {"objects": [], "nodes": []}

def dims_fit(obj_dims: list[float], box_dims: list[float]) -> bool:
  o = sorted(obj_dims)
  b = sorted(box_dims)
  return all(oi <= bi for oi, bi in zip(o, b))

def sample_centroid_open(hint=None) -> list[float]:
  if hint:
    hx, hy, hz = hint.get("x", 0.0), hint.get("y", 0.0), hint.get("z", 0.75)
  else:
    hx, hy, hz = 0.0, 0.0, 0.75
  return [
    round(random.uniform(hx - 1.0, hx + 1.0), 3),
    round(random.uniform(hy - 1.0, hy + 1.0), 3),
    round(random.uniform(hz - 0.15, hz + 0.15), 3),
  ]

def sample_centroid_in_drawer(drawer: dict, margin: float = 0.02) -> list[float]:
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

def _normalize(weights: list[float]) -> list[float]:
  s = sum(weights)
  if s <= 0:
    return [1.0 / len(weights)] * len(weights)
  return [w / s for w in weights]

def choose_semantic_location(label: str, drawers: list[dict]) -> tuple[str, dict | None]:
  key = label.lower()
  priors = SEMANTIC_PRIORS.get(key, None)
  if priors is None:
    priors = [("table", 2), ("bookshelf_open", 2), ("cabinet_open", 1), ("armchair", 1)]

  def drawers_of(kind: str) -> list[dict]:
    out = []
    for d in drawers:
      furn_id = d.get("furniture", None)
      furn_type = FURNITURE_MAP.get(furn_id, {}).get("type", None) if furn_id is not None else None
      if kind == "cabinet_drawer" and furn_type == "cabinet":
        out.append(d)
      elif kind == "bookshelf_drawer" and furn_type == "bookshelf":
        out.append(d)
    return out

  tags, weights = zip(*priors)
  probs = _normalize([float(w) for w in weights])
  choice = random.choices(tags, weights=probs, k=1)[0]

  if choice in {"trash_can", "armchair", "table", "table_near_door"}:
    return choice, None
  if choice == "cabinet_open":
    return "cabinet", None
  if choice == "bookshelf_open":
    return "bookshelf", None

  if choice in {"cabinet_drawer", "bookshelf_drawer"}:
    pool = drawers_of(choice)
    if not pool:
      return ("cabinet" if choice.startswith("cabinet") else "bookshelf"), None
    return f"{choice}", random.choice(pool)

  return "table", None

def maybe_misplace_drawer(label: str, drawer_choice: dict, same_type_drawers: list[dict]) -> dict:
  if label.lower() not in PERSONAL_SMALL_ITEMS:
    return drawer_choice
  if random.random() >= MISPLACE_TO_ADJACENT_DRAWER_PROB:
    return drawer_choice
  if not same_type_drawers or len(same_type_drawers) < 2:
    return drawer_choice
  candidates = [d for d in same_type_drawers if d.get("id") != drawer_choice.get("id")]
  if not candidates:
    return drawer_choice
  return random.choice(candidates)

def same_type_drawer_pool(drawers: list[dict], drawer_choice: dict) -> list[dict]:
  furn_id = drawer_choice.get("furniture", None)
  furn_type = FURNITURE_MAP.get(furn_id, {}).get("type", None) if furn_id is not None else None
  if furn_type is None:
    return [drawer_choice]
  pool = []
  for d in drawers:
    fid = d.get("furniture", None)
    ftype = FURNITURE_MAP.get(fid, {}).get("type", None) if fid is not None else None
    if ftype == furn_type:
      pool.append(d)
  return pool

def simulate():
  root = Path(DATA_ROOT)
  base = _latest_folder(root) if START_FROM_MOST_RECENT else _latest_folder(root)
  if base is None:
    raise FileNotFoundError("No dated folders found under DATA_ROOT")

  drawers_dir = base / "drawers"
  objects_dir = base / "objects"
  graph_fp = base / "graph.json"

  drawers = load_drawers(drawers_dir)
  objects = load_objects(objects_dir)
  graph = load_graph(graph_fp)

  placements = {}
  next_id = (max(objects.keys()) + 1) if objects else 1
  introduced = set([o.get("label", "").upper() for o in objects.values()])

  add_table_once = ADD_TABLE_NEAR_DOOR
  start_date = datetime.strptime(base.name, "%Y_%m_%d") + timedelta(days=1)
  out_root = Path(OUTPUT_ROOT)
  labels_by_id = {oid: o.get("label", str(oid)) for oid, o in objects.items()}

  for day in range(NUM_DAYS):
    date = start_date + timedelta(days=day)
    day_dir = out_root / date.strftime("%Y_%m_%d")
    (day_dir / "objects").mkdir(parents=True, exist_ok=True)
    (day_dir / "locations").mkdir(parents=True, exist_ok=True)

    for name_caps, dims in NEW_OBJECTS.items():
      if name_caps in introduced:
        continue
      if random.random() < NEW_OBJECT_DAILY_PROB:
        label = name_caps.replace("_", " ").lower()
        oid = next_id
        next_id += 1
        objects[oid] = {
          "id": oid,
          "label": label,
          "dimensions": dims,
          "centroid": [0, 0, 0]
        }
        labels_by_id[oid] = label
        introduced.add(name_caps)

    for oid, obj in objects.items():
      label = obj.get("label", "").lower()
      dims = obj.get("dimensions", obj.get("size", [0.1, 0.1, 0.1]))
      if (oid not in placements) or (random.random() < RELOCATION_DAILY_PROB):
        loc_tag, drawer_choice = choose_semantic_location(label, drawers)
        if drawer_choice is not None:
          if not dims_fit(dims, drawer_choice.get("dimensions", [0.3, 0.4, 0.1])):
            loc = "cabinet" if loc_tag.startswith("cabinet") else "bookshelf"
            placements[label] = loc
          else:
            pool = same_type_drawer_pool(drawers, drawer_choice)
            drawer_final = maybe_misplace_drawer(label, drawer_choice, pool)
            placements[label] = f"drawer_{drawer_final.get('id')}"
        else:
          if loc_tag in {"cabinet_open", "cabinet"}:
            placements[label] = "cabinet"
          elif loc_tag in {"bookshelf_open", "bookshelf"}:
            placements[label] = "bookshelf"
          elif loc_tag in OPEN_LOCATIONS:
            placements[label] = loc_tag
          else:
            placements[label] = "table"

    for oid, obj in objects.items():
      label = obj.get("label", "").lower()
      dims = obj.get("dimensions", obj.get("size", [0.1, 0.1, 0.1]))
      loc = placements.get(label, "table")
      if isinstance(loc, str) and loc.startswith("drawer_"):
        did = int(loc.split("_")[1])
        drawer = next((d for d in drawers if int(d.get("id", -1)) == did), None)
        centroid = sample_centroid_in_drawer(drawer if drawer else {})
        pose = {"position": {"x": centroid[0], "y": centroid[1], "z": centroid[2]},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}
      else:
        hint = None
        if loc == "table_near_door":
          hint = TABLE_NEAR_DOOR_NODE.get("pose_hint", None)
        centroid = sample_centroid_open(hint)
        pose = {"position": {"x": centroid[0], "y": centroid[1], "z": centroid[2]},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}

      out_obj = {
        "id": oid,
        "label": obj.get("label", ""),
        "centroid": centroid,
        "dimensions": dims,
        "pose": pose,
        "location": loc
      }
      _write_json(day_dir / "objects" / f"{oid}.json", out_obj)
      _write_json(day_dir / "locations" / f"{obj.get('label','')}.json", {"object": obj.get("label",""), "location": loc})

    g = {"date": date.strftime("%Y_%m_%d"),
         "objects": [{"id": oid, "label": objects[oid].get("label",""), "location": placements.get(objects[oid].get("label","").lower(), "table")} for oid in sorted(objects.keys())]}

    if add_table_once:
      add_table_once = False
      nodes = graph.get("nodes", [])
      nodes = nodes + [TABLE_NEAR_DOOR_NODE]
      graph["nodes"] = nodes

    _write_json(day_dir / "graph.json", g)
    _write_json(day_dir / "scene.json", {"scene": f"Simulated scene for {g['date']}"})

  print(f"Done. Wrote {NUM_DAYS} day(s) to: {OUTPUT_ROOT}")

if __name__ == "__main__":
  random.seed(7)  # deterministic runs
  simulate()
