import json
import os
import argparse
from typing import Dict, Any, List, Optional


def infer_relation(edge: Dict[str, Any], parent_node: Dict[str, Any]) -> str:
  """
  Map HOMER relation + parent properties to our canonical relation:
    - ON              -> ON_SURFACE
    - INSIDE + CAN_OPEN in parent -> INSIDE_DRAWER
    - INSIDE + else               -> IN_OPEN_STORAGE
  Fallback: ON_SURFACE
  """
  rtype = edge.get("relation_type", "")
  props = parent_node.get("properties", []) or []

  if rtype == "ON":
    return "ON_SURFACE"
  if rtype == "INSIDE":
    if "CAN_OPEN" in props:
      return "INSIDE_DRAWER"
    else:
      return "IN_OPEN_STORAGE"

  # default
  return "ON_SURFACE"


def build_scene_hier_for_snapshot(graph: Dict[str, Any],
                                  time_seconds: float) -> Dict[str, Any]:
  """
  Build a scene_hier snapshot from a single HOMER graph entry.
  Output format:

  {
    "time_seconds": <float>,
    "rooms": [
      {
        "room_name": <str>,
        "room_id": <int>,
        "furniture": {
          "<fid>": {
            "label": <str>,
            "centroid": null,
            "dimensions": null,
            "objects": [
              {
                "id": <int>,
                "label": <str>,
                "relation": "ON_SURFACE"|"IN_OPEN_STORAGE"|"INSIDE_DRAWER",
                "drawer": -1
              },
              ...
            ]
          },
          ...
        }
      },
      ...
    ],
    "unassigned_furniture": {
      "<fid>": { ... same furniture block ... },
      ...
    }
  }
  """
  nodes = {n["id"]: n for n in graph.get("nodes", [])}

  # --- classify nodes ---
  room_ids = {nid for nid, n in nodes.items() if n.get("category") == "Rooms"}

  furniture_ids = set()
  object_ids = set()

  for nid, n in nodes.items():
    cat = n.get("category")
    props = n.get("properties", []) or []

    if cat == "Furniture":
      furniture_ids.add(nid)
    elif cat == "Appliances" and ("CONTAINERS" in props or "SURFACES" in props):
      # treat container-like appliances (e.g. fridge) as furniture
      furniture_ids.add(nid)
    elif cat == "placable_objects":
      object_ids.add(nid)
    # extend here if you want certain Props/Decor as objects

  edges = graph.get("edges", [])

  # --- furniture -> room (INSIDE) ---
  furn_room = {}
  for e in edges:
    if (e.get("relation_type") == "INSIDE"
        and e.get("from_id") in furniture_ids
        and e.get("to_id") in room_ids):
      fid = e["from_id"]
      rid = e["to_id"]
      furn_room.setdefault(fid, set()).add(rid)

  # --- object -> furniture / room edges ---
  obj_furn_edges = {}
  obj_room_edges = {}

  for e in edges:
    from_id = e.get("from_id")
    to_id = e.get("to_id")
    if from_id in object_ids:
      if to_id in furniture_ids:
        obj_furn_edges.setdefault(from_id, []).append(e)
      elif to_id in room_ids:
        obj_room_edges.setdefault(from_id, []).append(e)

  # --- helper: build one furniture block, including attached objects ---
  def build_furniture_block(fid: int) -> Dict[str, Any]:
    fn = nodes[fid]
    furn = {
      "label": fn.get("class_name"),
      "centroid": None,      # HOMER has no geometry here
      "dimensions": None,
      "objects": []
    }

    # attach objects that point to this furniture
    for oid in object_ids:
      for e in obj_furn_edges.get(oid, []):
        if e.get("to_id") == fid:
          on = nodes[oid]
          rel = infer_relation(e, fn)
          furn["objects"].append({
            "id": oid,
            "label": on.get("class_name"),
            "relation": rel,
            "drawer": -1  # HOMER has no explicit drawers; stays -1
          })
          break

    return furn

  # --- room -> furniture assignment ---
  room_to_furn = {rid: [] for rid in room_ids}
  for fid in furniture_ids:
    rids = sorted(furn_room.get(fid, []))
    if rids:
      # choose first room if multiple; can refine later
      room_to_furn[rids[0]].append(fid)

  # --- build rooms array ---
  rooms_out: List[Dict[str, Any]] = []
  for rid in sorted(room_ids):
    room_node = nodes[rid]
    furn_dict: Dict[str, Any] = {}

    for fid in room_to_furn.get(rid, []):
      furn_dict[str(fid)] = build_furniture_block(fid)

    rooms_out.append({
      "room_name": room_node.get("class_name"),
      "room_id": rid,
      "furniture": furn_dict
    })

  # --- unassigned furniture (no room edge) ---
  assigned_furn = {fid for flist in room_to_furn.values() for fid in flist}
  unassigned: Dict[str, Any] = {}
  for fid in sorted(furniture_ids - assigned_furn):
    unassigned[str(fid)] = build_furniture_block(fid)

  return {
    "time_seconds": float(time_seconds),
    "rooms": rooms_out,
    "unassigned_furniture": unassigned
  }


def time_to_hhmm(time_seconds: float) -> str:
  """
  Map HOMER 'time' (seconds from 00:00) to HHMM string:
    e.g. 360 s -> 6 min -> 00:06 -> '0006'
  """
  total_minutes = int(time_seconds // 60)
  hours = total_minutes // 60
  minutes = total_minutes % 60
  return f"{hours:02d}{minutes:02d}"


def derive_day_index_from_filename(path: str) -> int:
  """
  Try to derive a day index from the filename.
  Example:
    '000.json'  -> 0
    '001.json'  -> 1
    'day2.json' -> 2
  If no leading digits are found, default to 0.
  """
  base = os.path.basename(path)
  name, _ = os.path.splitext(base)
  digits = ""
  for ch in name:
    if ch.isdigit():
      digits += ch
    else:
      break
  if digits:
    return int(digits)
  return 0


def convert_single_homer_file(
    homer_json_path: str,
    out_root: str,
    day_index: Optional[int] = None
) -> None:
  """
  Convert a single HOMER JSON (one day) into:
    out_root/day_<day_index>/HHMM_scene_hier.json
  """
  with open(homer_json_path, "r") as f:
    data = json.load(f)

  times = data.get("times", [])
  graphs = data.get("graphs", [])

  if len(times) != len(graphs):
    raise ValueError(
      f"Mismatch: {len(times)} times vs {len(graphs)} graphs in {homer_json_path}"
    )

  if day_index is None:
    day_index = derive_day_index_from_filename(homer_json_path)

  day_folder = os.path.join(out_root, f"day_{day_index}")
  os.makedirs(day_folder, exist_ok=True)

  for t, g in zip(times, graphs):
    hhmm = time_to_hhmm(t)
    scene_hier = build_scene_hier_for_snapshot(g, t)
    out_path = os.path.join(day_folder, f"{hhmm}_scene_hier.json")
    with open(out_path, "w") as f_out:
      json.dump(scene_hier, f_out, indent=2)


def convert_homer_path(
    input_path: str,
    out_root: str
) -> None:
  """
  input_path:
    - if file: treat as one day (derive day index from name)
    - if directory: treat each *.json file inside as one day,
      assign day index from filename digits.
  Output:
    out_root/day_<i>/HHMM_scene_hier.json
  """
  if os.path.isfile(input_path):
    os.makedirs(out_root, exist_ok=True)
    convert_single_homer_file(input_path, out_root)
    return

  if os.path.isdir(input_path):
    os.makedirs(out_root, exist_ok=True)
    files = [
      os.path.join(input_path, f)
      for f in os.listdir(input_path)
      if f.endswith(".json")
    ]
    files.sort()
    for path in files:
      day_idx = derive_day_index_from_filename(path)
      convert_single_homer_file(path, out_root, day_index=day_idx)
    return

  raise FileNotFoundError(f"Input path not found: {input_path}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Convert HOMER graphs (000.json etc.) into day_*/HHMM_scene_hier.json"
  )
  parser.add_argument(
    "input_path",
    help="Path to HOMER JSON file for a single day OR a directory with multiple day JSONs"
  )
  parser.add_argument(
    "--out_root",
    default="out_homer",
    help="Output root directory (default: out_homer)"
  )
  args = parser.parse_args()

  convert_homer_path(args.input_path, args.out_root)
  print(f"Conversion completed. Output at: {args.out_root}")