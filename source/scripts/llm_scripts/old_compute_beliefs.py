
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_beliefs.py

Build compact per-object histories and compute POMDP-like beliefs over location classes,
rolling forward day by day with optional coverage logs.

Inputs (directory with dated folders YYYY_MM_DD):
  - locations/*.json (preferred) OR graph.json -> provides object->location per day
  - drawers/*.json (optional) -> maps drawer_<id> to cabinet_drawer or bookshelf_drawer
  - coverage.json (optional, per day) -> detection coverage used in observation update

Outputs:
  - compact_memory/<object>.json  (same format as build_compact_memory.py)
  - beliefs/<latest_date>/<object>.json with fields:
      {
        "object": "...",
        "date": "YYYY_MM_DD",
        "belief": {loc_class: prob, ...},
        "topk": [{"loc_class": "...","p": ...}, ...],
        "last_seen": {...},
        "note": "..."
      }

Configuration is at the top. Indentation uses 2 spaces.
"""

import re
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math

# -------------------------
# Config
# -------------------------
DATA_ROOT = Path(".")                  # parent with YYYY_MM_DD day folders
MEMORY_DIR = Path("compact_memory")    # where to write compact bundles
BELIEF_DIR = Path("beliefs")           # where to write beliefs for the latest day

HISTORY_DAYS = None                    # None -> use all, or set int window (e.g., 120)
DECAY = 0.92                           # per-day decay for frequencies and transitions
LAST_N_EVENTS = 5
TOPK_FURNITURE = 5

# POMDP knobs
HAZARD = 0.01                          # daily disappearance to Ø class
BIRTH_RATE = 0.005                     # reintroduction per day from Ø to prior pi
DEFAULT_PRIOR = {"table": 1.0}         # used only if object has no history

# Coverage defaults if coverage.json is missing
DEFAULT_DET_OPEN = 0.0                 # detection on open surfaces when not logged
DEFAULT_DET_DRAWER_OPEN = 0.0          # detection for opened drawers when not logged
DEFAULT_DET_DRAWER_CLOSED = 0.0        # detection for closed drawers when not logged

# If coverage.json exists, we expect a schema like:
# {
#   "classes": {"table": 0.8, "armchair": 0.6, "cabinet": 0.7, "bookshelf": 0.7},
#   "drawers_opened": [17, 19],     # list of opened drawer IDs that day
#   "drawer_open_prob": 0.9,
#   "drawer_closed_prob": 0.05
# }
#
# Only classes included will be used; others assume 0 unless DEFAULT_* above overrides.

# Furniture id -> type mapping for drawers (from drawers/<id>.json "furniture")
FURNITURE_MAP = {
  1: {"type": "cabinet"},
  2: {"type": "bookshelf"}
}

OPEN_CLASSES = {"trash_can", "armchair", "table", "table_near_door", "cabinet", "bookshelf"}
DRAWER_CLASSES = {"cabinet_drawer", "bookshelf_drawer", "drawer", "unknown_drawer"}
NULL_CLASS = "Ø"  # outside modeled world

# ---------------- Household Beliefs ----------------
def compute_household_belief(graph_fp: Path, scene_fp: Path, profile_fp: Path):
    """
    Compute household-level belief distribution and entropy.
    """
    with open(graph_fp) as f:
        graph = json.load(f)
    with open(scene_fp) as f:
        scene = json.load(f)
    with open(profile_fp) as f:
        profile = json.load(f)

    reloc_counts = Counter()
    for day in profile.get("history", []):
        for _, move in day.get("relocations", {}).items():
            if "to" in move:
                reloc_counts[move["to"]] += 1

    total = sum(reloc_counts.values())
    if total == 0:
        return {"belief": {}, "entropy": 0.0, "type": "unknown", "room_entropies": {}}

    dist = {fid: c / total for fid, c in reloc_counts.items()}
    entropy = -sum(p * math.log(p + 1e-9) for p in dist.values())

    if entropy < 0.5:
        htype = "tidy/unimodal"
    elif entropy < 1.5:
        htype = "habitual/bimodal"
    else:
        htype = "messy/multimodal"

    rooms = defaultdict(list)
    for furn in scene.get("furniture", []):
        rid = "living_room" if furn["centroid"][0] < 0 else "bedroom"
        rooms[rid].append(furn["id"])

    room_entropies = {}
    for rname, rnodes in rooms.items():
        mass = [dist.get(fid, 0) for fid in rnodes]
        Z = sum(mass)
        if Z > 0:
            norm = [m / Z for m in mass]
            rent = -sum(p * math.log(p + 1e-9) for p in norm)
        else:
            rent = 0
        room_entropies[rname] = rent

    return {
        "belief": dist,
        "entropy": entropy,
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

    counts = Counter()
    history = st_graph.get("history", [])
    for entry in history:
        fid = entry["furniture_id"]
        delta = max(0, day_index - history.index(entry))
        weight = (DECAY ** delta) * entry.get("confidence", 1.0)
        counts[fid] += weight

    total = sum(counts.values())
    posterior = {fid: c / total for fid, c in counts.items()} if total > 0 else {}

    obj_entropy = -sum(p * math.log(p + 1e-9) for p in posterior.values())

    recency_penalty = 1.0
    for entry in coverage:
        if entry["id"] == oid:
            days_since = entry.get("last_seen_since_days", 0)
            recency_penalty = DECAY ** days_since
            break

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

# -------------------------
# Utilities
# -------------------------
def _is_date_folder(name: str) -> bool:
  return bool(re.fullmatch(r"\d{4}_\d{2}_\d{2}", name))

def _parse_date(name: str) -> datetime:
  return datetime.strptime(name, "%Y_%m_%d")

def _sorted_date_folders(root: Path) -> List[Path]:
  cand = [p for p in root.iterdir() if p.is_dir() and _is_date_folder(p.name)]
  cand.sort(key=lambda p: _parse_date(p.name))
  return cand

def _apply_history_window(folders: List[Path]) -> List[Path]:
  if HISTORY_DAYS is None or HISTORY_DAYS <= 0 or not folders:
    return folders
  last_date = _parse_date(folders[-1].name)
  keep = []
  for d in folders:
    delta = (last_date - _parse_date(d.name)).days
    if 0 <= delta <= HISTORY_DAYS:
      keep.append(d)
  return keep

def _load_json(p: Path) -> Optional[dict]:
  try:
    with open(p, "r") as f:
      return json.load(f)
  except Exception:
    return None

def _load_drawers_meta(day_dir: Path) -> Dict[int, dict]:
  out = {}
  ddir = day_dir / "drawers"
  if not ddir.exists():
    return out
  for fp in ddir.glob("*.json"):
    d = _load_json(fp)
    if not d:
      continue
    try:
      did = int(d.get("id") if "id" in d else fp.stem)
      out[did] = d
    except Exception:
      continue
  return out

def _classify_location(loc: str, drawers_meta: Dict[int, dict]) -> Tuple[str, Optional[int], Optional[int]]:
  loc = (loc or "").strip().lower()
  if not loc:
    return "unknown", None, None
  if loc in OPEN_CLASSES:
    return loc, None, None
  if loc.startswith("drawer_"):
    try:
      did = int(loc.split("_", 1)[1])
    except Exception:
      return "unknown_drawer", None, None
    meta = drawers_meta.get(did, {})
    furn_id = meta.get("furniture", None)
    furn_type = FURNITURE_MAP.get(furn_id, {}).get("type") if furn_id is not None else None
    if furn_type == "cabinet":
      return "cabinet_drawer", furn_id, did
    if furn_type == "bookshelf":
      return "bookshelf_drawer", furn_id, did
    return "drawer", None, did
  return loc, None, None

def _collect_day_locations(day_dir: Path) -> Dict[str, str]:
  out = {}
  loc_dir = day_dir / "locations"
  if loc_dir.exists():
    for fp in loc_dir.glob("*.json"):
      d = _load_json(fp); 
      if not d: 
        continue
      label = str(d.get("object", "")).strip().lower()
      loc = str(d.get("location", "")).strip().lower()
      if label:
        out[label] = loc
    if out:
      return out
  g = _load_json(day_dir / "graph.json")
  if g and isinstance(g.get("objects"), list):
    for o in g["objects"]:
      label = str(o.get("label", "")).strip().lower()
      loc = str(o.get("location", "")).strip().lower()
      if label:
        out[label] = loc
  return out

def _decay_weight(days: int) -> float:
  days = max(0, days)
  return (DECAY ** days)

def _normalize_map(m: Dict[str, float]) -> Dict[str, float]:
  s = float(sum(m.values()))
  return {k: (v / s if s > 0 else 0.0) for k, v in m.items()}

def _normalize_vec(vec: List[float]) -> List[float]:
  s = float(sum(vec))
  return [(v / s if s > 0 else 0.0) for v in vec]

# -------------------------
# History + memory
# -------------------------
def build_compact_memory(data_root: Path) -> Tuple[Dict[str, List[dict]], Dict[str, dict]]:
  """
  Returns:
    timelines: label -> list of events with (date, loc_class, furniture_id, drawer_id, raw location)
    bundles: label -> compact summary used by LLM or priors
  """
  folders = _apply_history_window(_sorted_date_folders(data_root))
  if not folders:
    raise FileNotFoundError(f"No dated folders under {data_root}")
  last_date = _parse_date(folders[-1].name)

  timelines: Dict[str, List[dict]] = defaultdict(list)
  # collect events day by day
  for day_dir in folders:
    date_str = day_dir.name
    drawers_meta = _load_drawers_meta(day_dir)
    loc_map = _collect_day_locations(day_dir)
    for label, raw_loc in loc_map.items():
      loc_class, furn_id, drawer_id = _classify_location(raw_loc, drawers_meta)
      timelines[label].append({
        "date": date_str,
        "location": raw_loc,
        "loc_class": loc_class,
        "furniture_id": furn_id,
        "drawer_id": drawer_id
      })
  # sort by date
  for label in timelines:
    timelines[label].sort(key=lambda e: _parse_date(e["date"]))

  # build bundles
  bundles: Dict[str, dict] = {}
  for label, events in timelines.items():
    if not events:
      continue
    # last events
    last_events = list(reversed(events))[:LAST_N_EVENTS]

    # decayed frequencies
    loc_counts: Dict[str, float] = defaultdict(float)
    furn_counts: Dict[str, float] = defaultdict(float)
    for e in events:
      days = (_parse_date(folders[-1].name) - _parse_date(e["date"])).days
      w = _decay_weight(days)
      loc_counts[e["loc_class"]] += w
      if e["furniture_id"] is not None:
        furn_counts[str(e["furniture_id"])] += w

    freq_loc_class = _normalize_map(loc_counts)
    furn_sorted = sorted(furn_counts.items(), key=lambda kv: kv[1], reverse=True)
    top_furn = furn_sorted[:TOPK_FURNITURE]
    denom = sum(v for _, v in top_furn)
    freq_furniture = [{"furniture_id": int(fid), "p": (w / denom if denom > 0 else 0.0)} for fid, w in top_furn]

    # Markov transitions
    trans_acc: Dict[Tuple[str, str], float] = defaultdict(float)
    for i in range(len(events) - 1):
      a = events[i]["loc_class"]
      b = events[i + 1]["loc_class"]
      days = (_parse_date(folders[-1].name) - _parse_date(events[i]["date"])).days
      w = _decay_weight(days)
      trans_acc[(a, b)] += w

    states_sorted = [k for k, _ in sorted(freq_loc_class.items(), key=lambda kv: kv[1], reverse=True) if k]
    idx = {s: i for i, s in enumerate(states_sorted)}
    mat = [[0.0 for _ in states_sorted] for _ in states_sorted]
    row = [0.0 for _ in states_sorted]
    for (a, b), w in trans_acc.items():
      if a in idx and b in idx:
        i, j = idx[a], idx[b]
        mat[i][j] += w
        row[i] += w
    for i in range(len(states_sorted)):
      if row[i] > 0:
        mat[i] = [v / row[i] for v in mat[i]]

    bundles[label] = {
      "object": label,
      "dates": {"start": events[0]["date"], "end": events[-1]["date"]},
      "last_seen": last_events[0] if last_events else None,
      "last_events": last_events,
      "freq_loc_class": freq_loc_class,
      "freq_furniture": freq_furniture,
      "markov": {"states": states_sorted, "matrix": mat}
    }
  return timelines, bundles

# -------------------------
# Coverage handling
# -------------------------
def load_coverage(day_dir: Path) -> dict:
  cov = {
    "classes": {},
    "drawers_opened": [],
    "drawer_open_prob": DEFAULT_DET_DRAWER_OPEN,
    "drawer_closed_prob": DEFAULT_DET_DRAWER_CLOSED
  }
  fp = day_dir / "coverage.json"
  d = _load_json(fp)
  if not d:
    # fall back to defaults only
    return cov
  cov["classes"] = d.get("classes", {})
  cov["drawers_opened"] = d.get("drawers_opened", [])
  cov["drawer_open_prob"] = float(d.get("drawer_open_prob", DEFAULT_DET_DRAWER_OPEN))
  cov["drawer_closed_prob"] = float(d.get("drawer_closed_prob", DEFAULT_DET_DRAWER_CLOSED))
  return cov

def detection_prob(loc_class: str, cov: dict, drawer_id: Optional[int]) -> float:
  if loc_class in OPEN_CLASSES:
    return float(cov["classes"].get(loc_class, DEFAULT_DET_OPEN))
  if loc_class in DRAWER_CLASSES:
    if drawer_id is not None and drawer_id in cov.get("drawers_opened", []):
      return float(cov.get("drawer_open_prob", DEFAULT_DET_DRAWER_OPEN))
    else:
      return float(cov.get("drawer_closed_prob", DEFAULT_DET_DRAWER_CLOSED))
  return 0.0

# -------------------------
# POMDP belief update
# -------------------------
def belief_update_over_days(bundles: Dict[str, dict], timelines: Dict[str, List[dict]], data_root: Path) -> None:
  # Prepare days
  days = _apply_history_window(_sorted_date_folders(data_root))
  if not days:
    return
  latest = days[-1]
  # For output dirs
  MEMORY_DIR.mkdir(parents=True, exist_ok=True)
  out_day_dir = BELIEF_DIR / latest.name
  out_day_dir.mkdir(parents=True, exist_ok=True)

  # Persist compact bundles
  idx = []
  for label, b in bundles.items():
    out_fp = MEMORY_DIR / f"{label.replace(' ', '_')}.json"
    with open(out_fp, "w") as f:
      json.dump(b, f, indent=2)
    idx.append({"object": label, "path": str(out_fp)})
  with open(MEMORY_DIR / "_index.json", "w") as f:
    json.dump({"generated_from": str(data_root), "objects": idx}, f, indent=2)

  # For each object compute belief rolled in time
  for label, b in bundles.items():
    events = timelines[label]
    # state space for this object
    states = b["markov"]["states"]
    if not states:
      # fallback prior
      prior = _normalize_map(DEFAULT_PRIOR)
      belief = prior.copy()
      note = "No history; using DEFAULT_PRIOR."
    else:
      # initialize belief at day 0
      # Use last seen class one-hot, else use decayed freq as prior
      last_seen = b.get("last_seen")
      if last_seen and last_seen.get("loc_class") in states:
        belief = [1.0 if s == last_seen["loc_class"] else 0.0 for s in states]
        note = "Initialized from last_seen class."
      else:
        # use normalized freq over states
        freq = [float(b["freq_loc_class"].get(s, 0.0)) for s in states]
        belief = _normalize_vec(freq)
        note = "Initialized from decayed frequency prior."

      # Roll through each day in window
      # Transition matrix
      T = b["markov"]["matrix"]
      # Extend with Ø (null) state
      states_aug = states + [NULL_CLASS]
      # For birth, define a semantic prior pi over states as normalized freq
      pi = _normalize_vec([float(b["freq_loc_class"].get(s, 0.0)) for s in states])
      b_null = 0.0

      for day in days:
        # Prediction
        # b_pred = T^T * b
        b_pred = [0.0 for _ in states]
        for i in range(len(states)):
          for j in range(len(states)):
            b_pred[j] += T[i][j] * belief[i]

        # Hazard to null
        b_pred = [(1.0 - HAZARD) * v for v in b_pred]
        b_null = (1.0 - HAZARD) * b_null + HAZARD

        # Observation update from coverage
        cov = load_coverage(day)
        # Compute (1 - d) for each state; need drawer id, but we do not know specific drawer per class.
        # Use an expectation: if last event that day had a drawer id for this object, use it; else assume closed drawer prob.
        # For open classes: use class coverage from cov.
        one_minus_d = []
        # Try to get the last event for this day for drawer id context
        drawer_id_today = None
        for e in events:
          if e["date"] == day.name and e.get("drawer_id") is not None:
            drawer_id_today = e["drawer_id"]
        for s in states:
          if s in OPEN_CLASSES:
            dprob = detection_prob(s, cov, None)
          elif s in DRAWER_CLASSES:
            dprob = detection_prob(s, cov, drawer_id_today)
          else:
            dprob = 0.0
          one_minus_d.append(max(0.0, 1.0 - dprob))

        # Apply observation likelihood for "no detection"
        b_upd = [b_pred[i] * one_minus_d[i] for i in range(len(states))]
        # Birth from null to states by prior pi
        if b_null > 0 and any(pi):
          add = BIRTH_RATE * b_null
          b_upd = [b_upd[i] + add * pi[i] for i in range(len(states))]
          b_null = (1.0 - BIRTH_RATE) * b_null

        # Normalize including null
        ssum = sum(b_upd) + b_null
        if ssum > 0:
          b_upd = [v / ssum for v in b_upd]
          b_null = b_null / ssum
        belief = b_upd

      # Convert to map
      belief = {states[i]: float(belief[i]) for i in range(len(states))}
      # add Ø
      belief[NULL_CLASS] = float(b_null)

    # Top-k
    topk = sorted([(k, v) for k, v in belief.items() if k != NULL_CLASS], key=lambda kv: kv[1], reverse=True)[:5]
    topk = [{"loc_class": k, "p": v} for k, v in topk]

    out = {
      "object": label,
      "date": latest.name,
      "belief": belief,
      "topk": topk,
      "last_seen": b.get("last_seen"),
      "note": note
    }
    with open((out_day_dir / f"{label.replace(' ', '_')}.json"), "w") as f:
      json.dump(out, f, indent=2)

  # Write an index file
  index = [{"object": k, "path": str((out_day_dir / f"{k.replace(' ', '_')}.json"))} for k in bundles.keys()]
  with open(out_day_dir / "_index.json", "w") as f:
    json.dump({"date": latest.name, "objects": index}, f, indent=2)

  print(f"Wrote beliefs for {len(bundles)} objects to {out_day_dir}")

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
  timelines, bundles = build_compact_memory(DATA_ROOT)
  belief_update_over_days(bundles, timelines, DATA_ROOT)
