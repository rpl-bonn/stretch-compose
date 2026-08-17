#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kallax search GUI server.

This version keeps searchnet_execution.py unchanged as the main execution pipeline.
The GUI starts one object-search request; the server launches SearchNet, watches
its stdout, and updates /progress so the HTML shelf can show:
  idle / searching / not_found / found / coffee_table

It also:
  - keeps English object names for the model/search pipeline
  - uses German object names for GUI messages
  - uses existing German audio keys through AudioFeedbackInterface.send(...)
  - exposes the newest saved detection/SAM3 image as /latest_detection_image
"""

import glob
import json
import logging
import os
import re
import shlex
import subprocess
import threading
import time
from typing import Dict, Optional

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# Initialize ROS before using the audio interface
import rclpy
if not rclpy.ok():
    rclpy.init(args=None)

from utils.audio_utils import AudioFeedbackInterface

audio_interface = AudioFeedbackInterface()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = BASE_DIR
ROS_SETUP = "/opt/ros/humble/setup.bash"

if not os.path.isdir(SCRIPTS_DIR):
    raise RuntimeError(f"SCRIPTS_DIR not found: {SCRIPTS_DIR}")

# Folders where SAM3 / detection images may be saved.
DETECTION_IMAGE_DIRS = [
    "/home/ws/data/images/sam3/sam3"
]
try:
    from utils.recursive_config import Config
    _cfg = Config()
    DETECTION_IMAGE_DIRS.insert(0, _cfg.get_subpath("images"))
except Exception:
    pass

# English input -> German text for GUI labels/messages.
OBJECT_DE: Dict[str, str] = {
    "yellow ball": "Gelber Ball",
    "red ball": "Roter Ball",
    "tennis ball": "Tennisball",
    "bottle": "Flasche",
    "water bottle": "Wasserflasche",
    "cup": "Tasse",
    "mug": "Becher",
    "book": "Buch",
    "keys": "Schlüssel",
    "key": "Schlüssel",
    "phone": "Handy",
    "mobile phone": "Handy",
    "wallet": "Geldbörse",
    "glasses": "Brille",
    "pen": "Stift",
    "pencil": "Bleistift",
    "scissors": "Schere",
    "tape": "Klebeband",
    "charger": "Ladegerät",
    "mouse": "Maus",
    "notebook": "Notizbuch",
    "spoon": "Löffel",
    "fork": "Gabel",
    "knife": "Messer",
    "plate": "Teller",
    "toy": "Spielzeug",
    "plushie": "Plüschtier",
    "watering can": "Gießkanne",
    "plant": "Pflanze",
    "picture frame": "Bilderrahmen",
    "frame": "Bilderrahmen",
    "remote": "Fernbedienung",
    "milk carton": "Milchkarton",
    "pringles": "Pringles",
}

# Quick-pick ids from the HTML -> English object names for SearchNet.
OBJECT_MAP = {
    "watering_can": "watering can",
    "cup": "cup",
    "water_bottle": "water bottle",
    "milk_carton": "milk carton",
    "pringles": "pringles",
    "tennis_ball": "tennis ball",
    "book": "book",
    "keys": "keys",
    "remote": "remote",
    "phone": "phone",
    "wallet": "wallet",
}

# English object name -> German audio key.
# IMPORTANT: edit the values on the right to match the exact keys your colleague used.
OBJECT_AUDIO_KEY_DE: Dict[str, str] = {
    "yellow ball": "gelber_ball",
    "red ball": "roter_ball",
    "tennis ball": "tennisball",
    "bottle": "flasche",
    "water bottle": "wasserflasche",
    "cup": "tasse",
    "mug": "becher",
    "book": "buch",
    "keys": "schluessel",
    "key": "schluessel",
    "phone": "handy",
    "mobile phone": "handy",
    "wallet": "geldboerse",
    "glasses": "brille",
    "pen": "stift",
    "pencil": "bleistift",
    "scissors": "schere",
    "tape": "klebeband",
    "charger": "ladegeraet",
    "mouse": "maus",
    "notebook": "notizbuch",
    "spoon": "loeffel",
    "fork": "gabel",
    "knife": "messer",
    "plate": "teller",
    "toy": "spielzeug",
    "plushie": "plueschtier",
    "watering can": "giesskanne",
    "plant": "pflanze",
    "picture frame": "bilderrahmen",
    "frame": "bilderrahmen",
    "remote": "fernbedienung",
    "milk carton": "milchkarton",
    "pringles": "pringles",
}

# Generic German audio event keys.
# IMPORTANT: edit these values to match the exact audio keys in your repo.
AUDIO_KEYS = {
    "greeting": "greeting",
    "searching": "searching",        # e.g., "Ich suche nach ..."
    "found": "found",                # e.g., "Ich habe es gefunden"
    "not_found": "not_found",        # e.g., "Dort habe ich es nicht gefunden"
    "concealed": "concealed",        # e.g., "Ich suche jetzt in geschlossenen Fächern"
    "coffee_table": "coffee_table",  # e.g., "Ich suche jetzt am Couchtisch"
    "done": "done",
    "error": "error",
}


def normalize_object_key(obj: str) -> str:
    return " ".join(str(obj).strip().lower().replace("_", " ").split())


def german_object_name(obj: str) -> str:
    key = normalize_object_key(obj)
    return OBJECT_DE.get(key, obj)


def format_name(raw: str) -> str:
    return " ".join(w.capitalize() for w in raw.replace("_", " ").split())


DRAWER_MAP = {
    "13": "left_white_drawer_upper",
    "15": "left_white_drawer_lower",
    "19": "right_black_drawer_upper",
    "14": "right_black_drawer_lower",
}

# Edit this mapping if your scene-graph drawer IDs differ.
GRAPH_ID_TO_GUI = {
    "13": "white_drawer_upper",
    "15": "white_drawer_lower",
    "19": "black_drawer_upper",
    "14": "black_drawer_lower",
    "white_door": "white_door",
    "black_drawers": "black_drawers",
    "black_drawer_upper": "black_drawer_upper",
    "black_drawer_lower": "black_drawer_lower",
    "white_drawers": "white_drawers",
    "white_drawer_upper": "white_drawer_upper",
    "white_drawer_lower": "white_drawer_lower",
    "top_left_open": "top_left_open",
    "top_right_open": "top_right_open",
    "middle_right_open": "middle_right_open",
    "bottom_left_open": "bottom_left_open",
    "bottom_right_open": "bottom_right_open",
    "coffee_table": "coffee_table",
}

FURNITURE_NAME_TO_GUI = {
    "coffee": "coffee_table",
    "table": "coffee_table",
    "white door": "white_door",
    "door": "white_door",
    "black drawer": "black_drawers",
    "black drawers": "black_drawers",
    "white drawer": "white_drawers",
    "white drawers": "white_drawers",
    "shelf": "top_left_open",
    "bookshelf": "top_left_open",
}

SEARCH_SEQUENCE = [
    "top_left_open",
    "top_right_open",
    "middle_right_open",
    "bottom_left_open",
    "bottom_right_open",
    "white_door",
    "black_drawer_upper",
    "black_drawer_lower",
    "white_drawer_upper",
    "white_drawer_lower",
    "coffee_table",
]

logging.getLogger("werkzeug").setLevel(logging.ERROR)
app = Flask(__name__)
CORS(app)

_run_lock = threading.Lock()
_is_running = False

GUI_STATE = {
    "state": "idle",
    "object": "",
    "object_de": "",
    "compartment": "",
    "message": "Idle",
    "image_version": 0,
}

_STATE_LOCK = threading.Lock()


def set_gui_state(**kwargs):
    with _STATE_LOCK:
        GUI_STATE.update(kwargs)
        GUI_STATE["timestamp"] = time.time()
    print(f"[GUI_STATE] {GUI_STATE}", flush=True)


def get_gui_state():
    with _STATE_LOCK:
        return dict(GUI_STATE)


def speak_audio_key(key: str):
    """Play one predefined audio key using the existing AudioFeedbackInterface."""
    if not key:
        return
    try:
        audio_interface.send(key)
        print(f"[AUDIO] played key: {key}", flush=True)
    except Exception as e:
        print(f"[AUDIO] failed key={key}: {e}", flush=True)


def speak_object_key(obj: str):
    """Play the German object-name audio key, if configured."""
    obj_key = normalize_object_key(obj)
    audio_key = OBJECT_AUDIO_KEY_DE.get(obj_key)
    if audio_key:
        speak_audio_key(audio_key)
    else:
        print(f"[AUDIO] no object audio key configured for: {obj_key}", flush=True)


def speak_searching_object(obj: str):
    speak_audio_key(AUDIO_KEYS.get("searching", ""))
    speak_object_key(obj)


def speak_found_object(obj: str):
    speak_audio_key(AUDIO_KEYS.get("found", ""))
    speak_object_key(obj)


def speak_not_found_here(obj: str):
    speak_audio_key(AUDIO_KEYS.get("not_found", ""))
    speak_object_key(obj)


def speak_concealed_search(obj: str):
    speak_audio_key(AUDIO_KEYS.get("concealed", ""))
    speak_object_key(obj)


def speak_coffee_table(obj: str):
    speak_audio_key(AUDIO_KEYS.get("coffee_table", ""))
    speak_object_key(obj)


def launch_script(script: str, args=None, capture_output: bool = False) -> subprocess.Popen:
    args = args or []
    argv = " ".join(shlex.quote(str(a)) for a in args)
    py_flag = "-u" if capture_output else ""
    cmd = (
        f'bash -lc "'
        f'source {shlex.quote(ROS_SETUP)} && '
        f'cd {shlex.quote(SCRIPTS_DIR)} && '
        f'python3 {py_flag} {shlex.quote(script)} {argv}'
        f'"'
    )
    print(f"[GUI] Launch: {cmd}", flush=True)
    return subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.STDOUT if capture_output else None,
        text=True if capture_output else False,
        bufsize=1 if capture_output else -1,
    )


def launch_searchnet_object(obj: str, obj_de: str) -> subprocess.Popen:
    """Call searchnet_execution.execute_search(obj, obj_de) without modifying searchnet_execution.py."""
    obj_literal = json.dumps(obj)
    obj_de_literal = json.dumps(obj_de)
    code = f"import demo_searchnet_openvocab as s; s.execute_search({obj_literal}, {obj_de_literal})"
    bash_cmd = (
        f'source {shlex.quote(ROS_SETUP)} && '
        f'cd {shlex.quote(SCRIPTS_DIR)} && '
        f'python3 -u -c {shlex.quote(code)}'
    )
    print(f"[GUI] Launch SearchNet: bash -lc {shlex.quote(bash_cmd)}", flush=True)
    return subprocess.Popen(
        ["bash", "-lc", bash_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


def _try_start_action():
    if not _run_lock.acquire(blocking=False):
        return False, (jsonify({"status": "busy"}), 409)
    global _is_running
    if _is_running:
        _run_lock.release()
        return False, (jsonify({"status": "busy"}), 409)
    _is_running = True
    return True, None


def _release_action_lock():
    global _is_running
    _is_running = False
    try:
        _run_lock.release()
    except Exception:
        pass


def _watch_proc(proc: subprocess.Popen):
    proc.wait()
    _release_action_lock()
    print("[GUI] Action finished.", flush=True)


def _normalize_compartment(raw: Optional[str]) -> str:
    if not raw:
        return ""
    raw = str(raw).strip()
    if raw in GRAPH_ID_TO_GUI:
        return GRAPH_ID_TO_GUI[raw]
    low = raw.lower()
    for key, gui_id in FURNITURE_NAME_TO_GUI.items():
        if key in low:
            return gui_id
    return raw


def _next_sequence_compartment() -> str:
    current = get_gui_state().get("compartment", "")
    if current not in SEARCH_SEQUENCE:
        return SEARCH_SEQUENCE[0]
    idx = SEARCH_SEQUENCE.index(current)
    return SEARCH_SEQUENCE[min(idx + 1, len(SEARCH_SEQUENCE) - 1)]


def _parse_searchnet_line(line: str, obj: str, obj_de: str):
    text = line.strip()
    if not text:
        return
    lower = text.lower()

    # Example: Searching for bottle in/on kitchen shelf (...)
    m = re.search(r"Searching for .*? in/on (.*?)\s*(?:\(| at |$)", text, re.IGNORECASE)
    if m:
        furniture = m.group(1).strip()
        comp = _normalize_compartment(furniture)
        set_gui_state(
            state="searching",
            object=obj,
            object_de=obj_de,
            compartment=comp,
            message=f"Ich suche nach {obj_de}: {furniture}",
        )
        return

    # Example: bottle is in the scene graph. Searching for it in/on table at ...
    m = re.search(r"Searching for it in/on (.*?) at", text, re.IGNORECASE)
    if m:
        furniture = m.group(1).strip()
        comp = _normalize_compartment(furniture)
        set_gui_state(
            state="searching",
            object=obj,
            object_de=obj_de,
            compartment=comp,
            message=f"Ich suche nach {obj_de}: {furniture}",
        )
        return

    if "searching in concealed spaces" in lower:
        comp = _next_sequence_compartment()
        set_gui_state(
            state="searching",
            object=obj,
            object_de=obj_de,
            compartment=comp,
            message=f"{obj_de} wurde offen nicht gefunden. Ich suche in geschlossenen Fächern.",
        )
        speak_concealed_search(obj)
        return

    # Example: Found bottle in door 13 in bookshelf.
    m = re.search(r"Found .*? in door (\d+)", text, re.IGNORECASE)
    if m:
        comp = _normalize_compartment(m.group(1))
        set_gui_state(
            state="found",
            object=obj,
            object_de=obj_de,
            compartment=comp,
            message=f"Ich habe {obj_de} gefunden.",
            image_version=get_gui_state().get("image_version", 0) + 1,
        )
        speak_found_object(obj)
        return

    if lower.startswith("found ") or f"found {obj.lower()}" in lower:
        comp = get_gui_state().get("compartment", "") or _next_sequence_compartment()
        set_gui_state(
            state="found",
            object=obj,
            object_de=obj_de,
            compartment=comp,
            message=f"Ich habe {obj_de} gefunden.",
            image_version=get_gui_state().get("image_version", 0) + 1,
        )
        speak_found_object(obj)
        return

    m = re.search(r"Did not find .*? in/on (.*?)\.", text, re.IGNORECASE)
    if m:
        furniture = m.group(1).strip()
        comp = _normalize_compartment(furniture)
        set_gui_state(
            state="not_found",
            object=obj,
            object_de=obj_de,
            compartment=comp,
            message=f"{obj_de} wurde dort nicht gefunden.",
        )
        # Usually avoid speaking this every time if it becomes too repetitive.
        # Uncomment if you want audio for every failed compartment:
        # speak_not_found_here(obj)
        return

    if "not found" in lower and "coffee" in lower:
        set_gui_state(
            state="coffee_table",
            object=obj,
            object_de=obj_de,
            compartment="coffee_table",
            message=f"{obj_de} wurde im Regal nicht gefunden. Ich suche am Couchtisch.",
        )
        speak_coffee_table(obj)
        return


def _watch_searchnet_proc(proc: subprocess.Popen, obj: str, obj_de: str):
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(f"[SEARCHNET] {line}", end="", flush=True)
            _parse_searchnet_line(line, obj, obj_de)
        return_code = proc.wait()
        state = get_gui_state().get("state")
        if return_code == 0:
            if state not in ("found", "coffee_table"):
                set_gui_state(
                    state="done",
                    object=obj,
                    object_de=obj_de,
                    message=f"Suche nach {obj_de} beendet.",
                )
                speak_audio_key(AUDIO_KEYS.get("done", ""))
        else:
            set_gui_state(
                state="error",
                object=obj,
                object_de=obj_de,
                message=f"Fehler bei der Suche nach {obj_de}.",
            )
            speak_audio_key(AUDIO_KEYS.get("error", ""))
    finally:
        _release_action_lock()
        print("[GUI] SearchNet action finished.", flush=True)


def find_latest_detection_image() -> Optional[str]:
    patterns = []
    for root in DETECTION_IMAGE_DIRS:
        if not root:
            continue
        patterns.extend([
            os.path.join(root, "*.png"),
            os.path.join(root, "*.jpg"),
            os.path.join(root, "*.jpeg"),
            os.path.join(root, "**", "*.png"),
            os.path.join(root, "**", "*.jpg"),
            os.path.join(root, "**", "*.jpeg"),
        ])
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern, recursive=True))

    def _is_file(p):
        return os.path.isfile(p)

    # Tier 1: sam3 output images (path or filename contains "sam3" AND starts with "sam3_")
    tier1 = [
        p for p in candidates
        if _is_file(p) and os.path.basename(p).lower().startswith("sam3_")
    ]
    if tier1:
        return max(tier1, key=os.path.getmtime)

    # Tier 2: detection/mask/gripper images — explicitly exclude raw camera frames
    tier2 = [
        p for p in candidates
        if _is_file(p)
        and any(k in os.path.basename(p).lower()
                for k in ("sam", "detect", "detection", "mask", "gripper"))
        and not any(k in os.path.basename(p).lower()
                    for k in ("camera_image", "rgb", "frame", "raw"))
    ]
    if tier2:
        return max(tier2, key=os.path.getmtime)

    # Tier 3: fallback — any candidate file
    usable = [p for p in candidates if _is_file(p)]
    if not usable:
        return None
    return max(usable, key=os.path.getmtime)


@app.route("/search", methods=["POST"])
def run_search():
    payload = request.get_json(silent=True) or {}
    raw_name = str(payload.get("object", "")).strip()
    is_raw = bool(payload.get("raw", False))
    if not raw_name:
        return jsonify({"status": "error", "message": "No object name"}), 400

    obj = raw_name if is_raw else OBJECT_MAP.get(raw_name, raw_name)
    obj = obj.replace("_", " ").strip()
    obj_de = german_object_name(obj)

    ok, busy = _try_start_action()
    if not ok:
        return busy

    set_gui_state(
        state="started",
        object=obj,
        object_de=obj_de,
        compartment="",
        message=f"Ich suche nach {obj_de}.",
    )
    speak_searching_object(obj)

    try:
        proc = launch_searchnet_object(obj, obj_de)
        threading.Thread(target=_watch_searchnet_proc, args=(proc, obj, obj_de), daemon=True).start()
        return jsonify({"status": "ok", "object": obj, "object_de": obj_de})
    except Exception as e:
        _release_action_lock()
        set_gui_state(state="error", object=obj, object_de=obj_de, message=str(e))
        speak_audio_key(AUDIO_KEYS.get("error", ""))
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/progress", methods=["GET"])
def progress():
    return jsonify(get_gui_state())


@app.route("/internal_update", methods=["POST"])
def internal_update():
    payload = request.get_json(silent=True) or {}
    set_gui_state(**payload)
    return jsonify({"status": "ok", "state": get_gui_state()})


@app.route("/latest_detection_image", methods=["GET"])
def latest_detection_image():
    latest = find_latest_detection_image()
    if latest is None:
        return jsonify({"status": "error", "message": "No detection image found"}), 404
    return send_file(latest)


@app.route("/", methods=["GET"])
def index():
    return send_file(os.path.join(BASE_DIR, "kallax-ui_v4.html"))


@app.route("/kallax.png", methods=["GET"])
def shelf_image():
    return send_file(os.path.join(BASE_DIR, "kallax.png"))


@app.route("/status", methods=["GET"])
def status():
    return jsonify({"running": _is_running})


@app.route("/audio", methods=["POST"])
def play_audio():
    try:
        speak_audio_key(AUDIO_KEYS.get("greeting", "greeting"))
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# Kept for backward compatibility with the old drawer page.
@app.route("/run", methods=["POST"])
def run_drawer():
    payload = request.get_json(silent=True) or {}
    drawer_id = str(payload.get("id", "")).strip()
    if drawer_id not in DRAWER_MAP:
        return jsonify({"status": "error", "message": f"Unknown drawer id: {drawer_id}"}), 400
    ok, busy = _try_start_action()
    if not ok:
        return busy
    human = format_name(DRAWER_MAP[drawer_id])
    try:
        proc = launch_script("demo_open_drawer.py", [drawer_id])
        threading.Thread(target=_watch_proc, args=(proc,), daemon=True).start()
        return jsonify({"status": "ok", "drawer_id": drawer_id, "formatted": human})
    except Exception as e:
        _release_action_lock()
        return jsonify({"status": "error", "message": str(e)}), 500


# Kept for backward compatibility with the old grasp page.
@app.route("/grasp", methods=["POST"])
def run_grasp():
    payload = request.get_json(silent=True) or {}
    raw_name = str(payload.get("object", "")).strip()
    is_raw = bool(payload.get("raw", False))
    if not raw_name:
        return jsonify({"status": "error", "message": "No object name"}), 400
    mapped = raw_name if is_raw else OBJECT_MAP.get(raw_name, raw_name)
    ok, busy = _try_start_action()
    if not ok:
        return busy
    try:
        proc = launch_script("demo_grasping.py", ["--object", mapped])
        threading.Thread(target=_watch_proc, args=(proc,), daemon=True).start()
        return jsonify({"status": "ok", "object": mapped})
    except Exception as e:
        _release_action_lock()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/release", methods=["POST"])
def release_gripper():
    ok, busy = _try_start_action()
    if not ok:
        return busy
    try:
        proc = launch_script("open_gripper.py")
        threading.Thread(target=_watch_proc, args=(proc,), daemon=True).start()
        return jsonify({"status": "ok"})
    except Exception as e:
        _release_action_lock()
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" Stretch Search GUI Server ".center(70))
    print(f" Scripts dir: {SCRIPTS_DIR}".center(70))
    print(" http://127.0.0.1:5007 ".center(70))
    print("=" * 70 + "\n", flush=True)
    app.run(host="127.0.0.1", port=5007, debug=False, use_reloader=False)
