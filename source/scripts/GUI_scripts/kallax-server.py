#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, shlex, logging, threading, subprocess, time
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize ROS before using the audio interface
import rclpy
if not rclpy.ok():
    rclpy.init(args=None)

# Your audio interface (already in your repo)
from utils.audio_utils import AudioFeedbackInterface
audio_interface = AudioFeedbackInterface()

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))   # this file's folder
SCRIPTS_DIR = BASE_DIR                                      # demos live here
ROS_SETUP   = "/opt/ros/humble/setup.bash"                  # adjust if needed

if not os.path.isdir(SCRIPTS_DIR):
    raise RuntimeError(f"SCRIPTS_DIR not found: {SCRIPTS_DIR}")

# ── Pretty names / maps ───────────────────────────────────────────────────────
DRAWER_MAP = {
    "13": "left_white_drawer_upper",
    "15": "left_white_drawer_lower",
    "19": "right_black_drawer_upper",
    "14": "right_black_drawer_lower",
}
OBJECT_MAP = {
    "watering_can": "watering can",
    "cup": "cup",
    "water_bottle": "water bottle",   # use the spaced name most pipelines expect
    "milk_carton": "milk carton",
    "pringles": "pringles",
    "tennis_ball": "tennis ball",
}
def format_name(raw: str) -> str:
    return " ".join(w.capitalize() for w in raw.replace("_", " ").split())

# ── App / state ───────────────────────────────────────────────────────────────
logging.getLogger("werkzeug").setLevel(logging.ERROR)
app = Flask(__name__)
CORS(app)

_run_lock   = threading.Lock()
_is_running = False

# ── Optional: background greeter (disabled by default) ────────────────────────
def idle_greeter():
    while True:
        time.sleep(180)
        # If you want periodic greeting while idle, uncomment:
        # if not _is_running:
        #     try: audio_interface.send("greeting")
        #     except Exception: pass
threading.Thread(target=idle_greeter, daemon=True).start()

# ── Helpers ───────────────────────────────────────────────────────────────────
def launch_script(script: str, args=None) -> subprocess.Popen:
    """Launch a Python script in SCRIPTS_DIR with optional args (list[str])."""
    args = args or []
    argv = " ".join(shlex.quote(a) for a in args)
    cmd = (
        f'bash -lc "'
        f'source {shlex.quote(ROS_SETUP)} && '
        f'cd {shlex.quote(SCRIPTS_DIR)} && '
        f'python3 {shlex.quote(script)} {argv}'
        f'"'
    )
    print(f"[GUI] Launch: {cmd}", flush=True)
    return subprocess.Popen(cmd, shell=True)

def _watch_proc(proc: subprocess.Popen):
    global _is_running
    proc.wait()
    _is_running = False
    try: _run_lock.release()
    except Exception: pass
    print("[GUI] Action finished.", flush=True)

def _try_start_action():
    """Acquire the lock + set running; return (ok: bool, busy_response)."""
    if not _run_lock.acquire(blocking=False):
        return False, (jsonify({"status": "busy"}), 409)
    global _is_running
    if _is_running:
        _run_lock.release()
        return False, (jsonify({"status": "busy"}), 409)
    _is_running = True
    return True, None

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/run", methods=["POST"])
def run_drawer():
    """Open a drawer by numeric ID. Body: {'id': '20'}"""
    payload = request.get_json(silent=True) or {}
    drawer_id = str(payload.get("id", "")).strip()
    print(f"[GUI] /run payload: {payload}", flush=True)

    if drawer_id not in DRAWER_MAP:
        return jsonify({"status": "error", "message": f"Unknown drawer id: {drawer_id}"}), 400

    ok, busy = _try_start_action()
    if not ok: return busy

    human = format_name(DRAWER_MAP[drawer_id])
    print(f"[GUI] Drawer clicked: id={drawer_id} ({human})", flush=True)
    try:
        proc = launch_script("demo_open_drawer_new.py", ["--drawer_id", drawer_id])
        threading.Thread(target=_watch_proc, args=(proc,), daemon=True).start()
        return jsonify({"status": "ok", "drawer_id": drawer_id, "formatted": human})
    except Exception as e:
        global _is_running
        _is_running = False
        _run_lock.release()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/grasp", methods=["POST"])
def run_grasp():
    """
    Grasp an object.
    Body (quick-pick): {"object":"water_bottle"}
    Body (free-text) : {"object":"ketchup bottle","raw":true}
    """
    payload = request.get_json(silent=True) or {}
    raw_name = str(payload.get("object", "")).strip()
    is_raw   = bool(payload.get("raw", False))
    print(f"[GUI] /grasp payload: {payload}", flush=True)

    if not raw_name:
        return jsonify({"status": "error", "message": "No object name"}), 400

    mapped = raw_name if is_raw else OBJECT_MAP.get(raw_name, raw_name)

    ok, busy = _try_start_action()
    if not ok: return busy

    print(f"[GUI] Grasp request: {'RAW' if is_raw else 'mapped'}='{mapped}'", flush=True)
    try:
        proc = launch_script("demo_grasping.py", ["--object", mapped])
        threading.Thread(target=_watch_proc, args=(proc,), daemon=True).start()
        return jsonify({"status": "ok", "object": mapped})
    except Exception as e:
        global _is_running
        _is_running = False
        _run_lock.release()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/release", methods=["POST"])
def release_gripper():
    """
    Open the gripper using open_gripper.py (no args).
    This uses the run lock (movement).
    """
    ok, busy = _try_start_action()
    if not ok: return busy

    print("[GUI] Release gripper requested", flush=True)
    try:
        proc = launch_script("open_gripper.py")
        threading.Thread(target=_watch_proc, args=(proc,), daemon=True).start()
        return jsonify({"status": "ok"})
    except Exception as e:
        global _is_running
        _is_running = False
        _run_lock.release()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/audio", methods=["POST"])
def play_audio():
    """
    Play a welcome sound via AudioFeedbackInterface.
    NOTE: No robot lock — can play while actions run.
    """
    try:
        audio_interface.send("greeting")  # ← only this line runs for Welcome
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/status", methods=["GET"])
def status():
    return jsonify({"running": _is_running})

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" Stretch GUI Server ".center(70))
    print(f" Scripts dir: {SCRIPTS_DIR}".center(70))
    print(" http://127.0.0.1:5007 ".center(70))
    print("="*70 + "\n", flush=True)
    app.run(host="127.0.0.1", port=5007, debug=False, use_reloader=False)
