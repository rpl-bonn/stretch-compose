#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, shlex, logging, threading, subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
from utils.audio_utils import AudioFeedbackInterface
import rclpy
import time

audio_interface = AudioFeedbackInterface()

# Adjust if needed
ROS_SETUP = "/opt/ros/humble/setup.bash"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STRETCH_DEMOS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../GUI_scripts"))



# Drawer IDs → names
DRAWER_MAP = {
    "20": "left_white_drawer_upper",
    "19": "left_white_drawer_lower",
    "23": "right_black_drawer_upper",
    "16": "right_black_drawer_lower",
}

DRAWER_AUDIO_MAP =  {
    "20": "the upper white drawer on the left",
    "19": "the lower white drawer on the left",
    "23": "the upper black drawer on the right",
    "16": "the lower black drawer on the right",
}

# Known quick-pick objects
OBJECT_MAP = {
    "watering_can": "watering can",
    "cup": "cup",
    "water_bottle": "bottle",
    "milk_carton": "milk carton",
    "pringles": "pringles",
    "tennis_ball": "tennis ball",
}

def format_name(raw: str) -> str:
    return " ".join(w.capitalize() for w in raw.replace("_", " ").split())

logging.getLogger("werkzeug").setLevel(logging.ERROR)
app = Flask(__name__)
CORS(app)

_run_lock = threading.Lock()
_is_running = False

def idle_greeter():
    while True:
        if not _is_running:
            #audio_interface.send("greeting")
            # only greet every 120 seconds max
            time.sleep(180)
        else:
            # check again sooner if busy
            time.sleep(10)
            
threading.Thread(target=idle_greeter, daemon=True).start()

def launch_command(script: str, arg_name: str, arg_value: str):
    q = shlex.quote(arg_value)
    cmd = (
        f'bash -lc "'
        f'source {shlex.quote(ROS_SETUP)} && '
        f'cd {shlex.quote(STRETCH_DEMOS_DIR)} && '
        f'python3 {shlex.quote(script)} --{arg_name} {q}'
        f'"'
    )
    print(f"[GUI] Launch: {cmd}", flush=True)
    return subprocess.Popen(cmd, shell=True)

def _watch_proc(proc):
    global _is_running
    proc.wait()
    _is_running = False
    try:
        _run_lock.release()
    except Exception:
        pass
    print("[GUI] Action finished.", flush=True)

@app.route("/run", methods=["POST"])
def run_drawer():
    global _is_running
    payload = request.get_json(silent=True) or {}
    drawer_id = str(payload.get("id", "")).strip()
    print(f"[GUI] /run payload: {payload}", flush=True)

    if drawer_id not in DRAWER_MAP:
        audio_interface.send("demo_door_not_impl")
        return jsonify({"status": "error", "message": f"Unknown drawer id: {drawer_id}"}), 400

    if not _run_lock.acquire(blocking=False):
        return jsonify({"status": "busy"}), 409

    try:
        if _is_running:
            _run_lock.release()
            return jsonify({"status": "busy"}), 409
        _is_running = True
        human = format_name(DRAWER_MAP[drawer_id])
        print(f"[GUI] Drawer clicked: id={drawer_id} ({human})", flush=True)
        drawer_audio_name = DRAWER_AUDIO_MAP[drawer_id]
        audio_interface.send("demo_drawer", drawer_id = drawer_audio_name)
        proc = launch_command("demo_open_drawer_new.py", "drawer_id", drawer_id)
        threading.Thread(target=_watch_proc, args=(proc,), daemon=True).start()
        return jsonify({"status": "ok", "drawer_id": drawer_id, "formatted": human})
    except Exception as e:
        _is_running = False
        _run_lock.release()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/grasp", methods=["POST"])
def run_grasp():
    """
    Accepts either:
      - Known quick-pick click: {"object":"water_bottle"}
      - Free-text run:          {"object":"ketchup bottle", "raw":true}
    """
    global _is_running
    payload = request.get_json(silent=True) or {}
    raw_name = str(payload.get("object", "")).strip()
    is_raw = bool(payload.get("raw", False))
    print(f"[GUI] /grasp payload: {payload}", flush=True)
    
    audio_interface.send("demo_grasp", object=raw_name)

    if not raw_name:
        return jsonify({"status": "error", "message": "No object name"}), 400

    # Map quick-picks, otherwise pass raw text through
    mapped = raw_name if is_raw else OBJECT_MAP.get(raw_name, raw_name)

    if not _run_lock.acquire(blocking=False):
        return jsonify({"status": "busy"}), 409

    try:
        if _is_running:
            _run_lock.release()
            return jsonify({"status": "busy"}), 409
        _is_running = True
        print(f"[GUI] Grasp request: {'RAW' if is_raw else 'mapped'}='{mapped}' (from '{raw_name}')", flush=True)
        proc = launch_command("demo_grasping.py", "object", mapped)
        threading.Thread(target=_watch_proc, args=(proc,), daemon=True).start()
        return jsonify({"status": "ok", "object": mapped})
    except Exception as e:
        _is_running = False
        _run_lock.release()
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/status", methods=["GET"])
def status():
    return jsonify({"running": _is_running})

if __name__ == "__main__":
    if not rclpy.ok: 
        rclpy.init(args=None)
        
    print("\n" + "="*70)
    print(" Stretch GUI Server ".center(70))
    print(f" Scripts: {STRETCH_DEMOS_DIR}".center(70))
    print(" http://127.0.0.1:5002 ".center(70))
    print("="*70 + "\n", flush=True)
    app.run(host="127.0.0.1", port=5002, debug=False, use_reloader=False)
