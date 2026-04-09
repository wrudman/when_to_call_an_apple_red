from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import json
import os
from supabase import create_client, Client

# ---------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------

app = Flask(__name__, static_folder="static", static_url_path="")

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------
# Supabase setup
# ---------------------------------------------------------------------

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("Missing Supabase environment variables")

supabase: Client = create_client(
    SUPABASE_URL,
    SUPABASE_SERVICE_KEY
)

# ---------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------

PROFILE_DIR = (
    BASE_DIR.parent
    / "data"
    / "prolific_stimuli"
    / "profiles"
)

PROFILE_FILES = {
    p.stem: p
    for p in PROFILE_DIR.glob("profile_*.json")
    if not p.name.startswith("debug")
}

assert len(PROFILE_FILES) == 74, f"Expected 74 profiles, found {len(PROFILE_FILES)}"

# Debug profile (never assigned)
DEBUG_PROFILE_PATH = PROFILE_DIR / "debug_profile.json"
assert DEBUG_PROFILE_PATH.exists(), "debug_profile.json not found"

# ---------------------------------------------------------------------
# Profile assignment helpers
# ---------------------------------------------------------------------

def claim_profile(prolific_pid: str):
    resp = (
        supabase
        .table("profile_assignments")
        .select("*")
        .eq("completed", False)
        .limit(1)
        .execute()
    )

    if not resp.data:
        return None

    profile = resp.data[0]

    supabase.table("profile_assignments").update({
        "assigned_to": prolific_pid,
        "assigned_at": "now()"
    }).eq("profile_id", profile["profile_id"]).execute()
    
    return  profile


def release_profile(profile_id: str):
    """
    Release a profile after dropout.
    """
    supabase.table("profile_assignments").update({
        "assigned_to": None,
        "assigned_at": None
    }).eq("profile_id", profile_id).execute()


def complete_profile(profile_id: str):
    """
    Mark a profile as completed.
    """
    supabase.table("profile_assignments").update({
        "completed": True
    }).eq("profile_id", profile_id).execute()

# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/get_profile")
def get_profile():

    """prolific_pid = request.args.get("PROLIFIC_PID")

    if not prolific_pid:
        return jsonify({"error": "Missing PROLIFIC_PID"}), 400

    is_test = prolific_pid.startswith("TEST_") or prolific_pid == "DEBUG"

    # Block re-entry for real participants
    if not is_test:
        existing = (
            supabase
            .table("results")
            .select("exit_reason")
            .eq("prolific_pid", prolific_pid)
            .limit(1)
            .execute()
        )

        if existing.data:
            return jsonify({
                "status": "blocked",
                "reason": existing.data[0]["exit_reason"]
            }), 403

    # Claim a profile
    profile = claim_profile(prolific_pid)
    print(profile)

    if profile is None:
        return jsonify({
            "status": "no_profiles_left"
        }), 410

    profile_id = profile["profile_id"]

    if profile_id not in PROFILE_FILES:
        return jsonify({"error": "Profile file not found"}), 500

    with open(PROFILE_FILES[profile_id], "r") as f:
        profile_data = json.load(f)

    return jsonify({
        "profile_id": profile_id,
        "profile_index": profile["profile_index"],
        "questions": profile_data["questions"],
    })"""
    # hard code last missing profile
    profile_id = "profile_2_first"
    profile_path = PROFILE_FILES[profile_id]

    with open(profile_path, "r") as f:
        profile = json.load(f)

    return jsonify({
        "profile_id": profile_id,
        "profile_index": 70,  # known index
        "questions": profile["questions"],
    })


@app.route("/save_results", methods=["POST"])
def save_results():
    payload = request.get_json()

    prolific_pid = payload.get("PROLIFIC_PID", "UNKNOWN")
    data = payload.get("data", [])

    meta = data[0] if data else {}

    profile_id = meta.get("profile_id")
    exit_reason = meta.get("exit_reason", "completed")

    row = {
        "prolific_pid": prolific_pid,
        "profile_id": profile_id,
        "profile_index": meta.get("profile_index"),
        "exit_reason": exit_reason,
        "experiment_start_time": meta.get("experiment_start_time"),
        "exit_time": meta.get("exit_time"),
        "data": data,
    }

    supabase.table("results").insert(row).execute()

    # Update profile state
    if profile_id:
        if exit_reason == "completed":
            complete_profile(profile_id)
        else:
            release_profile(profile_id)

    return jsonify({"status": "ok"}), 200


@app.route("/finish.html")
def finish():
    return send_from_directory("static", "finish.html")


@app.route("/decline.html")
def decline():
    return send_from_directory("static", "decline.html")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
