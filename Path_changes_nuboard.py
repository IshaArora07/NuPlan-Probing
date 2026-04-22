"""
fix_nuboard_paths.py

Patches the .nuboard file to point to your local simulation_log folder
instead of the Docker /work/... path.

Usage:
python fix_nuboard_paths.py
"""

import pickle
import glob
import os

# ── Config ───────────────────────────────────────────────────

TRAINING_DIR = os.path.expanduser("~/Desktop/Thesis/Training3")
LOCAL_SIM_LOG = os.path.join(TRAINING_DIR, "simulation_log")

# ─────────────────────────────────────────────────────────────

def fix_paths(obj, local_sim_log):
    """Recursively replace docker simulation_log paths with local path."""

    if isinstance(obj, dict):
        return {k: fix_paths(v, local_sim_log) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [fix_paths(i, local_sim_log) for i in obj]

    elif isinstance(obj, str):
        # Only patch paths that actually contain simulation_log
        if "/work" in obj and "simulation_log" in obj:
            try:
                # Extract suffix after simulation_log
                suffix = obj.split("simulation_log", 1)[1]
                new_path = os.path.join(local_sim_log, suffix.lstrip("/"))

                print(f"  Replacing:")
                print(f"    OLD: {obj}")
                print(f"    NEW: {new_path}")

                return new_path
            except Exception:
                # fallback: replace entire string
                print(f"  [WARN] Fallback replace for: {obj}")
                return local_sim_log

    return obj


def main():
    # ── Find .nuboard file ───────────────────────────────────

    matches = glob.glob(os.path.join(TRAINING_DIR, "*.nuboard"))

    if not matches:
        print(f"ERROR: No .nuboard file found in {TRAINING_DIR}")
        return

    nuboard_path = matches[0]

    print(f"Found .nuboard file: {nuboard_path}")
    print(f"Local simulation_log: {LOCAL_SIM_LOG}")

    if not os.path.exists(LOCAL_SIM_LOG):
        print(f"WARNING: simulation_log folder not found at {LOCAL_SIM_LOG}")
        print("Make sure you have synced simulation_log from S3 first.\n")

    # ── Load ────────────────────────────────────────────────

    with open(nuboard_path, "rb") as f:
        data = pickle.load(f)

    print("\nBefore patching:")
    print(data)

    # ── Patch ───────────────────────────────────────────────

    data = fix_paths(data, LOCAL_SIM_LOG)

    print("\nAfter patching:")
    print(data)

    # ── Backup ──────────────────────────────────────────────

    backup_path = nuboard_path + ".backup"

    with open(nuboard_path, "rb") as f:
        original = f.read()

    with open(backup_path, "wb") as f:
        f.write(original)

    print(f"\nBackup saved: {backup_path}")

    # ── Save patched file ───────────────────────────────────

    with open(nuboard_path, "wb") as f:
        pickle.dump(data, f)

    print(f"Patched file saved: {nuboard_path}")
    print("\nDone. Rerun nuBoard.")


if __name__ == "__main__":
    main()
