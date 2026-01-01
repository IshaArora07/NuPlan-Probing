#!/usr/bin/env python3
"""
Given tokens.txt (scenario tokens as hex strings), scan ALL nuPlan split DBs and output (log_name, scenario_token).

This version is BLOB-safe:
  - scene.token is BLOB
  - scene.log_token is BLOB
We match via SQLite HEX(BLOB) strings:
  - lower(hex(scene.token)) compared to tokens.txt normalized strings.

Outputs:
  - selected_scenarios.csv    (log_name,scenario_token_hex_lower)
  - selected_scenarios.jsonl  {"log_name":..., "token":...}
  - missing_tokens.txt

Usage:
  export NUPLAN_DATA_ROOT=/path/to/nuplan
  python tokens_to_logname_pairs_blobsafe.py --split trainval --tokens tokens.txt --out_dir ./out

Notes:
  - Does NOT require listing DB files.
  - Does NOT assume a 'scenario' table exists; tries 'scene' then 'scenario'.
  - Resolves log name from table 'log' column 'logfile' (common in nuPlan DBs).
"""

import argparse
import csv
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from tqdm import tqdm


HEX_RE = re.compile(r"^[0-9a-fA-F]+$")


# -----------------------
# SQLite helpers
# -----------------------
def list_tables(conn: sqlite3.Connection) -> Set[str]:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return {r[0] for r in cur.fetchall()}


def table_columns(conn: sqlite3.Connection, table: str) -> Set[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return {r[1] for r in cur.fetchall()}


def first_existing_table(tables: Set[str], candidates: Sequence[str]) -> Optional[str]:
    for t in candidates:
        if t in tables:
            return t
    return None


def pick_log_name_col(log_cols: Set[str]) -> Optional[str]:
    # Your DB shows: log.logfile
    for c in ["logfile", "log_name", "name", "filename", "log_file", "logpath"]:
        if c in log_cols:
            return c
    return None


def normalize_token_str(s: str) -> Optional[str]:
    """
    Normalize token strings to lower hex without hyphens/spaces.
    Returns None if it doesn't look like hex.
    """
    t = s.strip().lower().replace("-", "").replace(" ", "")
    if not t:
        return None
    if len(t) % 2 != 0:
        # hex(BLOB) always even length
        return None
    if not HEX_RE.match(t):
        return None
    return t


def read_tokens_file(path: Path) -> List[str]:
    toks: List[str] = []
    for line in path.read_text().splitlines():
        nt = normalize_token_str(line)
        if nt is None:
            continue
        toks.append(nt)
    # de-dupe preserving order
    seen = set()
    out = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def build_logtoken_to_logname(conn: sqlite3.Connection) -> Dict[str, str]:
    """
    Build mapping: lower(hex(log.token)) -> log_name (logfile/log_name/etc).
    """
    tables = list_tables(conn)
    if "log" not in tables:
        return {}

    log_cols = table_columns(conn, "log")
    if "token" not in log_cols:
        return {}

    name_col = pick_log_name_col(log_cols)
    if name_col is None:
        return {}

    out: Dict[str, str] = {}
    # log table is typically small; read all
    q = f"SELECT hex(token) AS log_hex, {name_col} AS log_name FROM log"
    for row in conn.execute(q):
        log_hex = (row[0] or "").strip().lower()
        log_name = str(row[1]) if row[1] is not None else ""
        if log_hex:
            out[log_hex] = log_name
    return out


def find_pairs_in_db(db_path: Path, wanted_hex: Set[str]) -> Dict[str, str]:
    """
    Return mapping: scenario_token_hex_lower -> log_name, for matches found in this DB.
    """
    out: Dict[str, str] = {}
    if not wanted_hex:
        return out

    conn = sqlite3.connect(str(db_path))
    try:
        tables = list_tables(conn)

        scen_table = first_existing_table(tables, ["scene", "scenario"])
        if scen_table is None:
            return out

        scen_cols = table_columns(conn, scen_table)
        # Need BLOB token + log_token
        if "token" not in scen_cols or "log_token" not in scen_cols:
            return out

        logtoken_to_name = build_logtoken_to_logname(conn)
        if not logtoken_to_name:
            return out

        # Stream over scene/scenario rows and filter in Python
        q = f"SELECT hex(token) AS scen_hex, hex(log_token) AS log_hex FROM {scen_table}"
        cur = conn.execute(q)

        while True:
            rows = cur.fetchmany(5000)
            if not rows:
                break
            for scen_hex_u, log_hex_u in rows:
                if scen_hex_u is None or log_hex_u is None:
                    continue
                scen_hex = str(scen_hex_u).strip().lower()
                if scen_hex in wanted_hex:
                    log_hex = str(log_hex_u).strip().lower()
                    log_name = logtoken_to_name.get(log_hex, "")
                    if log_name:
                        out[scen_hex] = log_name

        return out
    except sqlite3.Error:
        return out
    finally:
        conn.close()


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="Split name, e.g. mini, trainval")
    ap.add_argument("--tokens", required=True, type=str, help="tokens.txt with one scenario token per line (hex)")
    ap.add_argument("--out_dir", required=True, type=str, help="Output directory")
    ap.add_argument("--db_glob", default="*.db", help="Glob for DB files inside split dir (default: *.db)")
    args = ap.parse_args()

    data_root = os.environ.get("NUPLAN_DATA_ROOT", None)
    if not data_root:
        raise RuntimeError("NUPLAN_DATA_ROOT is not set")

    split_dir = Path(data_root) / "nuplan-v1.1" / "splits" / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split dir not found: {split_dir}")

    db_files = sorted(split_dir.glob(args.db_glob))
    if not db_files:
        raise FileNotFoundError(f"No DB files found in {split_dir} with glob '{args.db_glob}'")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tokens = read_tokens_file(Path(args.tokens))
    wanted = set(tokens)

    print(f"[INFO] split_dir: {split_dir}")
    print(f"[INFO] #db_files: {len(db_files)}")
    print(f"[INFO] #wanted_tokens(valid hex): {len(tokens)}")

    token_to_log: Dict[str, str] = {}
    remaining = set(tokens)

    for db in tqdm(db_files, desc="Scanning DBs", unit="db"):
        if not remaining:
            break
        found_map = find_pairs_in_db(db, remaining)
        if found_map:
            token_to_log.update(found_map)
            remaining -= set(found_map.keys())

    found = len(token_to_log)
    missing = [t for t in tokens if t not in token_to_log]

    print(f"[INFO] Found: {found} / {len(tokens)}")
    print(f"[INFO] Missing: {len(missing)}")

    # Write outputs
    csv_path = out_dir / "selected_scenarios.csv"
    jsonl_path = out_dir / "selected_scenarios.jsonl"
    missing_path = out_dir / "missing_tokens.txt"

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["log_name", "scenario_token"])
        for t in tokens:
            ln = token_to_log.get(t, None)
            if ln is not None:
                w.writerow([ln, t])

    with jsonl_path.open("w") as f:
        for t in tokens:
            ln = token_to_log.get(t, None)
            if ln is not None:
                f.write(json.dumps({"log_name": ln, "token": t}) + "\n")

    missing_path.write_text("\n".join(missing) + ("\n" if missing else ""))

    print(f"[DONE] Wrote:\n  {csv_path}\n  {jsonl_path}\n  {missing_path}")


if __name__ == "__main__":
    main()
