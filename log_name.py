#!/usr/bin/env python3
"""
Given a tokens.txt (scenario tokens), scan ALL nuPlan split DBs and output (log_name, scenario_token) pairs.

Outputs:
  - selected_scenarios.csv    (log_name,scenario_token)
  - selected_scenarios.jsonl  {"log_name":..., "token":...}

Usage:
  export NUPLAN_DATA_ROOT=/path/to/nuplan
  python tokens_to_logname_pairs.py --split trainval --tokens tokens.txt --out_dir ./out

Notes:
  - Does NOT require you to list DB files.
  - Does NOT assume a 'scenario' table exists; will try 'scenario' then 'scene' then heuristics.
"""

import argparse
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import csv
import json


# -----------------------
# SQLite helpers
# -----------------------
def list_tables(conn: sqlite3.Connection) -> Set[str]:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return {r[0] for r in cur.fetchall()}


def table_columns(conn: sqlite3.Connection, table: str) -> Set[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
    return {r[1] for r in cur.fetchall()}


def first_existing_table(tables: Set[str], candidates: Sequence[str]) -> Optional[str]:
    for t in candidates:
        if t in tables:
            return t
    return None


def choose_token_column(cols: Set[str]) -> Optional[str]:
    # scenario token column is usually 'token' in nuPlan DBs
    if "token" in cols:
        return "token"
    # fallback names if schema differs
    for c in ["scenario_token", "scene_token", "id_token"]:
        if c in cols:
            return c
    return None


def choose_log_token_column(cols: Set[str]) -> Optional[str]:
    # typical linking column from scenario/scene -> log
    for c in ["log_token", "log_id", "log"]:
        if c in cols:
            return c
    return None


def choose_log_table_and_namecol(conn: sqlite3.Connection, tables: Set[str]) -> Optional[Tuple[str, str, str]]:
    """
    Returns (log_table, log_token_col, log_name_col) if possible.
    Common: log(token, log_name)
    """
    if "log" not in tables:
        return None
    cols = table_columns(conn, "log")
    # token column in log table
    log_token_col = "token" if "token" in cols else None
    if log_token_col is None:
        # rare fallback
        for c in ["log_token", "id", "uuid"]:
            if c in cols:
                log_token_col = c
                break
    if log_token_col is None:
        return None

    # name column in log table
    log_name_col = None
    for c in ["log_name", "name", "logfile", "filename"]:
        if c in cols:
            log_name_col = c
            break
    if log_name_col is None:
        return None

    return ("log", log_token_col, log_name_col)


def find_mapping_in_db(
    db_path: Path,
    wanted_tokens: Set[str],
) -> Dict[str, str]:
    """
    Return mapping token -> log_name found in this DB.
    """
    out: Dict[str, str] = {}
    if not wanted_tokens:
        return out

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        tables = list_tables(conn)

        # Pick scenario-like table
        scen_table = first_existing_table(tables, ["scenario", "scene"])
        if scen_table is None:
            # Heuristic: find any table that has 'token' and *some* log link
            for t in sorted(tables):
                cols = table_columns(conn, t)
                if "token" in cols and any(k in cols for k in ["log_token", "log_id", "log"]):
                    scen_table = t
                    break
        if scen_table is None:
            return out  # not a scenario DB (or schema too different)

        scen_cols = table_columns(conn, scen_table)
        token_col = choose_token_column(scen_cols)
        log_link_col = choose_log_token_column(scen_cols)
        if token_col is None or log_link_col is None:
            return out

        log_info = choose_log_table_and_namecol(conn, tables)
        if log_info is None:
            return out
        log_table, log_token_col, log_name_col = log_info

        # SQLite IN clause max vars is often 999; batch it
        wanted_list = list(wanted_tokens)
        B = 900

        for i in range(0, len(wanted_list), B):
            chunk = wanted_list[i : i + B]
            qmarks = ",".join(["?"] * len(chunk))

            # Join scenario/scene -> log to get log_name
            # Handle log_link_col being log_id-like vs token-like:
            # Most nuPlan DBs use log_token and log.token.
            query = f"""
                SELECT s.{token_col} AS scen_token, l.{log_name_col} AS log_name
                FROM {scen_table} s
                JOIN {log_table} l
                  ON s.{log_link_col} = l.{log_token_col}
                WHERE s.{token_col} IN ({qmarks})
            """
            try:
                rows = conn.execute(query, chunk).fetchall()
            except sqlite3.OperationalError:
                # If join failed (schema mismatch), give up for this DB
                return out

            for r in rows:
                tok = str(r["scen_token"])
                ln = str(r["log_name"])
                out[tok] = ln

        return out

    finally:
        conn.close()


# -----------------------
# Main
# -----------------------
def read_tokens_file(path: Path) -> List[str]:
    toks: List[str] = []
    for line in path.read_text().splitlines():
        t = line.strip()
        if not t:
            continue
        toks.append(t)
    # de-dupe but preserve order
    seen = set()
    out = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="Split name, e.g. mini, trainval")
    ap.add_argument("--tokens", required=True, type=str, help="tokens.txt with one scenario token per line")
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
    print(f"[INFO] #wanted_tokens: {len(tokens)}")

    token_to_log: Dict[str, str] = {}
    per_db_found: List[Tuple[str, int]] = []

    remaining = set(tokens)
    for db in db_files:
        if not remaining:
            break
        m = find_mapping_in_db(db, remaining)
        per_db_found.append((db.name, len(m)))
        token_to_log.update(m)
        remaining -= set(m.keys())

    found = len(token_to_log)
    missing = [t for t in tokens if t not in token_to_log]

    print(f"[INFO] Found: {found} / {len(tokens)}")
    print(f"[INFO] Missing: {len(missing)}")
    print("[INFO] Per-DB hits (top 15):")
    for name, cnt in sorted(per_db_found, key=lambda x: x[1], reverse=True)[:15]:
        print(f"  {name}: {cnt}")

    # Write outputs
    csv_path = out_dir / "selected_scenarios.csv"
    jsonl_path = out_dir / "selected_scenarios.jsonl"
    missing_path = out_dir / "missing_tokens.txt"
    stats_path = out_dir / "summary.json"

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["log_name", "scenario_token"])
        for t in tokens:
            if t in token_to_log:
                w.writerow([token_to_log[t], t])

    with jsonl_path.open("w") as f:
        for t in tokens:
            if t in token_to_log:
                f.write(json.dumps({"log_name": token_to_log[t], "token": t}) + "\n")

    missing_path.write_text("\n".join(missing) + ("\n" if missing else ""))

    summary = {
        "split": args.split,
        "split_dir": str(split_dir),
        "db_files": len(db_files),
        "wanted_tokens": len(tokens),
        "found": found,
        "missing": len(missing),
        "per_db_found": [{"db": n, "count": c} for n, c in per_db_found],
        "outputs": {
            "csv": str(csv_path),
            "jsonl": str(jsonl_path),
            "missing_tokens": str(missing_path),
        },
    }
    stats_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"[DONE] Wrote:\n  {csv_path}\n  {jsonl_path}\n  {missing_path}\n  {stats_path}")


if __name__ == "__main__":
    main()
