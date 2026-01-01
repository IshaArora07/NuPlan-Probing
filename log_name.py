#!/usr/bin/env python3
"""
Given a tokens.txt (scenario tokens as hex strings), scan ALL nuPlan split DBs and output
(log_name, scenario_token_hex) pairs.

- Handles nuPlan schema where scene/scenario.token and log_token are BLOB (usually 8 bytes).
- Accepts tokens.txt lines like: 01fcea70e3e4517f (case-insensitive, no hyphens needed).
- Outputs:
  - selected_scenarios.csv    (log_name,scenario_token)
  - selected_scenarios.jsonl  {"log_name":..., "token":...}
  - missing_tokens.txt
  - summary.json

Usage:
  export NUPLAN_DATA_ROOT=/path/to/nuplan
  python tokens_to_logname_pairs.py --split trainval --tokens tokens.txt --out_dir ./out
"""

import argparse
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
import csv
import json

from tqdm import tqdm


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


def choose_token_column(cols: Set[str]) -> Optional[str]:
    # nuPlan typically uses token
    if "token" in cols:
        return "token"
    for c in ["scenario_token", "scene_token", "id_token"]:
        if c in cols:
            return c
    return None


def choose_log_token_column(cols: Set[str]) -> Optional[str]:
    # nuPlan typically uses log_token in scene/scenario
    for c in ["log_token", "log_id", "log"]:
        if c in cols:
            return c
    return None


def choose_log_table_and_namecol(conn: sqlite3.Connection, tables: Set[str]) -> Optional[Tuple[str, str, str]]:
    """
    Returns (log_table, log_token_col, log_name_col).
    Common in nuPlan:
      log(token BLOB, logfile VARCHAR(64))
    Sometimes also:
      log(token, log_name) or (token, name)
    """
    if "log" not in tables:
        return None
    cols = table_columns(conn, "log")

    # token col
    log_token_col = None
    if "token" in cols:
        log_token_col = "token"
    else:
        for c in ["log_token", "id", "uuid"]:
            if c in cols:
                log_token_col = c
                break
    if log_token_col is None:
        return None

    # name col
    log_name_col = None
    for c in ["logfile", "log_name", "name", "logfile_name", "filename"]:
        if c in cols:
            log_name_col = c
            break
    if log_name_col is None:
        return None

    return ("log", log_token_col, log_name_col)


def normalize_hex_token(s: str) -> str:
    """
    Normalize a token line to lowercase hex without 0x and without whitespace.
    """
    t = s.strip()
    if t.startswith("0x") or t.startswith("0X"):
        t = t[2:]
    t = t.replace("-", "").replace(" ", "").replace("\t", "").replace("\r", "")
    return t.lower()


def hex_to_blob_bytes(hex_token: str) -> Optional[bytes]:
    """
    Convert a hex string (e.g. '01fcea70e3e4517f') into bytes for sqlite BLOB matching.
    Returns None if invalid.
    """
    ht = normalize_hex_token(hex_token)
    if not ht:
        return None
    if len(ht) % 2 != 0:
        return None
    try:
        return bytes.fromhex(ht)
    except Exception:
        return None


def find_mapping_in_db(db_path: Path, wanted_hex_tokens: Set[str]) -> Dict[str, str]:
    """
    Return mapping: token_hex_lower -> log_name found in this DB.
    This assumes scene/scenario.token is a BLOB and your wanted tokens are hex strings.
    """
    out: Dict[str, str] = {}
    if not wanted_hex_tokens:
        return out

    # Pre-convert wanted tokens into (hex_lower, blob_bytes)
    wanted_pairs: List[Tuple[str, bytes]] = []
    for ht in wanted_hex_tokens:
        b = hex_to_blob_bytes(ht)
        if b is not None:
            wanted_pairs.append((normalize_hex_token(ht), b))

    if not wanted_pairs:
        return out

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        tables = list_tables(conn)

        scen_table = first_existing_table(tables, ["scene", "scenario"])
        if scen_table is None:
            # heuristic fallback
            for t in sorted(tables):
                cols = table_columns(conn, t)
                if "token" in cols and any(k in cols for k in ["log_token", "log_id", "log"]):
                    scen_table = t
                    break
        if scen_table is None:
            return out

        scen_cols = table_columns(conn, scen_table)
        token_col = choose_token_column(scen_cols)
        log_link_col = choose_log_token_column(scen_cols)
        if token_col is None or log_link_col is None:
            return out

        log_info = choose_log_table_and_namecol(conn, tables)
        if log_info is None:
            return out
        log_table, log_token_col, log_name_col = log_info

        # SQLite variable limit ~999 -> keep safe
        B = 900

        # We query using BLOB params and SELECT hex(token) to map back to hex string
        for i in range(0, len(wanted_pairs), B):
            chunk_pairs = wanted_pairs[i : i + B]
            blob_chunk = [sqlite3.Binary(b) for (_, b) in chunk_pairs]
            qmarks = ",".join(["?"] * len(blob_chunk))

            query = f"""
                SELECT hex(s.{token_col}) AS scen_hex, l.{log_name_col} AS log_name
                FROM {scen_table} s
                JOIN {log_table} l
                  ON s.{log_link_col} = l.{log_token_col}
                WHERE s.{token_col} IN ({qmarks})
            """
            try:
                rows = conn.execute(query, blob_chunk).fetchall()
            except sqlite3.OperationalError:
                return out

            for r in rows:
                # SQLite hex() returns uppercase hex without 0x
                scen_hex = str(r["scen_hex"]).strip().lower()
                ln = str(r["log_name"])
                out[scen_hex] = ln

        return out
    finally:
        conn.close()


# -----------------------
# Main
# -----------------------
def read_tokens_file(path: Path) -> List[str]:
    toks: List[str] = []
    for line in path.read_text().splitlines():
        t = normalize_hex_token(line)
        if not t:
            continue
        toks.append(t)
    # de-dup preserve order
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

    tokens = read_tokens_file(Path(args.tokens))  # normalized lowercase hex
    wanted = set(tokens)

    print(f"[INFO] split_dir: {split_dir}")
    print(f"[INFO] #db_files: {len(db_files)}")
    print(f"[INFO] #wanted_tokens: {len(tokens)}")

    token_to_log: Dict[str, str] = {}
    per_db_found: List[Tuple[str, int]] = []

    remaining = set(tokens)

    for db in tqdm(db_files, total=len(db_files), desc="Scanning DBs", unit="db"):
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
