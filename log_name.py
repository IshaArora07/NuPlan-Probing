#!/usr/bin/env python3
"""
Given a tokens.txt (scenario tokens as 16 hex chars per line), scan ALL nuPlan split DBs and output
(logfile, scenario_token_hex) pairs.

Outputs (written incrementally while scanning):
  - selected_scenarios.csv    (log_name,scenario_token)
  - selected_scenarios.jsonl  {"log_name":..., "token":...}
  - missing_tokens.txt
  - summary.json

Usage:
  export NUPLAN_DATA_ROOT=/path/to/nuplan
  python tokens_to_logname_pairs.py --split trainval --tokens tokens.txt --out_dir ./out

Notes:
  - Does NOT require you to list DB files.
  - Handles nuPlan schema where scene.token and scene.log_token are BLOB(8).
  - tokens.txt lines must be 16 hex chars (case-insensitive). Whitespace is ignored.
"""

import argparse
import csv
import json
import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

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
    for c in ["token", "scenario_token", "scene_token", "id_token"]:
        if c in cols:
            return c
    return None


def choose_log_token_column(cols: Set[str]) -> Optional[str]:
    for c in ["log_token", "log_id", "log"]:
        if c in cols:
            return c
    return None


def choose_log_table_and_namecol(conn: sqlite3.Connection, tables: Set[str]) -> Optional[Tuple[str, str, str]]:
    """
    Returns (log_table, log_token_col, log_name_col) if possible.
    nuPlan DBs commonly have:
      log(token BLOB, logfile VARCHAR)
    """
    if "log" not in tables:
        return None

    cols = table_columns(conn, "log")

    # token column in log table
    log_token_col = None
    for c in ["token", "log_token", "id", "uuid"]:
        if c in cols:
            log_token_col = c
            break
    if log_token_col is None:
        return None

    # name column in log table
    log_name_col = None
    for c in ["logfile", "log_name", "name", "filename"]:
        if c in cols:
            log_name_col = c
            break
    if log_name_col is None:
        return None

    return ("log", log_token_col, log_name_col)


def find_mapping_in_db(
    db_path: Path,
    wanted_tokens_blob: Set[bytes],
) -> Dict[bytes, str]:
    """
    Return mapping token_blob -> log_name found in this DB.
    """
    out: Dict[bytes, str] = {}
    if not wanted_tokens_blob:
        return out

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        tables = list_tables(conn)

        # scenario-like table
        scen_table = first_existing_table(tables, ["scenario", "scene"])
        if scen_table is None:
            for t in sorted(tables):
                cols = table_columns(conn, t)
                if choose_token_column(cols) and choose_log_token_column(cols):
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

        # SQLite variable limit batching
        wanted_list = list(wanted_tokens_blob)
        B = 900

        for i in range(0, len(wanted_list), B):
            chunk = wanted_list[i : i + B]
            qmarks = ",".join(["?"] * len(chunk))

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
                return out

            for r in rows:
                tok_blob = bytes(r["scen_token"])  # BLOB -> bytes
                ln = str(r["log_name"])
                out[tok_blob] = ln

        return out
    finally:
        conn.close()


# -----------------------
# Token parsing
# -----------------------
def normalize_hex_token(line: str) -> Optional[str]:
    """
    Accepts a line that should contain 16 hex chars (case-insensitive).
    Returns lowercase 16-hex string, or None if invalid.
    """
    t = line.strip().lower()
    if not t:
        return None
    # allow optional "0x"
    if t.startswith("0x"):
        t = t[2:]
    # must be exactly 16 hex chars for BLOB(8)
    if len(t) != 16:
        return None
    try:
        int(t, 16)
    except ValueError:
        return None
    return t


def read_tokens_file_hex(path: Path) -> List[str]:
    toks: List[str] = []
    for line in path.read_text().splitlines():
        t = normalize_hex_token(line)
        if t is None:
            continue
        toks.append(t)
    # de-dupe preserve order
    seen = set()
    out = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def hex_to_blob(tok_hex: str) -> bytes:
    return bytes.fromhex(tok_hex)


def blob_to_hex(tok_blob: bytes) -> str:
    return tok_blob.hex()


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="Split name, e.g. mini, trainval")
    ap.add_argument("--tokens", required=True, type=str, help="tokens.txt with one token per line (16 hex chars)")
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

    tokens_hex = read_tokens_file_hex(Path(args.tokens))
    if not tokens_hex:
        raise RuntimeError("No valid tokens read from tokens file. Expected 16-hex-char tokens per line.")
    tokens_blob = [hex_to_blob(t) for t in tokens_hex]

    # fast membership
    wanted_blob: Set[bytes] = set(tokens_blob)

    print(f"[INFO] split_dir: {split_dir}")
    print(f"[INFO] #db_files: {len(db_files)}")
    print(f"[INFO] #wanted_tokens(valid hex): {len(tokens_hex)}")

    csv_path = out_dir / "selected_scenarios.csv"
    jsonl_path = out_dir / "selected_scenarios.jsonl"
    missing_path = out_dir / "missing_tokens.txt"
    stats_path = out_dir / "summary.json"

    token_to_log: Dict[bytes, str] = {}
    per_db_found: List[Tuple[str, int]] = []

    # Write incrementally
    csv_f = csv_path.open("w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow(["log_name", "scenario_token"])

    jsonl_f = jsonl_path.open("w")

    try:
        remaining: Set[bytes] = set(tokens_blob)

        pbar = tqdm(db_files, total=len(db_files), desc="Scanning DBs", unit="db")
        for db in pbar:
            if not remaining:
                break

            m = find_mapping_in_db(db, remaining)
            per_db_found.append((db.name, len(m)))

            if m:
                # append only newly found
                for tok_blob, log_name in m.items():
                    if tok_blob in token_to_log:
                        continue
                    token_to_log[tok_blob] = log_name
                    tok_hex = blob_to_hex(tok_blob)  # 16 hex chars, lowercase
                    csv_w.writerow([log_name, tok_hex])
                    jsonl_f.write(json.dumps({"log_name": log_name, "token": tok_hex}) + "\n")

                csv_f.flush()
                jsonl_f.flush()

                remaining -= set(m.keys())

            pbar.set_postfix(found=len(token_to_log), remaining=len(remaining))

    finally:
        jsonl_f.close()
        csv_f.close()

    found = len(token_to_log)

    # tokens missing (in original order)
    missing_hex: List[str] = []
    for t_hex, t_blob in zip(tokens_hex, tokens_blob):
        if t_blob not in token_to_log:
            missing_hex.append(t_hex)

    missing_path.write_text("\n".join(missing_hex) + ("\n" if missing_hex else ""))

    summary = {
        "split": args.split,
        "split_dir": str(split_dir),
        "db_files": len(db_files),
        "wanted_tokens": len(tokens_hex),
        "found": found,
        "missing": len(missing_hex),
        "per_db_found_top15": [{"db": n, "count": c} for n, c in sorted(per_db_found, key=lambda x: x[1], reverse=True)[:15]],
        "outputs": {
            "csv": str(csv_path),
            "jsonl": str(jsonl_path),
            "missing_tokens": str(missing_path),
        },
    }
    stats_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"[DONE] Found {found} / {len(tokens_hex)}")
    print(f"[DONE] Wrote:\n  {csv_path}\n  {jsonl_path}\n  {missing_path}\n  {stats_path}")


if __name__ == "__main__":
    main()
