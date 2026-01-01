#!/usr/bin/env python3
"""
Given:
  - a nuPlan sqlite DB (with tables like log + scenario OR log + scene)
  - a tokens.txt (one token per line)

Output:
  - scenario_tokens.json : list of [log_name, token] pairs (ScenarioFilter.scenario_tokens compatible)
  - scenario_tokens.tsv  : tab-separated log_name  token
  - summary stats printed

Usage:
  python tokens_to_log_and_token_pairs.py \
    --db /path/to/nuplan.db \
    --tokens tokens.txt \
    --out_dir ./out_pairs
"""

import argparse
import json
import os
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def read_tokens(tokens_path: Path) -> List[str]:
    toks = []
    with tokens_path.open("r") as f:
        for line in f:
            t = line.strip()
            if t and not t.startswith("#"):
                toks.append(t)
    # de-dup, preserve order
    seen = set()
    out = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    )
    return cur.fetchone() is not None


def get_table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.execute(f"PRAGMA table_info({table});")
    return [row[1] for row in cur.fetchall()]  # row[1] = column name


def choose_scene_table(conn: sqlite3.Connection) -> str:
    # Prefer "scenario" if present, else fallback to "scene"
    if table_exists(conn, "scenario"):
        return "scenario"
    if table_exists(conn, "scene"):
        return "scene"
    # last resort: try to find any table that has a "token" column and looks plausible
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cur.fetchall()]
    candidates = []
    for t in tables:
        cols = get_table_columns(conn, t)
        if "token" in cols:
            candidates.append(t)
    raise RuntimeError(
        f"Could not find 'scenario' or 'scene' table. Tables with a 'token' column: {candidates}"
    )


def choose_log_join_columns(conn: sqlite3.Connection, scene_table: str) -> Tuple[str, str]:
    """
    Returns (scene_log_fk_col, log_name_col)
    Typical:
      scene_table has: log_token (FK -> log.token)
      log table has: log_name (or name)
    """
    if not table_exists(conn, "log"):
        raise RuntimeError("DB has no 'log' table, cannot map tokens -> log_name.")

    scene_cols = set(get_table_columns(conn, scene_table))
    log_cols = set(get_table_columns(conn, "log"))

    # Find FK column in scene/scenario table pointing to log
    for fk in ["log_token", "log_id", "log", "log_key"]:
        if fk in scene_cols:
            scene_log_fk_col = fk
            break
    else:
        # try anything containing 'log' and 'token' first
        maybe = [c for c in scene_cols if ("log" in c.lower() and "token" in c.lower())]
        if maybe:
            scene_log_fk_col = maybe[0]
        else:
            raise RuntimeError(
                f"Couldn't find a log foreign-key column in '{scene_table}'. "
                f"Columns are: {sorted(scene_cols)}"
            )

    # Find log name column
    for lc in ["log_name", "name", "logfile", "filename"]:
        if lc in log_cols:
            log_name_col = lc
            break
    else:
        raise RuntimeError(
            f"Couldn't find a log name column in 'log'. Columns are: {sorted(log_cols)}"
        )

    return scene_log_fk_col, log_name_col


def build_token_to_logname(
    conn: sqlite3.Connection, scene_table: str, scene_log_fk_col: str, log_name_col: str
) -> Dict[str, str]:
    """
    Build mapping token -> log_name using a join:
      SELECT s.token, l.<log_name_col>
      FROM <scene_table> s
      JOIN log l ON s.<scene_log_fk_col> = l.token
    """
    # log primary key token is typically "token"
    log_cols = set(get_table_columns(conn, "log"))
    if "token" not in log_cols:
        raise RuntimeError(f"'log' table doesn't have 'token' column. Columns: {sorted(log_cols)}")

    query = f"""
    SELECT s.token as scene_token, l.{log_name_col} as log_name
    FROM {scene_table} s
    JOIN log l
      ON s.{scene_log_fk_col} = l.token
    """
    cur = conn.execute(query)
    mapping = {}
    for scene_token, log_name in cur.fetchall():
        if scene_token is None or log_name is None:
            continue
        mapping[str(scene_token)] = str(log_name)
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, type=str, help="Path to nuPlan sqlite DB file")
    ap.add_argument("--tokens", required=True, type=str, help="Path to tokens.txt (one token per line)")
    ap.add_argument("--out_dir", required=True, type=str, help="Output directory")
    ap.add_argument("--strict", action="store_true", help="If set, fail if any token is missing in DB")
    args = ap.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    tokens_path = Path(args.tokens).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        raise FileNotFoundError(db_path)
    if not tokens_path.exists():
        raise FileNotFoundError(tokens_path)

    tokens = read_tokens(tokens_path)
    print(f"[INFO] Read {len(tokens)} unique tokens from {tokens_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        scene_table = choose_scene_table(conn)
        scene_log_fk_col, log_name_col = choose_log_join_columns(conn, scene_table)

        print(f"[INFO] Using scene/scenario table: {scene_table}")
        print(f"[INFO] Join columns: {scene_table}.{scene_log_fk_col} -> log.token, log.{log_name_col} as log_name")

        token_to_log = build_token_to_logname(conn, scene_table, scene_log_fk_col, log_name_col)
        print(f"[INFO] Built token->log_name map for {len(token_to_log)} rows from DB")

        pairs: List[List[str]] = []
        missing = []
        for t in tokens:
            ln = token_to_log.get(t)
            if ln is None:
                missing.append(t)
                continue
            pairs.append([ln, t])

        if missing:
            msg = f"[WARN] {len(missing)} / {len(tokens)} tokens not found in DB."
            if args.strict:
                raise RuntimeError(msg + " Use without --strict to write the found subset.")
            print(msg)
            print("[WARN] First 10 missing tokens:", missing[:10])

        # Write outputs
        json_path = out_dir / "scenario_tokens.json"
        tsv_path = out_dir / "scenario_tokens.tsv"

        with json_path.open("w") as f:
            json.dump(pairs, f)

        with tsv_path.open("w") as f:
            for ln, t in pairs:
                f.write(f"{ln}\t{t}\n")

        # Stats
        log_counts = Counter([p[0] for p in pairs])
        print(f"[INFO] Wrote {len(pairs)} pairs to:")
        print(f"       - {json_path}")
        print(f"       - {tsv_path}")
        print(f"[INFO] Unique logs in selection: {len(log_counts)}")
        print("[INFO] Top 10 logs by count:")
        for k, v in log_counts.most_common(10):
            print(f"  {k}: {v}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
