#!/usr/bin/env python3
"""
Recover (log_name, scenario_token) pairs from nuPlan split DBs given scenario tokens.

Inputs:
  - tokens.txt: one scenario token per line

Outputs:
  - scenario_tokens.json : JSON list of [log_name, token] pairs (Hydra-friendly)
  - scenario_tokens.txt  : "log_name token" per line
  - missing_tokens.txt   : tokens not found in split

Usage:
  export NUPLAN_DATA_ROOT=/path/to/nuplan

  python recover_log_names_from_tokens.py \
    --split trainval \
    --tokens_txt /path/to/tokens.txt \
    --out_dir ./out_tokens \
    --db_glob "*.db"

Notes:
  - This does NOT instantiate scenarios, so it’s much faster than ScenarioBuilder.
  - Assumes tokens are *scenario tokens* (not lidarpc tokens).
"""

import os
import json
import argparse
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from glob import glob


def read_tokens(tokens_txt: Path) -> List[str]:
    tokens = []
    with tokens_txt.open("r") as f:
        for line in f:
            t = line.strip()
            if t:
                tokens.append(t)
    # de-dup preserving order
    seen = set()
    out = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
        (table,),
    )
    return cur.fetchone() is not None


def get_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    cur = conn.execute(f"PRAGMA table_info({table});")
    return [r[1] for r in cur.fetchall()]  # r[1] is column name


def chunked(seq: Sequence[str], n: int) -> List[List[str]]:
    return [list(seq[i : i + n]) for i in range(0, len(seq), n)]


def pick_first(existing: List[str], candidates: List[str]) -> Optional[str]:
    s = set(existing)
    for c in candidates:
        if c in s:
            return c
    return None


def resolve_log_name_from_log_row(row: sqlite3.Row, log_cols: List[str]) -> Optional[str]:
    """
    Try common log-name columns. nuPlan DBs often have something like:
      - log.logfile
      - log.name
      - log.log_name
    """
    for c in ["logfile", "name", "log_name"]:
        if c in log_cols and row[c] is not None:
            return str(row[c])
    return None


def build_query(conn: sqlite3.Connection) -> Tuple[str, str, str]:
    """
    Returns (sql, mode, logname_expr_description)

    mode:
      - "direct": scenario table has log_name column
      - "join":  scenario joins log table
    """
    if not table_exists(conn, "scenario"):
        raise RuntimeError("DB does not contain a 'scenario' table.")

    scenario_cols = get_columns(conn, "scenario")

    # Best case: scenario has log_name directly
    if "log_name" in scenario_cols:
        sql = "SELECT token AS scenario_token, log_name AS log_name FROM scenario WHERE token IN ({placeholders})"
        return sql, "direct", "scenario.log_name"

    # Otherwise, try join with log table
    if not table_exists(conn, "log"):
        raise RuntimeError(
            "DB 'scenario' table has no log_name column and DB has no 'log' table to join."
        )

    log_cols = get_columns(conn, "log")

    # scenario.log_token -> log.token is the typical pattern
    scen_fk = pick_first(scenario_cols, ["log_token", "log_id", "log"])
    log_pk = pick_first(log_cols, ["token", "id", "log_token"])

    if scen_fk is None or log_pk is None:
        raise RuntimeError(
            f"Could not identify scenario->log join keys. "
            f"scenario cols={scenario_cols}, log cols={log_cols}"
        )

    # pick the best available log name expression
    log_name_col = pick_first(log_cols, ["logfile", "name", "log_name"])
    if log_name_col is None:
        raise RuntimeError(
            f"Could not find a log name column in log table. log cols={log_cols}"
        )

    sql = (
        "SELECT s.token AS scenario_token, l.{log_name_col} AS log_name "
        "FROM scenario s "
        "JOIN log l ON s.{scen_fk} = l.{log_pk} "
        "WHERE s.token IN ({placeholders})"
    ).format(log_name_col=log_name_col, scen_fk=scen_fk, log_pk=log_pk)

    return sql, "join", f"JOIN scenario.{scen_fk} -> log.{log_pk}, log.{log_name_col}"


def fetch_pairs_from_db(db_path: Path, tokens: List[str]) -> Dict[str, str]:
    """
    Returns mapping: scenario_token -> log_name for tokens found in this DB.
    """
    out: Dict[str, str] = {}

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        sql_template, mode, desc = build_query(conn)

        # SQLite IN (...) has a practical parameter limit; 900 is safe.
        for chunk in chunked(tokens, 900):
            placeholders = ",".join(["?"] * len(chunk))
            sql = sql_template.format(placeholders=placeholders)

            cur = conn.execute(sql, chunk)
            rows = cur.fetchall()
            for r in rows:
                st = str(r["scenario_token"])
                ln = r["log_name"]
                if ln is None:
                    continue
                out[st] = str(ln)

    finally:
        conn.close()

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, type=str, help="mini, trainval, etc.")
    parser.add_argument("--tokens_txt", required=True, type=str, help="Path to tokens.txt (one scenario token per line)")
    parser.add_argument("--out_dir", required=True, type=str, help="Output directory")
    parser.add_argument("--db_glob", default="*.db", type=str, help="DB filename glob under split directory (default: *.db)")
    args = parser.parse_args()

    data_root = os.environ.get("NUPLAN_DATA_ROOT", None)
    if not data_root:
        raise RuntimeError("Please set NUPLAN_DATA_ROOT.")

    split_root = Path(data_root) / "nuplan-v1.1" / "splits" / args.split
    if not split_root.exists():
        raise FileNotFoundError(f"Split root not found: {split_root}")

    tokens_txt = Path(args.tokens_txt).expanduser().resolve()
    tokens = read_tokens(tokens_txt)
    if not tokens:
        raise RuntimeError(f"No tokens found in: {tokens_txt}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    db_paths = [Path(p) for p in sorted(glob(str(split_root / args.db_glob)))]
    if not db_paths:
        raise RuntimeError(f"No DB files found under {split_root} with glob {args.db_glob}")

    # We’ll progressively remove tokens as we find them to avoid repeated work
    remaining = list(tokens)
    found_map: Dict[str, str] = {}

    print(f"[INFO] split_root={split_root}")
    print(f"[INFO] tokens_in={len(tokens)} (deduped)")
    print(f"[INFO] db_files={len(db_paths)}")

    for i, dbp in enumerate(db_paths):
        if not remaining:
            break
        print(f"[INFO] ({i+1}/{len(db_paths)}) scanning {dbp.name} ... remaining={len(remaining)}")
        m = fetch_pairs_from_db(dbp, remaining)
        if m:
            found_map.update(m)
            found_set = set(m.keys())
            remaining = [t for t in remaining if t not in found_set]

    # Preserve original token order for outputs
    pairs: List[List[str]] = []
    for t in tokens:
        ln = found_map.get(t, None)
        if ln is not None:
            pairs.append([ln, t])

    missing = [t for t in tokens if t not in found_map]

    # Write outputs
    json_path = out_dir / "scenario_tokens.json"
    txt_path = out_dir / "scenario_tokens.txt"
    missing_path = out_dir / "missing_tokens.txt"

    json_path.write_text(json.dumps(pairs, indent=2))
    with txt_path.open("w") as f:
        for ln, t in pairs:
            f.write(f"{ln} {t}\n")
    missing_path.write_text("\n".join(missing) + ("\n" if missing else ""))

    print("\n[SUMMARY]")
    print(f"  found   : {len(pairs)}")
    print(f"  missing : {len(missing)}")
    print(f"  wrote   : {json_path}")
    print(f"  wrote   : {txt_path}")
    print(f"  wrote   : {missing_path}")


if __name__ == "__main__":
    main()
