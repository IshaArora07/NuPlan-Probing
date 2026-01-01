#!/usr/bin/env python3
"""
Find (log_name, scenario_token) pairs for a given tokens.txt by scanning all nuPlan split .db files.

Outputs:
  - selected_scenario_tokens.json        : list of [log_name, token]
  - selected_scenario_tokens.txt         : "log_name token" per line
  - selected_scenario_tokens_stats.json  : summary stats

Usage example:
  python tokens_to_log_and_token.py \
    --split_root /data/nuplan/nuplan-v1.1/splits/trainval \
    --tokens_txt /path/to/tokens.txt \
    --out_dir ./out_tokens
"""

import argparse
import json
import os
import sqlite3
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

Token = str


# ----------------------------- helpers -----------------------------
def read_tokens(tokens_txt: Path) -> List[Token]:
    toks: List[str] = []
    with tokens_txt.open("r") as f:
        for line in f:
            t = line.strip()
            if t:
                toks.append(t)
    # de-dup preserving order
    seen = set()
    out = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def is_hex_string(s: str) -> bool:
    if len(s) == 0:
        return False
    try:
        int(s, 16)
        return True
    except Exception:
        return False


def normalize_token_value(v: Union[str, bytes, memoryview, int, None]) -> Optional[str]:
    """
    Normalize sqlite token value to a comparable string:
      - if bytes/memoryview -> hex string
      - if str -> stripped as-is
      - else -> str(...)
    """
    if v is None:
        return None
    if isinstance(v, memoryview):
        v = v.tobytes()
    if isinstance(v, (bytes, bytearray)):
        return bytes(v).hex()
    if isinstance(v, str):
        return v.strip()
    return str(v).strip()


def list_tables(conn: sqlite3.Connection) -> Set[str]:
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {r[0] for r in rows}


def table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    # PRAGMA table_info: (cid, name, type, notnull, dflt_value, pk)
    return [r[1] for r in rows]


def pick_first(existing: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    s = set(existing)
    for c in candidates:
        if c in s:
            return c
    return None


# ----------------------------- core per-db extraction -----------------------------
def extract_log_lookup(conn: sqlite3.Connection, log_table: str) -> Tuple[Dict[str, str], Optional[str], Optional[str]]:
    """
    Returns mapping log_key -> log_name
    where log_key is normalized token/id string, depending on schema.
    """
    cols = table_columns(conn, log_table)

    # log key column: either token (BLOB/TEXT) or id (int)
    log_key_col = pick_first(cols, ["token", "log_token", "id"])
    if log_key_col is None:
        return {}, None, None

    # log name column: try common variants
    log_name_col = pick_first(cols, ["logfile", "log_name", "name", "filename", "file_name"])
    if log_name_col is None:
        # fallback: any column that contains "log" and "name" or "file"
        for c in cols:
            lc = c.lower()
            if ("file" in lc or "log" in lc) and ("name" in lc or "file" in lc):
                log_name_col = c
                break

    if log_name_col is None:
        return {}, log_key_col, None

    lookup: Dict[str, str] = {}
    cur = conn.execute(f"SELECT {log_key_col}, {log_name_col} FROM {log_table}")
    for k, n in cur.fetchall():
        nk = normalize_token_value(k)
        nn = normalize_token_value(n)
        if nk and nn:
            lookup[nk] = nn
    return lookup, log_key_col, log_name_col


def detect_scene_schema(conn: sqlite3.Connection) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Detect:
      - scene/scenario table name
      - scenario token column
      - log reference column (log_token or log_id etc)
    """
    tables = list_tables(conn)

    scene_table = None
    if "scene" in tables:
        scene_table = "scene"
    elif "scenario" in tables:
        scene_table = "scenario"
    else:
        # try any table containing "scene" or "scenario"
        for t in tables:
            tl = t.lower()
            if "scene" in tl:
                scene_table = t
                break
        if scene_table is None:
            for t in tables:
                tl = t.lower()
                if "scenario" in tl:
                    scene_table = t
                    break

    if scene_table is None:
        return None, None, None

    cols = table_columns(conn, scene_table)

    token_col = pick_first(cols, ["token", "scenario_token"])
    if token_col is None:
        # fallback: any column containing "token"
        for c in cols:
            if "token" in c.lower():
                token_col = c
                break

    # how to link to log table
    log_ref_col = pick_first(cols, ["log_token", "log_id", "log_idx", "log"])
    if log_ref_col is None:
        # fallback: any column starting with "log"
        for c in cols:
            if c.lower().startswith("log"):
                log_ref_col = c
                break

    return scene_table, token_col, log_ref_col


def stream_scene_rows(
    conn: sqlite3.Connection, scene_table: str, token_col: str, log_ref_col: str, fetch_size: int = 20000
) -> Iterable[Tuple[Optional[str], Optional[str]]]:
    """
    Stream (scenario_token, log_ref) from scene table.
    """
    cur = conn.execute(f"SELECT {token_col}, {log_ref_col} FROM {scene_table}")
    while True:
        rows = cur.fetchmany(fetch_size)
        if not rows:
            break
        for tok, logref in rows:
            yield normalize_token_value(tok), normalize_token_value(logref)


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_root", required=True, type=str, help="Folder containing many .db files (recursively)")
    ap.add_argument("--tokens_txt", required=True, type=str, help="tokens.txt (one scenario token per line)")
    ap.add_argument("--out_dir", required=True, type=str, help="Output directory")
    ap.add_argument("--stop_when_done", action="store_true", help="Stop scanning DBs once all tokens are found")
    args = ap.parse_args()

    split_root = Path(args.split_root).expanduser().resolve()
    tokens_txt = Path(args.tokens_txt).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tokens_list = read_tokens(tokens_txt)
    tokens_set: Set[str] = set(tokens_list)

    remaining: Set[str] = set(tokens_list)  # shrink as we find tokens
    found_pairs: List[Tuple[str, str]] = []  # (log_name, token)
    found_by_token: Dict[str, str] = {}  # token -> log_name (avoid dupes)
    per_db_found = Counter()
    errors = []

    db_files = sorted(split_root.rglob("*.db"))
    if not db_files:
        raise FileNotFoundError(f"No .db files found under {split_root}")

    print(f"[INFO] split_root: {split_root}")
    print(f"[INFO] tokens_txt: {tokens_txt}  ({len(tokens_list)} unique tokens)")
    print(f"[INFO] found {len(db_files)} db files")

    for db_path in db_files:
        if args.stop_when_done and not remaining:
            break

        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = None

            tables = list_tables(conn)
            if "log" not in tables:
                conn.close()
                continue

            scene_table, token_col, log_ref_col = detect_scene_schema(conn)
            if not scene_table or not token_col or not log_ref_col:
                conn.close()
                continue

            log_lookup, log_key_col, log_name_col = extract_log_lookup(conn, "log")
            if not log_lookup:
                conn.close()
                continue

            # stream scene rows and pick only tokens we need
            local_found = 0
            for tok, logref in stream_scene_rows(conn, scene_table, token_col, log_ref_col):
                if not tok or tok not in remaining:
                    continue

                # map logref -> logname
                log_name = None
                if logref and logref in log_lookup:
                    log_name = log_lookup[logref]
                else:
                    # sometimes scene stores log_id but lookup key is token, or vice versa
                    # try best-effort by also checking token-like conversions
                    if logref and is_hex_string(logref) and logref.lower() in log_lookup:
                        log_name = log_lookup[logref.lower()]
                    elif logref and logref.upper() in log_lookup:
                        log_name = log_lookup[logref.upper()]

                if log_name is None:
                    continue

                found_by_token[tok] = log_name
                remaining.remove(tok)
                local_found += 1

                if args.stop_when_done and not remaining:
                    break

            per_db_found[str(db_path.name)] += local_found

            conn.close()

            if local_found > 0:
                print(f"[INFO] {db_path.name}: found {local_found} (remaining {len(remaining)})")

        except Exception as e:
            errors.append({"db": str(db_path), "error": repr(e)})
            try:
                conn.close()
            except Exception:
                pass
            continue

    # preserve original order
    for tok in tokens_list:
        if tok in found_by_token:
            found_pairs.append((found_by_token[tok], tok))

    out_json = out_dir / "selected_scenario_tokens.json"
    out_txt = out_dir / "selected_scenario_tokens.txt"
    out_stats = out_dir / "selected_scenario_tokens_stats.json"

    with out_json.open("w") as f:
        json.dump([[ln, tk] for ln, tk in found_pairs], f)

    with out_txt.open("w") as f:
        for ln, tk in found_pairs:
            f.write(f"{ln} {tk}\n")

    stats = {
        "split_root": str(split_root),
        "tokens_txt": str(tokens_txt),
        "num_input_tokens_unique": len(tokens_list),
        "num_found": len(found_pairs),
        "num_missing": len(remaining),
        "missing_examples": list(sorted(remaining))[:20],
        "db_files_scanned": len(db_files),
        "dbs_with_hits": sum(1 for _, v in per_db_found.items() if v > 0),
        "top_dbs_by_hits": per_db_found.most_common(20),
        "errors_count": len(errors),
        "errors_examples": errors[:10],
    }
    with out_stats.open("w") as f:
        json.dump(stats, f, indent=2)

    print("\n[SUMMARY]")
    print(f"  input tokens (unique): {len(tokens_list)}")
    print(f"  found:                {len(found_pairs)}")
    print(f"  missing:              {len(remaining)}")
    print(f"  wrote: {out_json}")
    print(f"        {out_txt}")
    print(f"        {out_stats}")


if __name__ == "__main__":
    main()
