#!/usr/bin/env python3
"""
Given a tokens.txt (scenario tokens), scan ALL nuPlan split DBs and output (logfile, scenario_token) pairs.

Outputs (written incrementally as results are found):
  - selected_scenarios.csv    (logfile,scenario_token)
  - selected_scenarios.jsonl  {"logfile":..., "token":...}
Final outputs:
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
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import csv
import json

from tqdm import tqdm


# -----------------------
# SQLite helpers
# -----------------------
def list_tables(conn: sqlite3.Connection) -> Set[str]:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return {r[0] for r in cur.fetchall()}


def table_columns(conn: sqlite3.Connection, table: str) -> List[Tuple[int, str, str, int, str, int]]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
    return [(int(r[0]), str(r[1]), str(r[2]), int(r[3]), str(r[4]), int(r[5])) for r in cur.fetchall()]


def first_existing_table(tables: Set[str], candidates: Sequence[str]) -> Optional[str]:
    for t in candidates:
        if t in tables:
            return t
    return None


def choose_token_column(cols: Set[str]) -> Optional[str]:
    if "token" in cols:
        return "token"
    for c in ["scenario_token", "scene_token", "id_token"]:
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
    Common: log(token, log_name) OR log(token, logfile)
    """
    if "log" not in tables:
        return None

    cols_info = table_columns(conn, "log")
    cols = {name for _, name, *_ in cols_info}

    log_token_col = "token" if "token" in cols else None
    if log_token_col is None:
        for c in ["log_token", "id", "uuid"]:
            if c in cols:
                log_token_col = c
                break
    if log_token_col is None:
        return None

    log_name_col = None
    for c in ["log_name", "logfile", "name", "log_file", "filename"]:
        if c in cols:
            log_name_col = c
            break
    if log_name_col is None:
        return None

    return ("log", log_token_col, log_name_col)


def _infer_token_representation(conn: sqlite3.Connection, scen_table: str, token_col: str) -> str:
    """
    Returns "BLOB" or "TEXT".
    Uses PRAGMA table_info + a quick typeof() probe.
    """
    try:
        cols_info = table_columns(conn, scen_table)
        for _, name, coltype, *_ in cols_info:
            if name == token_col:
                ct = (coltype or "").upper()
                if "BLOB" in ct:
                    return "BLOB"
                if "CHAR" in ct or "TEXT" in ct or "VARCHAR" in ct:
                    return "TEXT"
                break
    except Exception:
        pass

    try:
        row = conn.execute(f"SELECT typeof({token_col}) AS t FROM {scen_table} WHERE {token_col} IS NOT NULL LIMIT 1").fetchone()
        if row is not None and row[0] is not None:
            t = str(row[0]).upper()
            if t == "BLOB":
                return "BLOB"
            if t == "TEXT":
                return "TEXT"
    except Exception:
        pass

    # default
    return "TEXT"


def _token_txt_to_blob(token_txt: str) -> Optional[bytes]:
    """
    Convert token string (likely hex) -> bytes for binding to SQLite BLOB column.
    Supports:
      - hex without 0x prefix
      - hex with 0x prefix
      - ignores hyphens/spaces
    Returns None if not valid hex.
    """
    t = token_txt.strip().lower()
    if not t:
        return None
    if t.startswith("0x"):
        t = t[2:]
    t = t.replace("-", "").replace(" ", "")
    if any(ch not in "0123456789abcdef" for ch in t):
        return None
    if len(t) % 2 != 0:
        # cannot be bytes if odd hex length
        return None
    try:
        return bytes.fromhex(t)
    except Exception:
        return None


def find_mapping_in_db(
    db_path: Path,
    remaining_tokens_txt: Set[str],
) -> Tuple[Dict[str, str], str, str, str]:
    """
    Return:
      - mapping token_txt -> log_name/logfile found in this DB
      - chosen scen_table
      - token_representation ("BLOB" or "TEXT")
      - chosen log_name_col
    """
    out: Dict[str, str] = {}
    if not remaining_tokens_txt:
        return out, "", "", ""

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        tables = list_tables(conn)

        scen_table = first_existing_table(tables, ["scenario", "scene"])
        if scen_table is None:
            for t in sorted(tables):
                cols_info = table_columns(conn, t)
                cols = {name for _, name, *_ in cols_info}
                if "token" in cols and any(k in cols for k in ["log_token", "log_id", "log"]):
                    scen_table = t
                    break
        if scen_table is None:
            return out, "", "", ""

        scen_cols_info = table_columns(conn, scen_table)
        scen_cols = {name for _, name, *_ in scen_cols_info}

        token_col = choose_token_column(scen_cols)
        log_link_col = choose_log_token_column(scen_cols)
        if token_col is None or log_link_col is None:
            return out, scen_table, "", ""

        log_info = choose_log_table_and_namecol(conn, tables)
        if log_info is None:
            return out, scen_table, "", ""
        log_table, log_token_col, log_name_col = log_info

        token_repr = _infer_token_representation(conn, scen_table, token_col)

        # Prepare wanted list in the right binding type (BLOB vs TEXT)
        # We'll keep a reverse map so we can convert back to the original token_txt.
        bind_vals: List[Union[str, bytes]] = []
        bind_to_txt: Dict[Union[str, bytes], str] = {}

        if token_repr == "BLOB":
            for ttxt in remaining_tokens_txt:
                b = _token_txt_to_blob(ttxt)
                if b is None:
                    continue
                bind_vals.append(b)
                bind_to_txt[b] = ttxt
        else:
            for ttxt in remaining_tokens_txt:
                bind_vals.append(ttxt)
                bind_to_txt[ttxt] = ttxt

        if not bind_vals:
            return out, scen_table, token_repr, log_name_col

        # SQLite IN clause max vars is often 999; batch it
        B = 900
        for i in range(0, len(bind_vals), B):
            chunk = bind_vals[i : i + B]
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
                return out, scen_table, token_repr, log_name_col

            for r in rows:
                scen_tok = r["scen_token"]
                ln = str(r["log_name"])

                if token_repr == "BLOB":
                    # scen_tok will be bytes-like for BLOB
                    if isinstance(scen_tok, (bytes, bytearray, memoryview)):
                        b = bytes(scen_tok)
                        ttxt = bind_to_txt.get(b, None)
                        if ttxt is None:
                            # fallback: hex string (lower)
                            ttxt = b.hex()
                        out[ttxt] = ln
                    else:
                        # unexpected, fallback stringify
                        out[str(scen_tok)] = ln
                else:
                    # TEXT
                    ttxt = str(scen_tok)
                    # If token text differs only by case, map to the original requested one (best effort)
                    # but we still store under the token we have in remaining_tokens_txt set.
                    if ttxt in remaining_tokens_txt:
                        out[ttxt] = ln
                    else:
                        low = ttxt.lower()
                        if low in remaining_tokens_txt:
                            out[low] = ln
                        else:
                            out[ttxt] = ln

        return out, scen_table, token_repr, log_name_col

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
    # de-dup preserve order
    seen = set()
    out: List[str] = []
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
    wanted_set = set(tokens)
    remaining = set(tokens)

    print(f"[INFO] split_dir: {split_dir}")
    print(f"[INFO] #db_files: {len(db_files)}")
    print(f"[INFO] #wanted_tokens: {len(tokens)}")

    # Output paths
    csv_path = out_dir / "selected_scenarios.csv"
    jsonl_path = out_dir / "selected_scenarios.jsonl"
    missing_path = out_dir / "missing_tokens.txt"
    stats_path = out_dir / "summary.json"

    # Incremental writers
    token_to_log: Dict[str, str] = {}
    written: Set[str] = set()
    per_db_found: List[Dict[str, object]] = []

    # Open outputs once; append as we find results
    with csv_path.open("w", newline="") as fcsv, jsonl_path.open("w") as fjsonl:
        w = csv.writer(fcsv)
        w.writerow(["log_name", "scenario_token"])  # keep header name stable

        for db in tqdm(db_files, total=len(db_files), desc="Scanning DB files"):
            if not remaining:
                break

            m, scen_table, token_repr, log_name_col = find_mapping_in_db(db, remaining)

            per_db_found.append(
                {
                    "db": db.name,
                    "found": int(len(m)),
                    "scen_table": scen_table,
                    "token_repr": token_repr,
                    "log_name_col": log_name_col,
                }
            )

            if not m:
                continue

            token_to_log.update(m)

            # Append newly found rows immediately
            for tok_txt, log_name in m.items():
                if tok_txt in written:
                    continue
                written.add(tok_txt)
                w.writerow([log_name, tok_txt])
                fjsonl.write(json.dumps({"log_name": log_name, "token": tok_txt}) + "\n")

            fcsv.flush()
            fjsonl.flush()

            remaining -= set(m.keys())

    found = len(token_to_log)
    missing = [t for t in tokens if t not in token_to_log]

    missing_path.write_text("\n".join(missing) + ("\n" if missing else ""))

    summary = {
        "split": args.split,
        "split_dir": str(split_dir),
        "db_files": len(db_files),
        "wanted_tokens": len(tokens),
        "found": found,
        "missing": len(missing),
        "per_db_found": per_db_found,
        "outputs": {
            "csv": str(csv_path),
            "jsonl": str(jsonl_path),
            "missing_tokens": str(missing_path),
        },
    }
    stats_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"[INFO] Found: {found} / {len(tokens)}")
    print(f"[INFO] Missing: {len(missing)}")
    print("[INFO] Per-DB hits (top 15):")
    for d in sorted(per_db_found, key=lambda x: int(x["found"]), reverse=True)[:15]:
        print(f"  {d['db']}: {d['found']} (scen_table={d['scen_table']}, token_repr={d['token_repr']}, log_col={d['log_name_col']})")

    print(f"[DONE] Wrote incrementally:\n  {csv_path}\n  {jsonl_path}\nFinal:\n  {missing_path}\n  {stats_path}")


if __name__ == "__main__":
    main()
