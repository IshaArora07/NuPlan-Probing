#!/usr/bin/env python3
import argparse
from pathlib import Path

def chunk_round_robin(items, k):
    buckets = [[] for _ in range(k)]
    for i, x in enumerate(items):
        buckets[i % k].append(x)
    return buckets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True,
                    help="NUPLAN_DATA_ROOT (folder that contains nuplan-v1.1/splits/...)")
    ap.add_argument("--src_split", type=str, default="trainval",
                    help="Source split folder name under nuplan-v1.1/splits/")
    ap.add_argument("--parts", type=int, default=5)
    ap.add_argument("--dst_prefix", type=str, default="trainval",
                    help="Destination split base name; will create trainval1..trainval5 by default")
    ap.add_argument("--mode", type=str, default="symlink", choices=["symlink", "copy"],
                    help="symlink = no extra disk; copy = duplicates db files")
    args = ap.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    src_dir = data_root / "nuplan-v1.1" / "splits" / args.src_split
    if not src_dir.exists():
        raise FileNotFoundError(f"Source split dir not found: {src_dir}")

    # Collect db files (sometimes nested; we’ll include both top-level and nested)
    dbs = sorted([p for p in src_dir.rglob("*.db") if p.is_file()])
    if not dbs:
        raise RuntimeError(f"No .db files found under: {src_dir}")

    buckets = chunk_round_robin(dbs, args.parts)

    print(f"[INFO] Found {len(dbs)} .db files under {src_dir}")
    for i, bucket in enumerate(buckets, start=1):
        dst_split_name = f"{args.dst_prefix}{i}"
        dst_dir = data_root / "nuplan-v1.1" / "splits" / dst_split_name
        dst_dir.mkdir(parents=True, exist_ok=True)

        # Put symlinks/copies flat into dst_dir (NuPlanScenarioBuilder is fine with flat db_files)
        for src_db in bucket:
            dst_db = dst_dir / src_db.name
            if dst_db.exists() or dst_db.is_symlink():
                continue

            if args.mode == "symlink":
                dst_db.symlink_to(src_db)
            else:
                # copy
                dst_db.write_bytes(src_db.read_bytes())

        print(f"[INFO] {dst_split_name}: {len(bucket)} dbs -> {dst_dir}")

    print("[DONE] Created split folders with dbs as symlinks/copies.")
    print("      You can now use --split trainval1 / trainval2 / ... in your classifier.")

if __name__ == "__main__":
    main()
