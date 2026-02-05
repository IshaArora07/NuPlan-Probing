import os
import shutil
import subprocess
from pathlib import Path

# (1) Choose where to store nuPlan on the cluster.
# Use a large filesystem (e.g., /scratch, /data, /efs, /fsx). Do NOT use your home directory.
NUPLAN_ROOT = Path("/mnt/data/nuplan").expanduser()   # <-- change this to your cluster path
NUPLAN_ROOT.mkdir(parents=True, exist_ok=True)

# (2) Bucket (official)
S3_BUCKET = "s3://motional-nuplan"  # official AWS Open Data bucket

def run(cmd, check=True):
    print(" ".join(cmd))
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout)
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {p.returncode}")
    return p

print("Target download dir:", NUPLAN_ROOT)
print("aws present:", shutil.which("aws") is not None)




import platform
from pathlib import Path

if shutil.which("aws") is None:
    # Downloads AWS CLI v2 and installs to ~/.local/aws-cli (no sudo)
    tmp = Path("/tmp/awscli_install")
    tmp.mkdir(parents=True, exist_ok=True)

    zip_path = tmp / "awscliv2.zip"
    run(["bash", "-lc", f"cd {tmp} && curl -fL https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o {zip_path}"])
    run(["bash", "-lc", f"cd {tmp} && unzip -o {zip_path}"])
    install_dir = Path.home() / ".local" / "aws-cli"
    bin_dir = Path.home() / ".local" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    run(["bash", "-lc", f"{tmp}/aws/install -i {install_dir} -b {bin_dir}"])
    os.environ["PATH"] = f"{bin_dir}:{os.environ['PATH']}"

run(["bash", "-lc", "aws --version"])






def s3_ls(prefix: str = ""):
    prefix = prefix.strip("/")
    target = f"{S3_BUCKET}/{prefix}/" if prefix else f"{S3_BUCKET}/"
    run(["bash", "-lc", f"aws s3 ls --no-sign-request {target}"])

# Example: list top-level only (same as Cell 3)
s3_ls("")







def s3_sync_prefix(prefix: str, dst: Path):
    """
    Download everything under s3://motional-nuplan/<prefix>/ into dst/<prefix>/
    Resumable. No credentials.
    """
    prefix = prefix.strip("/")
    dst_dir = dst / prefix
    dst_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "bash", "-lc",
        f"aws s3 sync --no-sign-request "
        f"'{S3_BUCKET}/{prefix}/' '{dst_dir}/' "
        f"--only-show-errors"
    ]
    run(cmd)

# ====== EDIT THIS LIST ======
PREFIXES_TO_DOWNLOAD = [
    # Put exact prefixes you saw via s3_ls(...)
    # Examples (ONLY keep what exists in your bucket listing):
    # "maps",
    # "nuplan-v1.1/maps",
    # "nuplan-v1.1/mini",
]

for p in PREFIXES_TO_DOWNLOAD:
    s3_sync_prefix(p, NUPLAN_ROOT)











def s3_sync_filtered(prefix: str, dst: Path, includes: list[str], excludes: list[str] | None = None):
    """
    Sync only files matching includes (glob patterns) from a prefix.
    Example includes: ["*.db", "*maps/*", "*las_vegas*"]
    """
    prefix = prefix.strip("/")
    dst_dir = dst / prefix
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Default: exclude everything, then include what you want
    args = ["--exclude", "*"]
    for pat in includes:
        args += ["--include", pat]
    if excludes:
        for pat in excludes:
            args += ["--exclude", pat]

    cmd = (
        f"aws s3 sync --no-sign-request "
        f"'{S3_BUCKET}/{prefix}/' '{dst_dir}/' "
        + " ".join([f"'{a}'" if "*" in a else a for a in args])
        + " --only-show-errors"
    )
    run(["bash", "-lc", cmd])

# ====== EDIT THESE ======
FILTER_PREFIX = ""   # e.g. "nuplan-v1.1"
INCLUDE_PATTERNS = [
    # Examples:
    # "*.db",
    # "*maps*",
]
EXCLUDE_PATTERNS = [
    # Optional examples:
    # "*sensor_blobs*",
]

if FILTER_PREFIX and INCLUDE_PATTERNS:
    s3_sync_filtered(FILTER_PREFIX, NUPLAN_ROOT, INCLUDE_PATTERNS, EXCLUDE_PATTERNS)
else:
    print("Set FILTER_PREFIX and INCLUDE_PATTERNS to use filtered sync.")
