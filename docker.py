# Use the exact base image your org expects (from your supervisor commands).
# This should be a GPU-capable image (CUDA + Python), usually from dkr.eu.
FROM dkr.eu/REPLACE/WITH/BASE_IMAGE:TAG

WORKDIR /workspace

# System packages often needed for pip builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates curl wget unzip gnupg \
 && rm -rf /var/lib/apt/lists/*

# Copy both codebases into the image
COPY nuplan-devkit/ /workspace/nuplan-devkit/
COPY emoe/ /workspace/emoe/

# Install Python deps:
# 1) nuplan-devkit requirements
# 2) emoe requirements
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118

# ------------------------------------------------------------
# NATTEN — install from LOCAL wheel (no expired certs)
# You must have: wheels/natten-0.14.6*.whl in build context
# ------------------------------------------------------------
COPY wheels/ /wheels/
RUN pip install --no-cache-dir /wheels/natten-0.14.6*.whl

# ------------------------------------------------------------
# nuPlan devkit
# (editable install + its requirements)
# ------------------------------------------------------------
RUN pip install --no-cache-dir -r /workspace/nuplan-devkit/requirements.txt && \
    pip install --no-cache-dir -e /workspace/nuplan-devkit

# ------------------------------------------------------------
# EMOE / PLUTO requirements
# (this replaces: pip install -r ./requirements.txt in setup_env.sh)
# ------------------------------------------------------------
RUN pip install --no-cache-dir -r /workspace/emoe/requirements.txt

# ------------------------------------------------------------
# Make EMOE importable
# ------------------------------------------------------------
RUN if [ -f /workspace/emoe/setup.py ] || [ -f /workspace/emoe/pyproject.toml ]; then \
      pip install --no-cache-dir -e /workspace/emoe ; \
    fi

# Entrypoint to run classification + S3 sync
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

ENTRYPOINT ["/workspace/entrypoint.sh"]
