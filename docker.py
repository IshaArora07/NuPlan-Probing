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
RUN pip install --no-cache-dir -r /workspace/nuplan-devkit/requirements.txt && \
    pip install --no-cache-dir -r /workspace/emoe/requirements.txt

# Make both importable.
# nuplan-devkit is typically installed editable.
RUN pip install --no-cache-dir -e /workspace/nuplan-devkit

# If your emoe folder is a proper package (has setup.py or pyproject.toml), install editable.
# If it is not, we'll use PYTHONPATH in entrypoint.
RUN if [ -f /workspace/emoe/setup.py ] || [ -f /workspace/emoe/pyproject.toml ]; then \
      pip install --no-cache-dir -e /workspace/emoe ; \
    fi

# Entrypoint to run classification + S3 sync
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

ENTRYPOINT ["/workspace/entrypoint.sh"]
