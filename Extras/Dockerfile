# cNMF Docker image (sklearn backend)
#
# Builds a self-contained environment for running cNMF with the default
# sklearn NMF backend.  PyTorch is intentionally excluded because the correct
# wheel depends on the host GPU driver (CUDA version).  See the README for
# instructions on adding GPU support at build time or at runtime.
#
# Build:
#   docker build -t cnmf -f Extras/Dockerfile .
#
# Run (interactive):
#   docker run --rm -it -v /path/to/data:/data cnmf
#
# Run (single command):
#   docker run --rm -v /path/to/data:/data cnmf \
#     cnmf prepare --output-dir /data --name my_run -c /data/counts.h5ad \
#     -k 7 8 9 10 --n-iter 100 --seed 14

FROM python:3.11-slim

LABEL maintainer="PhilipPeffer" \
      description="Consensus NMF (cNMF) for scRNA-seq — sklearn backend"

# System dependencies needed by scanpy / scipy / matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libhdf5-dev \
        pkg-config \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy source and install in editable mode so the cnmf CLI is on PATH
WORKDIR /opt/cnmf
COPY . .
RUN pip install --no-cache-dir -e ".[test]"

WORKDIR /data
ENTRYPOINT []
CMD ["/bin/bash"]
