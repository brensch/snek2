FROM golang:1.25.5-bookworm

WORKDIR /workspace

# Python for training + export
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ca-certificates \
    git \
  && rm -rf /var/lib/apt/lists/*

# Install Python deps (torch/polars/etc)
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv \
  && /opt/venv/bin/python -m pip install --upgrade pip \
  && /opt/venv/bin/python -m pip install --no-cache-dir -r /tmp/requirements.txt

ENV PATH=/opt/venv/bin:$PATH

# Pre-fetch Go deps for faster startup
COPY go.mod go.sum ./
RUN go mod download

# Copy the repo
COPY . .

ENV PYTHONUNBUFFERED=1

# Ensure runtime can find ONNX Runtime shared lib shipped in repo root
ENV LD_LIBRARY_PATH=/workspace

ENTRYPOINT ["/workspace/infrastructure/pipeline/orchestrate.sh"]
