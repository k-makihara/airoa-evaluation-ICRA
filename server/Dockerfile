FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    VIRTUAL_ENV=/workspace/.venv \
    PATH="/workspace/.venv/bin:${PATH}" \
    PYTHONPATH="/workspace"

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    git-lfs \
    build-essential \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    libgl1 \
    libglib2.0-0 \
    libaom-dev \
    cmake \
    ninja-build \
    iproute2 \
    iputils-ping \
    net-tools \
    dnsutils \
 && rm -rf /var/lib/apt/lists/* \
 && git lfs install

WORKDIR /workspace

COPY packages/policy-client /workspace/packages/policy-client
COPY src/Isaac-GR00T /workspace/src/Isaac-GR00T
COPY runtime_core /workspace/runtime_core
COPY server /workspace/server

RUN python3 -m venv /workspace/.venv \
 && /workspace/.venv/bin/pip install --upgrade pip setuptools wheel \
 && /workspace/.venv/bin/pip install -e /workspace/packages/policy-client \
 && /workspace/.venv/bin/pip install -e /workspace/src/Isaac-GR00T[base] \
 && (/workspace/.venv/bin/pip uninstall -y transformer-engine || true)

RUN /workspace/.venv/bin/pip install --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
 && /workspace/.venv/bin/pip install --force-reinstall numpy==1.26.4 \
 && (/workspace/.venv/bin/pip uninstall -y opencv-python opencv-python-headless || true) \
 && (/workspace/.venv/bin/pip uninstall -y flash-attn flash_attn || true) \
 && (rm -rf /usr/local/lib/python3.10/dist-packages/cv2 || true) \
 && /workspace/.venv/bin/pip install opencv-python==4.8.0.74 \
 && /workspace/.venv/bin/pip install "accelerate>=0.26.0"

RUN chmod +x /workspace/server/entrypoint.sh

ENTRYPOINT ["/workspace/server/entrypoint.sh"]
