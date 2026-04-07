FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl wget git git-lfs \
    build-essential pkg-config \
    python3 python3-venv python3-dev \
    libaom-dev \
    cmake ninja-build \
    iproute2 iputils-ping net-tools dnsutils \
 && rm -rf /var/lib/apt/lists/* \
 && git lfs install

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /workspace
# Copy pyproject.toml for dependencies
COPY pyproject.toml .
# Install dependencies from pyproject.toml
RUN pip install -e .[base]
# There's a conflict in the native python, so we have to resolve it by
RUN pip uninstall -y transformer-engine
RUN pip install flash_attn==2.7.1.post4 -U --force-reinstall
# Clean any existing OpenCV installations
RUN pip uninstall -y opencv-python opencv-python-headless || true
RUN rm -rf /usr/local/lib/python3.10/dist-packages/cv2 || true
RUN pip install opencv-python==4.8.0.74
RUN pip install --force-reinstall torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 numpy==1.26.4
COPY getting_started /workspace/getting_started
COPY scripts /workspace/scripts
COPY demo_data /workspace/demo_data
RUN pip install -e . --no-deps
# need to install accelerate explicitly to avoid version conflicts
RUN pip install accelerate>=0.26.0
