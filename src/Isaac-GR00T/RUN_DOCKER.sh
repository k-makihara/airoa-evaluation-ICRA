#!/bin/bash
set -euo pipefail

# コンテナ名とイメージ名を変数化しておくと後で変更しやすい
CONTAINER_NAME="gr00t-dev"
IMAGE_NAME="gr00t-uv:cu121"

docker run --rm -it \
  --gpus all \
  --network host \
  --shm-size=16g \
  -e LCM_DEFAULT_URL="udpm://239.255.76.67:7667?ttl=1" \
  -v "$PWD":/workspace \
  -w /workspace \
  --name "$CONTAINER_NAME" \
  "$IMAGE_NAME"
