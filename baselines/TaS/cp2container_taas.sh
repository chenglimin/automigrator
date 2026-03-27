#!/usr/bin/env bash
set -euo pipefail

CID="${1:?Usage: $0 <container_id_or_name> [dest_dir]}"
DEST="${2:-/openhands/code}"

sudo docker cp ncnn_with_demo "$CID:$DEST"
sudo docker cp riscv64-glibc-ubuntu-24.04-gcc-nightly-2025.06.13-nightly.tar.xz "$CID:$DEST"
sudo docker cp config_taas.sh "$CID:$DEST"

