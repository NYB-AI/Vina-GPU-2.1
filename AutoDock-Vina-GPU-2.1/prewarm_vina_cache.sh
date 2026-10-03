#!/bin/bash
# Pre-warm the OpenCL binary cache by running Vina-GPU once on the example
# input. This populates ${VINA_CACHE_ROOT}/_prewarm/Kernel*_Opt.bin which
# run_vina_mps.sh then copies into each per-process cache, avoiding ~5-30s
# of duplicate clBuildProgram per Ray task.
#
# Usage:
#   prewarm_vina_cache.sh             # uses default GPU 0
#   VINA_GPU_DEVICE=1 prewarm_vina_cache.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VINA_CACHE_ROOT="${VINA_CACHE_ROOT:-/tmp/vina_cache}"
VINA_BOOST_LIB="${VINA_BOOST_LIB:-/mnt/sda/home/tuanhai/LigoSpace/library_data/boost_1_77_0/stage/lib}"
VINA_GPU_DEVICE="${VINA_GPU_DEVICE:-0}"

export CUDA_VISIBLE_DEVICES="$VINA_GPU_DEVICE"
export LD_LIBRARY_PATH="${VINA_BOOST_LIB}:${LD_LIBRARY_PATH:-}"

PREWARM_DIR="${VINA_CACHE_ROOT}/_prewarm"
mkdir -p "$PREWARM_DIR"

cd "$SCRIPT_DIR"
echo "[prewarm] building OpenCL kernel binaries into $PREWARM_DIR"
./AutoDock-Vina-GPU-2-1 \
    --config input_file_example/2bm2_config.txt \
    --opencl_binary_path "$PREWARM_DIR" \
    --gpu_device "$VINA_GPU_DEVICE" \
    --thread 1000 > "$PREWARM_DIR/prewarm.log" 2>&1

echo "[prewarm] done. Cached kernels:"
ls -la "$PREWARM_DIR"/Kernel*_Opt.bin
