#!/bin/bash
# Per-process Vina-GPU launcher with CUDA MPS + per-process binary cache.
#
# Usage:
#   run_vina_mps.sh [vina args...]
#
# Environment knobs:
#   VINA_GPU_DEVICE       - GPU index for this process (default: 0). Sets CUDA_VISIBLE_DEVICES.
#   VINA_BIN_DIR          - directory containing AutoDock-Vina-GPU-2-1 binary (default: script dir)
#   VINA_CACHE_ROOT       - root for per-process OpenCL binary caches (default: /tmp/vina_cache)
#   VINA_BOOST_LIB        - boost library path (default: ../../boost_1_77_0/stage/lib relative to script)
#   VINA_DISABLE_MPS=1    - skip MPS startup (default: starts MPS if not already running)
#
# Each invocation gets its own cache subdir (per-PID) so concurrent processes
# never race on writing Kernel*_Opt.bin. Run prewarm_vina_cache.sh once first
# to seed a shared parent cache that processes copy from.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VINA_BIN_DIR="${VINA_BIN_DIR:-$SCRIPT_DIR}"
VINA_CACHE_ROOT="${VINA_CACHE_ROOT:-/tmp/vina_cache}"
VINA_BOOST_LIB="${VINA_BOOST_LIB:-/mnt/sda/home/tuanhai/LigoSpace/library_data/boost_1_77_0/stage/lib}"
VINA_GPU_DEVICE="${VINA_GPU_DEVICE:-0}"

export CUDA_VISIBLE_DEVICES="$VINA_GPU_DEVICE"
export LD_LIBRARY_PATH="${VINA_BOOST_LIB}:${LD_LIBRARY_PATH:-}"

# --- Start CUDA MPS daemon if not already running and not disabled.
# MPS lets multiple processes co-execute on the same GPU instead of
# time-slicing, which is the main win for memory-bound kernels.
if [[ "${VINA_DISABLE_MPS:-0}" != "1" ]]; then
    export CUDA_MPS_PIPE_DIRECTORY="/tmp/nvidia-mps-${VINA_GPU_DEVICE}"
    export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-log-${VINA_GPU_DEVICE}"
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    if ! pgrep -f "nvidia-cuda-mps-control -d" > /dev/null 2>&1; then
        nvidia-cuda-mps-control -d 2>/dev/null || \
            echo "[run_vina_mps] warning: failed to start MPS daemon (may need root or persistence mode)" >&2
    fi
fi

# --- Per-process OpenCL binary cache directory.
# Default name keys off $$ (this shell's PID); override via VINA_CACHE_TAG
# for stable Ray-managed names if desired.
CACHE_TAG="${VINA_CACHE_TAG:-proc_$$}"
CACHE_DIR="${VINA_CACHE_ROOT}/${CACHE_TAG}"
mkdir -p "$CACHE_DIR"

# Seed from prewarmed shared cache if present, so we don't pay clBuildProgram
# more than once per host. Cache files: Kernel1_Opt.bin, Kernel2_Opt.bin.
if [[ -d "${VINA_CACHE_ROOT}/_prewarm" ]]; then
    for f in "${VINA_CACHE_ROOT}/_prewarm"/Kernel*_Opt.bin; do
        [[ -f "$f" ]] && cp -n "$f" "$CACHE_DIR/" 2>/dev/null || true
    done
fi

cd "$VINA_BIN_DIR"
exec ./AutoDock-Vina-GPU-2-1 --opencl_binary_path "$CACHE_DIR" --gpu_device "$VINA_GPU_DEVICE" "$@"
