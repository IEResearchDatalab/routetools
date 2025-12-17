#!/usr/bin/env bash
# Source this script to set GPU-first defaults for JAX in your shell/terminal.
# Usage: source scripts/activate_gpu.sh

export JAX_PLATFORM_NAME=${JAX_PLATFORM_NAME:-cuda}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# Avoid preallocating all GPU memory (optional, tweak as needed)
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}

echo "Set JAX_PLATFORM_NAME=$JAX_PLATFORM_NAME CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
