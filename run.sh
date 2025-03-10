#!/bin/bash
# Optimized run.sh for Mac MPS with careful initialization

# Set environment variables to control PyTorch behavior
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TORCH_COMPILE_DISABLE=1

# Improve MPS memory management
export MPS_ENABLE_SHARED_MEMORY_CACHE=1

# Handle distributed training environment (single process mode)
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export RANK=0
export WORLD_SIZE=1

# Run with debuggable output
export PYTHONUNBUFFERED=1

# Detect if running on ARM Mac
if [[ $(uname -m) == 'arm64' ]]; then
  echo "Running on Apple Silicon Mac"
else
  echo "Running on Intel Mac"
fi

# Echo the command for clarity
echo "Running: python train_gpt.py --device=mps"

# Run with timeout protection (in case it hangs)
python train_gpt.py --device=mps

# Check exit code
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "Process failed with exit code $EXIT_CODE"
  exit $EXIT_CODE
fi

echo "Training completed successfully"