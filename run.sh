# Run with debuggable output
export PYTHONUNBUFFERED=1

# Detect if running on ARM Mac
if [[ $(uname -m) == 'arm64' ]]; then
  echo "Running on Apple Silicon Mac"
else
  echo "Running on Intel Mac"
fi

# Echo the command for clarity
echo "Running: python train_gpt_mod.py --device=mps"

# Run with timeout protection (in case it hangs)
python train_gpt_mod.py --device=mps

# Check exit code
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "Process failed with exit code $EXIT_CODE"
  exit $EXIT_CODE
fi

echo "Training completed successfully"