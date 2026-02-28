#!/bin/bash
# GPU Test Launcher - Sets CUDA library paths before running Python

# Add CUDA libraries to library path
export LD_LIBRARY_PATH="$HOME/.local/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib/python3.12/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib/python3.12/site-packages/nvidia/cufft/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib/python3.12/site-packages/nvidia/curand/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib/python3.12/site-packages/nvidia/cusolver/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib/python3.12/site-packages/nvidia/cusparse/lib:$LD_LIBRARY_PATH"

echo "🔧 Setting CUDA library paths..."
echo "LD_LIBRARY_PATH configured for cuML/CuPy"
echo ""

# Run the Python script
python3 "$@"
