#!/bin/bash
#
# AI-RAN Environment Setup Script
# Run this once in the RHOAI workbench to install all dependencies
# Author: Fatih E. NAR
#

set -e  # Exit on error

echo "=================================================="
echo "AI-RAN Environment Setup"
echo "=================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "⚠️  requirements.txt not found!"
    echo "Please run this script from the aerial-airan directory"
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt
echo ""

# Create workspace directories
echo "📁 Creating workspace directories..."
mkdir -p /opt/app-root/src/data
mkdir -p /opt/app-root/src/models
mkdir -p /opt/app-root/src/results
echo "  ✅ Created: data/, models/, results/"
echo ""

# Optional: Clone neural_rx repository
read -p "📥 Clone neural_rx reference repository? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd /opt/app-root/src
    if [ -d "neural_rx" ]; then
        echo "  ⚠️  neural_rx directory already exists, skipping..."
    else
        echo "  Cloning neural_rx from GitHub..."
        git clone https://github.com/NVlabs/neural_rx.git
        cd neural_rx && pip install -r requirements.txt
        echo "  ✅ neural_rx cloned and installed"
    fi
    cd /opt/app-root/src
    echo ""
fi

# Verify installation
echo "🔍 Verifying installation..."
python << 'EOF'
import sys

try:
    import tensorflow as tf
    import sionna
    import h5py
    import matplotlib
    import seaborn
    import scipy
    import numpy as np
    import tqdm

    print("\n✅ Package versions:")
    print(f"  TensorFlow: {tf.__version__}")
    print(f"  Sionna: {sionna.__version__}")
    print(f"  h5py: {h5py.__version__}")
    print(f"  NumPy: {np.__version__}")
    print(f"  matplotlib: {matplotlib.__version__}")
    print(f"  seaborn: {seaborn.__version__}")
    print(f"  scipy: {scipy.__version__}")

    print("\n✅ GPU Detection:")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            print(f"  {gpu.name}: {gpu.device_type}")
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"\n  Total GPUs: {len(gpus)}")
    else:
        print("  ⚠️ No GPU detected!")
        sys.exit(1)

    print("\n✅ CUDA/cuDNN Check:")
    print(f"  CUDA available: {tf.test.is_built_with_cuda()}")
    print(f"  GPU available: {tf.test.is_gpu_available()}")

except ImportError as e:
    print(f"\n❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    sys.exit(1)
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ Setup Complete!"
    echo "=================================================="
    echo ""
    echo "Next steps:"
    echo "  1. Upload the 5 notebooks to JupyterLab"
    echo "  2. Start with 00-environment-validation.ipynb"
    echo "  3. Follow notebooks in sequence (01 → 02 → 03 → 04)"
    echo ""
    echo "🚀 Ready for AI-RAN experiments!"
else
    echo ""
    echo "=================================================="
    echo "❌ Setup Failed"
    echo "=================================================="
    echo ""
    echo "Please check the error messages above and retry."
    exit 1
fi
