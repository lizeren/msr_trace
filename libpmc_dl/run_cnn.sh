#!/bin/bash
# Quick Start Script for 1D CNN Training on RTX 4090

echo "======================================================================"
echo "PMC 1D CNN - Quick Start Guide"
echo "======================================================================"
echo ""

# Check if we're on the right machine
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  WARNING: nvidia-smi not found. Are you on the GPU host?"
    echo ""
fi

# Check CUDA availability
echo "Step 1: Checking GPU availability..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ GPU detected: {torch.cuda.get_device_name(0)}')
    print(f'   CUDA version: {torch.version.cuda}')
    print(f'   Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('❌ No GPU detected!')
    print('   Install PyTorch with CUDA:')
    print('   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121')
    exit(1)
" || exit 1

echo ""
echo "Step 2: Checking data..."
if [ ! -d "features" ]; then
    echo "❌ features/ directory not found!"
    echo "   Please ensure PMC feature JSON files are in features/"
    exit 1
fi

NUM_FILES=$(ls features/pmc_features_*.json 2>/dev/null | wc -l)
if [ $NUM_FILES -eq 0 ]; then
    echo "❌ No feature files found!"
    echo "   Expected: features/pmc_features_*.json"
    exit 1
fi

echo "✅ Found $NUM_FILES feature files"
echo ""

echo "Step 3: Starting training..."
echo ""
echo "Configuration:"
echo "  • Model: 1D CNN"
echo "  • Sequence length: 128"
echo "  • Batch size: 64 (adjust based on GPU memory)"
echo "  • Epochs: 50 (with early stopping)"
echo "  • Learning rate: 1e-3"
echo ""
echo "Training will take approximately 5-10 minutes on RTX 4090..."
echo ""
echo "======================================================================"
echo ""

# Train with default settings
python3 train_cnn.py \
    --features "features/pmc_features_*.json" \
    --seq-len 128 \
    --batch-size 64 \
    --epochs 50 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --dropout 0.3 \
    --patience 10 \
    --save-model \
    --model-path models/pmc_cnn.pt

EXIT_CODE=$?

echo ""
echo "======================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo ""
    echo "Model saved to: models/pmc_cnn.pt"
    echo ""
    echo "Next steps:"
    echo "  1. Check test accuracy in the output above"
    echo "  2. Compare with XGBoost: python3 train_xgboost.py"
    echo "  3. Experiment with hyperparameters (see README_CNN.md)"
else
    echo "❌ Training failed!"
    echo ""
    echo "Troubleshooting:"
    echo "  • Check GPU memory: nvidia-smi"
    echo "  • Reduce batch size: --batch-size 32"
    echo "  • Check logs above for error messages"
fi
echo "======================================================================"

