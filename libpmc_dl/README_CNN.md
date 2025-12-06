# 1D CNN Classifier for PMC Temporal Features

A PyTorch-based 1D Convolutional Neural Network for classifying OpenSSL cryptographic functions based on their hardware Performance Monitoring Counter (PMC) signatures.

## Overview

This classifier processes raw temporal PMC event sequences to identify which of 31 OpenSSL functions is executing. Unlike traditional feature engineering approaches, the CNN learns patterns directly from event rate time series.

### Key Features

- **Direct temporal learning**: Processes raw timestamp sequences converted to event rates
- **Per-event normalization**: Each of 38 hardware events normalized independently
- **GPU acceleration**: Optimized for NVIDIA GPUs (tested on RTX 4090)
- **Early stopping**: Automatic training termination when validation performance plateaus
- **Stratified splitting**: Balanced train/val/test split preserving class distributions

### Model Architecture

```
Input: [batch_size, 38 events, L timesteps]
  ↓
Conv1D(38→64, k=7) + BatchNorm + ReLU + MaxPool
  ↓
Conv1D(64→128, k=5) + BatchNorm + ReLU + MaxPool
  ↓
Conv1D(128→256, k=3) + BatchNorm + ReLU + MaxPool
  ↓
Conv1D(256→512, k=3) + BatchNorm + ReLU
  ↓
Global Average Pooling
  ↓
FC(512→256) + Dropout + ReLU
  ↓
FC(256→31)
  ↓
Output: [batch_size, 31 classes]
```

## Installation

### Requirements

- **Python**: 3.8 or higher
- **PyTorch**: 2.0+ with CUDA support (for GPU acceleration)
- **GPU**: NVIDIA GPU with CUDA support (recommended: RTX 4090)
- **RAM**: 16GB+ recommended for large datasets

### Step 1: Install PyTorch with CUDA Support

For RTX 4090 (CUDA 11.8 or 12.1):

```bash
# CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# OR CUDA 12.1 (newer, faster)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA installation:

```bash
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
CUDA available: True
CUDA version: 12.1
GPU: NVIDIA GeForce RTX 4090
```

### Step 2: Install Additional Dependencies

```bash
pip3 install numpy scikit-learn
```

Or use the provided requirements file:

```bash
pip3 install -r requirements.txt
```

### Step 3: Verify Installation

```bash
cd libpmc_ml
python3 train_cnn.py --help
```

## Data Preprocessing

### Input Data Format

The classifier expects JSON files in the `features/` directory:

```
libpmc_ml/
├── features/
│   ├── pmc_features_1.json
│   ├── pmc_features_2.json
│   ├── ...
│   └── pmc_features_N.json
├── train_cnn.py
└── README_CNN.md
```

Each JSON file contains multiple runs (samples):

```json
{
  "EVP_PKEY_generate": {
    "event_0": {
      "event_name": "BR_MISP_RETIRED.ALL_BRANCHES",
      "sampling_period": 100,
      "timestamps_ns": [0, 30158, 70314, ...],
      ...
    },
    "event_1": { ... },
    ...
    "event_37": { ... }
  },
  "RSA_new": { ... },
  ...
}
```

### Preprocessing Pipeline

For each run and each event:

1. **Extract timestamps**: `[t₀, t₁, ..., tₙ]`

2. **Compute time differences**:
   ```
   Δt[0] = t₀
   Δt[j] = t[j] - t[j-1]  for j ≥ 1
   ```

3. **Convert to event rates** (events per nanosecond):
   ```
   rate[j] = sampling_period / Δt[j]
   ```
   
4. **Apply log transformation** for stability:
   ```
   rate_log[j] = log(1 + rate[j])
   ```

5. **Pad or truncate** to fixed length L (default: 128)

6. **Normalize per-event** using training statistics:
   ```
   x'[e,t] = (x[e,t] - μₑ) / (σₑ + ε)
   ```

Result: Each sample becomes a `[38, L]` tensor.

## Usage

### Basic Training

Train with default settings:

```bash
cd libpmc_ml
python3 train_cnn.py --features "features/pmc_features_*.json"
```

### Custom Configuration

```bash
python3 train_cnn.py \
    --features "features/pmc_features_*.json" \
    --seq-len 256 \
    --batch-size 32 \
    --epochs 50 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --dropout 0.3 \
    --patience 10 \
    --save-model \
    --model-path models/pmc_cnn_best.pt
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--features` | `features/pmc_features_*.json` | Glob pattern for input JSON files |
| `--seq-len` | `128` | Fixed sequence length L |
| `--batch-size` | `64` | Batch size for training |
| `--epochs` | `50` | Maximum number of epochs |
| `--lr` | `1e-3` | Learning rate |
| `--weight-decay` | `1e-4` | AdamW weight decay |
| `--dropout` | `0.3` | Dropout rate |
| `--patience` | `10` | Early stopping patience |
| `--save-model` | `False` | Save trained model |
| `--model-path` | `models/pmc_cnn.pt` | Path to save model |

## Training on RTX 4090

### GPU Utilization

The code automatically detects and uses CUDA if available:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

### Performance Tips

1. **Batch Size**: RTX 4090 has 24GB VRAM, so you can use larger batches:
   ```bash
   python3 train_cnn.py --batch-size 128
   ```

2. **Mixed Precision Training** (optional, for faster training):
   Add this to the code for ~2x speedup:
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   ```

3. **Data Loading**: Use multiple workers:
   ```python
   DataLoader(..., num_workers=4)  # Already set in code
   ```

4. **Monitor GPU Usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

### Expected Training Time

On RTX 4090 with 12,400 samples:
- **Per epoch**: ~10-15 seconds
- **Total (50 epochs)**: ~10-15 minutes
- **With early stopping**: ~5-8 minutes (typically stops around epoch 20-30)

## Output

### Training Progress

```
============================================================
PMC 1D CNN Classifier
============================================================
Device: cuda
GPU: NVIDIA GeForce RTX 4090
CUDA Version: 12.1
============================================================

Loading PMC Features
============================================================
Pattern: features/pmc_features_*.json
Found 1200 files
Sequence length: 128

============================================================
Loaded 12400 samples total
Unique classes: 31
============================================================

Dataset Statistics:
  Total samples: 12400
  Number of classes: 31
  Samples per class: min=400, max=400, mean=400.0

============================================================
Data Split
============================================================
Train: 8680 samples (70.0%)
Val:   1860 samples (15.0%)
Test:  1860 samples (15.0%)

Computing per-event normalization statistics...
  Event 0: mean=0.0234, std=0.1234
  Event 1: mean=0.0456, std=0.2345
  ...

============================================================
Model Architecture
============================================================
Input shape: [batch_size, 38, 128]
Output: 31 classes
Total parameters: 1,234,567
Trainable parameters: 1,234,567

============================================================
Training
============================================================
Epochs: 50
Batch size: 64
Learning rate: 0.001
Weight decay: 0.0001
Dropout: 0.3
Early stopping patience: 10
============================================================

Epoch [  1/ 50] Train Loss: 2.1234 | Train Acc: 35.67% | Val Loss: 1.8765 | Val Acc: 42.34% | Time: 12.3s
Epoch [  2/ 50] Train Loss: 1.5432 | Train Acc: 56.78% | Val Loss: 1.4321 | Val Acc: 58.92% | Time: 11.8s
Epoch [  3/ 50] Train Loss: 1.2345 | Train Acc: 68.45% | Val Loss: 1.1234 | Val Acc: 70.12% | Time: 12.1s
...
Epoch [ 25/ 50] Train Loss: 0.2345 | Train Acc: 95.67% | Val Loss: 0.4567 | Val Acc: 88.92% | Time: 11.9s

Early stopping triggered at epoch 25

============================================================
Training Complete
============================================================
Total time: 298.5s (5.0 minutes)
Best validation accuracy: 88.92%

Loading best model from models/pmc_cnn.pt

============================================================
Test Set Evaluation
============================================================
Test Accuracy: 89.23%

Classification Report:
                              precision    recall  f1-score   support

              BN_bin2bn       0.9234    0.9100    0.9166        60
          EVP_DigestSign       0.8765    0.9000    0.8881        60
      EVP_DigestSignInit       0.9100    0.8833    0.8965        60
            EVP_MD_CTX_new     0.8950    0.9167    0.9057        60
       EVP_PKEY_CTX_free       0.9200    0.8833    0.9013        60
...
                   accuracy                        0.8923      1860
                  macro avg     0.8934    0.8923    0.8927      1860
               weighted avg     0.8934    0.8923    0.8927      1860

============================================================
Model saved to: models/pmc_cnn.pt
============================================================

Done!
```

### Saved Model Contents

The saved model includes:

```python
{
    'epoch': 25,
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'val_acc': 88.92,
    'label_encoder': ...,      # For class name mapping
    'event_stats': ...,        # Normalization parameters
    'seq_len': 128,            # Sequence length
}
```

## Comparison with Other Models

### Expected Performance

Based on your dataset (12,400 samples, 31 classes, 38 events):

| Model | Accuracy | Training Time | Feature Engineering |
|-------|----------|---------------|---------------------|
| **Logistic Regression** | ~78-85% | < 1 min | Manual (10 stats/event) |
| **XGBoost** | ~85-92% | ~2-3 min | Manual (10 stats/event) |
| **1D CNN** | **~85-93%** | ~5-10 min | **Automatic** |

### Advantages of 1D CNN

✅ **Automatic feature learning**: No manual feature engineering
✅ **Temporal patterns**: Learns from raw sequences
✅ **End-to-end**: Direct timestamp → classification
✅ **Scalable**: Can handle longer sequences if needed

### When to Use Each Model

- **XGBoost**: Best for quick experiments, interpretable features
- **1D CNN**: Best for discovering unknown temporal patterns, research applications
- **Both**: Train both and ensemble for maximum accuracy

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
python3 train_cnn.py --batch-size 32
```

### Slow Training

1. Check GPU is being used:
   ```bash
   nvidia-smi
   ```

2. Reduce sequence length:
   ```bash
   python3 train_cnn.py --seq-len 64
   ```

### Poor Accuracy

1. Increase model capacity:
   - Longer sequences: `--seq-len 256`
   - More epochs: `--epochs 100`
   - Lower dropout: `--dropout 0.2`

2. Check data quality:
   - Ensure all 38 events present
   - Verify timestamp consistency
   - Check class balance

### Installation Issues

**PyTorch not detecting GPU**:
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip3 uninstall torch torchvision torchaudio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Files

- `train_cnn.py`: Main training script
- `requirements.txt`: Python dependencies
- `README_CNN.md`: This file
- `models/`: Saved model checkpoints (created after `--save-model`)

## Citation

If you use this code in your research, please cite:

```
PMC-based Cryptographic Function Classification using 1D CNN
Hardware Performance Counter Analysis for Security Applications
```

## License

This code is part of the PMC temporal feature classification project.

---

**Questions or Issues?**

Check the troubleshooting section or refer to the PyTorch documentation:
- PyTorch Installation: https://pytorch.org/get-started/locally/
- CUDA Setup: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/

