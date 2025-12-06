# 1D CNN Implementation Summary

## ✅ Implementation Complete

All files have been created and are ready to migrate to your RTX 4090 host.

---

## 📁 Files Created

### 1. **`train_cnn.py`** (858 lines)
Main training script with complete pipeline:

**Data Loading & Preprocessing:**
- Loads all `pmc_features_*.json` files
- Converts timestamps → time differences → event rates
- Applies `log1p(rate)` for numerical stability
- Pads/truncates sequences to fixed length L (default: 128)
- Creates tensors: `[38 events, L timesteps]`

**Normalization:**
- Per-event (per-channel) normalization
- Computed on training set only
- Applied to train/val/test consistently
- Formula: `(x - μₑ) / (σₑ + ε)` for each event e

**Model Architecture:**
```
Input: [batch, 38, L]
Conv1D(38→64, k=7) + BN + ReLU + MaxPool(2)
Conv1D(64→128, k=5) + BN + ReLU + MaxPool(2)
Conv1D(128→256, k=3) + BN + ReLU + MaxPool(2)
Conv1D(256→512, k=3) + BN + ReLU
Global Average Pooling
FC(512→256) + Dropout(0.3) + ReLU
FC(256→31)
Output: [batch, 31 classes]
```

**Training Features:**
- Stratified 70/15/15 train/val/test split
- AdamW optimizer (lr=1e-3, weight_decay=1e-4)
- CrossEntropyLoss
- Early stopping (patience=10)
- Automatic GPU detection and usage
- Model checkpointing (saves best validation model)

**Evaluation:**
- Per-epoch train/val accuracy
- Final test accuracy
- Full classification report (precision/recall/F1 per class)
- Confusion matrix

### 2. **`README_CNN.md`** (comprehensive documentation)
Complete guide covering:

**Installation:**
- PyTorch with CUDA 12.1 for RTX 4090
- Step-by-step instructions
- GPU verification commands

**Usage:**
- Basic training examples
- All command-line arguments explained
- Hyperparameter tuning guide

**RTX 4090 Optimization:**
- Batch size recommendations (64-128)
- Expected training time (~10-15s per epoch)
- GPU utilization tips
- Memory optimization

**Troubleshooting:**
- CUDA out of memory solutions
- GPU not detected fixes
- Performance tuning tips

### 3. **`requirements_cnn.txt`**
Python dependencies:
```
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
numpy>=1.21.0
scikit-learn>=1.0.0
```

### 4. **`run_cnn.sh`** (convenience script)
Automated training script:
- Checks GPU availability
- Verifies data files exist
- Runs training with sensible defaults
- Reports success/failure

---

## 🚀 Quick Start on RTX 4090 Host

### Step 1: Install PyTorch with CUDA
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 2: Install Dependencies
```bash
cd libpmc_ml
pip3 install -r requirements_cnn.txt
```

### Step 3: Verify GPU
```bash
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
CUDA: True
GPU: NVIDIA GeForce RTX 4090
```

### Step 4: Train!

**Option A: Use the convenience script**
```bash
chmod +x run_cnn.sh
./run_cnn.sh
```

**Option B: Run directly**
```bash
python3 train_cnn.py --features "features/pmc_features_*.json" --save-model
```

---

## 📊 Data Flow

### Input Format
Each JSON file contains multiple runs (samples):
```json
{
  "EVP_PKEY_generate": {
    "event_0": {
      "event_name": "BR_MISP_RETIRED.ALL_BRANCHES",
      "sampling_period": 100,
      "timestamps_ns": [0, 30158, 70314, 100162, ...],
      "num_samples": 150
    },
    "event_1": { ... },
    ...
    "event_37": { ... }
  },
  "RSA_new": { ... },
  ...
}
```

### Preprocessing Steps

**For each event in each run:**

1. **Extract timestamps**: `[t₀, t₁, t₂, ..., tₙ]`

2. **Compute time differences**:
   ```
   Δt[0] = t₀
   Δt[j] = t[j] - t[j-1]  for j ≥ 1
   ```

3. **Convert to event rates** (events per nanosecond):
   ```
   rate[j] = sampling_period / max(Δt[j], ε)
   ```
   where ε = 1e-9 to avoid division by zero

4. **Apply log transformation**:
   ```
   rate_log[j] = log(1 + rate[j])
   ```

5. **Pad or truncate** to fixed length L:
   - If sequence has < L values: pad with zeros
   - If sequence has > L values: truncate to first L

6. **Stack events**: Create `[38, L]` tensor

7. **Normalize per-event**:
   ```
   x'[e, t] = (x[e, t] - μₑ) / (σₑ + ε)
   ```
   where μₑ, σₑ are computed from training set for event e

### Output Format
- **Input to CNN**: `[batch_size, 38, L]` tensor
- **Labels**: Integer class IDs (0-30) for 31 OpenSSL functions
- **Output**: Logits `[batch_size, 31]` → Softmax → Class probabilities

---

## ⚙️ Hyperparameters

### Model
- **Sequence length**: 128 (adjustable via `--seq-len`)
- **Dropout**: 0.3
- **Architecture**: 4 conv layers (64→128→256→512 channels)

### Training
- **Batch size**: 64 (increase to 128 for RTX 4090)
- **Learning rate**: 1e-3
- **Weight decay**: 1e-4
- **Optimizer**: AdamW
- **Epochs**: 50 (with early stopping)
- **Early stopping patience**: 10 epochs

### Data Split
- **Train**: 70% (stratified)
- **Validation**: 15% (stratified)
- **Test**: 15% (stratified)

---

## 📈 Expected Performance

### On Your Dataset
- **Samples**: 12,400 (400 per class)
- **Classes**: 31 OpenSSL functions
- **Events**: 38 PMC hardware events
- **Features**: Raw temporal sequences (no manual engineering)

### Training Time (RTX 4090)
- **Per epoch**: ~10-15 seconds
- **Total**: ~5-10 minutes (with early stopping around epoch 20-30)
- **Full 50 epochs**: ~10-15 minutes

### Expected Accuracy
- **Test accuracy**: **85-93%**
- **Validation accuracy**: Similar to test
- **Comparison**:
  - Logistic Regression: ~78-85%
  - XGBoost: ~85-92%
  - **1D CNN: ~85-93%** ⭐

---

## 🎯 Advantages of 1D CNN

### vs. XGBoost
✅ **Automatic feature learning** - No need to manually design statistical features
✅ **Temporal pattern discovery** - Learns from raw event rate sequences
✅ **End-to-end trainable** - Direct optimization for classification
✅ **Potentially higher ceiling** - Can discover complex patterns

### vs. Statistical Features
✅ **No information loss** - Uses full temporal sequence
✅ **Adaptive** - Model decides what patterns matter
✅ **Scalable** - Can handle longer sequences easily

---

## 🔧 Customization

### Experiment with Different Configurations

**Longer sequences (more temporal detail):**
```bash
python3 train_cnn.py --seq-len 256
```

**Larger batch size (faster training on RTX 4090):**
```bash
python3 train_cnn.py --batch-size 128
```

**More aggressive regularization:**
```bash
python3 train_cnn.py --dropout 0.5 --weight-decay 1e-3
```

**Lower learning rate (more stable):**
```bash
python3 train_cnn.py --lr 5e-4
```

### Architecture Modifications

Edit `train_cnn.py` to change model architecture:
- Add more conv layers
- Change channel dimensions
- Add residual connections
- Try different pooling strategies

---

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
python3 train_cnn.py --batch-size 32  # Reduce batch size
python3 train_cnn.py --seq-len 64     # Reduce sequence length
```

### Training Too Slow
```bash
nvidia-smi  # Check if GPU is being used
```

If GPU usage is low:
- Increase batch size: `--batch-size 128`
- Use more workers: Edit `train_cnn.py`, set `num_workers=8`

### Poor Accuracy
1. **Longer sequences**: `--seq-len 256` (capture more temporal info)
2. **Less regularization**: `--dropout 0.2` (if underfitting)
3. **More epochs**: `--epochs 100 --patience 20`
4. **Check data**: Verify all 38 events present, timestamps consistent

---

## 📝 Next Steps

### Immediate
1. ✅ Copy all files to RTX 4090 host
2. ✅ Install PyTorch + CUDA
3. ✅ Run `./run_cnn.sh` or train manually
4. ✅ Compare accuracy with XGBoost

### Future Enhancements
- **Data augmentation**: Time stretching, noise injection
- **Ensemble**: Combine CNN + XGBoost predictions
- **Visualization**: GradCAM to see which temporal patterns matter
- **Transfer learning**: Pre-train on related tasks
- **Architecture search**: Try deeper/wider networks

---

## 📚 Files Reference

| File | Lines | Purpose |
|------|-------|---------|
| `train_cnn.py` | 858 | Main training script |
| `README_CNN.md` | 450+ | Comprehensive documentation |
| `requirements_cnn.txt` | 14 | Python dependencies |
| `run_cnn.sh` | 80 | Convenience training script |
| `CNN_IMPLEMENTATION.md` | This file | Implementation summary |

---

## ✅ Ready to Train!

Everything is implemented and documented. Simply:

1. Copy `libpmc_ml/` folder to your RTX 4090 host
2. Follow the Quick Start guide above
3. Train and evaluate!

**Good luck with your experiments! 🚀**

