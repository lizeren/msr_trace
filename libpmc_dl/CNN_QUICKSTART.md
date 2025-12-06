# 1D CNN Quick Reference Card

## 🚀 Installation (RTX 4090 Host)

```bash
# Install PyTorch with CUDA 12.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip3 install -r requirements_cnn.txt

# Verify GPU
python3 -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"
```

## 🎯 Training

### Quick Start
```bash
python3 train_cnn.py --features "features/pmc_features_*.json" --save-model
```

### Custom Configuration
```bash
python3 train_cnn.py \
    --features "features/pmc_features_*.json" \
    --seq-len 128 \
    --batch-size 64 \
    --epochs 50 \
    --lr 1e-3 \
    --patience 10 \
    --save-model
```

### Or Use Convenience Script
```bash
chmod +x run_cnn.sh
./run_cnn.sh
```

## 📊 Data Format

**Input**: JSON files with PMC temporal data
```json
{
  "function_name": {
    "event_0": {
      "timestamps_ns": [0, 30158, 70314, ...],
      "sampling_period": 100
    },
    ...
    "event_37": { ... }
  }
}
```

**Preprocessing**: timestamps → time diffs → rates → log → pad/truncate → normalize

**CNN Input**: `[batch, 38 events, 128 timesteps]`

## 🏗️ Model

```
Conv1D(38→64→128→256→512) + BatchNorm + ReLU + MaxPool
Global Average Pooling
FC(512→256→31)
```

**Parameters**: ~1.2M

## ⚙️ Hyperparameters

| Parameter | Default | Range |
|-----------|---------|-------|
| Sequence length | 128 | 64-256 |
| Batch size | 64 | 32-128 |
| Learning rate | 1e-3 | 1e-4 to 1e-2 |
| Dropout | 0.3 | 0.2-0.5 |
| Epochs | 50 | 30-100 |
| Patience | 10 | 5-20 |

## 📈 Expected Results

**Dataset**: 12,400 samples, 31 classes

**Performance**:
- Test accuracy: **85-93%**
- Training time: **5-10 minutes** (RTX 4090)
- Time per epoch: **~10-15 seconds**

**Comparison**:
- Logistic Regression: 78-85%
- XGBoost: 85-92%
- **1D CNN: 85-93%** ⭐

## 🔧 Troubleshooting

### CUDA Out of Memory
```bash
python3 train_cnn.py --batch-size 32
```

### Slow Training
```bash
nvidia-smi  # Check GPU usage
```

### Poor Accuracy
```bash
# Try longer sequences
python3 train_cnn.py --seq-len 256

# Try less regularization
python3 train_cnn.py --dropout 0.2
```

## 📁 Files

- `train_cnn.py` - Training script (858 lines)
- `README_CNN.md` - Full documentation
- `requirements_cnn.txt` - Dependencies
- `run_cnn.sh` - Convenience script
- `CNN_IMPLEMENTATION.md` - Implementation summary

## 💡 Tips

1. **Use larger batches** on RTX 4090: `--batch-size 128`
2. **Monitor GPU**: `watch -n 1 nvidia-smi`
3. **Compare with XGBoost**: `python3 train_xgboost.py`
4. **Save models**: Always use `--save-model`
5. **Experiment**: Try different `--seq-len` values

## ✅ Checklist

- [ ] PyTorch + CUDA installed
- [ ] GPU detected (`torch.cuda.is_available() == True`)
- [ ] Feature files in `features/` directory
- [ ] Training runs without errors
- [ ] Model saved to `models/pmc_cnn.pt`
- [ ] Test accuracy > 85%

---

**Ready to train!** 🚀

For full documentation, see `README_CNN.md`

