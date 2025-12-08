## Hybrid Transformer (Statistical Features Only)
Uses 10 statistical features per event [38, 10]:
```bash
python3 train_hybrid.py --save-model
```

Use pre-computed 10 statistical features:
```bash
python3 train_hybrid.py --save-model --cache
```

## Dual-Stream Transformer (Timeseries + Statistical Features)
Uses BOTH timeseries [38, 128] AND statistical features [38, 10]:
```bash
python3 train_dual_stream.py --save-model
```

Optional arguments:
- `--seq-len 128`: Set timeseries sequence length
- `--epochs 50`: Number of training epochs
- `--batch-size 64`: Batch size
- `--lr 1e-4`: Learning rate