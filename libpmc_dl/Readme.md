## Compute 10 statistical features brefore training
```bash
# Default usage
python3 preprocess_features.py --features "features/pmc_features_*.json"

# if function has less than 50 samples, it will be filtered out
python3 preprocess_features.py --features "features/pmc_features_*.json" --min-samples 50

# if you want to sample N files per pattern, you can use the following command
python3 preprocess_features.py     --features "features/pmc_features_*.json"     --sample-per-pattern "rsa:100,wolfssl:100,http:100,slh_dsa:100"
```

## Hybrid CNN (Statistical Features Only)
Uses 10 statistical features per event [38, 10]:
```bash
python3 train_hybrid_cnn.py --save-model
```

Use pre-computed 10 statistical features:
```bash
python3 train_hybrid_cnn.py --save-model --cache
```

## XGBoost on GPU (Statistical Features Only)
```bash
python3 train_xgboost_gpu.py --cache --gpu --save-model --scale-pos-weight
```

## LightGBM on GPU (Statistical Features Only)
```bash
python3 train_lightgbm_gpu.py --cache --gpu --save-model --scale-pos-weight
```

## Dual-Stream CNN (Timeseries + Statistical Features)
Uses BOTH timeseries [38, 128] AND statistical features [38, 10]:
```bash
# Keep only functions with 50+ samples
# pre-processed cache will not be used becase we also need timeseries data
python3 train_dual_stream_cnn.py --save-model --min-samples 50
```

Optional arguments:
- `--seq-len 128`: Set timeseries sequence length
- `--epochs 50`: Number of training epochs
- `--batch-size 64`: Batch size
- `--lr 1e-4`: Learning rate