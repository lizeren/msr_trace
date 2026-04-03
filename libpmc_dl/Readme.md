## extract_events.py — subset events from JSON (CSV-driven)

Keeps only events whose `event_name` appears in the CSV (`event_name` column), re-indexes as `event_0`, `event_1`, …, and writes new `pmc_features_*.json` files under `--output`. Source folders are not modified. After that, point `preprocess_features.py`, `combo-*.sh`, and inference scripts at the new folder; no other pipeline changes.

```bash
python3 extract_events.py \
  --input 2024-5991-static-40events_mix \
  --events-csv pmc_events_deterministic.csv

# Optional explicit output path (otherwise the folder name is derived, e.g. 40events -> 10events from CSV row count)
python3 extract_events.py \
  --input 2024-5991-static-40events_mix \
  --events-csv pmc_events_deterministic.csv \
  --output 2024-5991-static-10events_mix
```

## Compute 10 statistical features brefore training
```bash
# Default usage
python3 preprocess_features.py --features "features/pmc_features_*.json"

# if function has less than 50 samples, it will be filtered out
python3 preprocess_features.py --features "features/pmc_features_*.json" --min-samples 50

# if you want to sample N files per class, you can use the following command by specifying the name of the json file
python3 preprocess_features.py     --features "features/pmc_features_*.json"     --sample-per-pattern "rsa:100,http:100,slh_dsa:100"
```

## combo.sh — O0/O3/Combined size sweep
Runs a 3×3 matrix (rows: O0, O3, Combined; columns: sample sizes) and prints test accuracy.
Patterns (patch/unpatch variants) are auto-detected from the folder.
Input argument is the folder name that contains all the jsons in patch/unpatch and differnet optimization levels.
```bash
# Default sizes 80 160 320
bash combo.sh CVE-2025-11187-static-combine-10events_mix

# Custom sizes
bash combo.sh CVE-2025-11187-shared-combine-10events_mix 50 100 200
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

## Inference
python3 inference_xgboost.py --features shared-default
python3 inference_xgboost.py --features ../CVE-2025-11187-static-combine-10events_mix