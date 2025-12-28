## Hybrid Transformer (Statistical Features Only)
Uses 16 statistical features per event [38, 16]:
- **Stats (1-6):** total_count_mean, total_count_std, duration_mean_ns, duration_std_ns, num_samples_mean, num_samples_std
- **Temporal (7-16):** total_duration, mean_interval, std_interval, min_interval, max_interval, sample_rate, num_samples, q25, q50, q75

```bash
python3 train_hybrid.py --save-model
```

Use pre-computed 16 statistical features:
```bash
python3 train_hybrid.py --save-model --cache
```

## Dual-Stream Transformer (Timeseries + Statistical Features)
Uses BOTH timeseries [38, 128] AND statistical features [38, 16]:
```bash
python3 train_dual_stream.py --save-model --cache
```

**Note:** First run with `--cache` will load JSON files once to create the dual-stream cache.
Subsequent runs will load instantly from cache (`dual_stream_cache_seq128.pkl`). This cache is different from the cache from libpmc_dl/features_16/features_cache.pkl.

Optional arguments:
- `--seq-len 128`: Set timeseries sequence length
- `--epochs 50`: Number of training epochs
- `--batch-size 64`: Batch size
- `--lr 1e-4`: Learning rate
- `--cache`: Use/create cached features in ../libpmc_dl/features_16/
