## example_usage.c

```bash
./example_usage 1000 sample
```

## example_l1_cache.c

```bash
# HIT_SWEEPS = 200, MISS_STEPS = 200, BIG_SIZE_MB = 64, MODE = sample, SAMPLE_PERIOD = 100
./example_l1_cache 200 200 64 sample 100
```

## example_cache_call.c

```bash
# ITERATIONS = 50000, MODE = sample, NEAR_CALL_PERIOD = 50, CACHE_PERIOD = 500
./example_cache_call 50000 sample 50 500
```