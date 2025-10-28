## release perf_event_paranoid
```bash
sudo sysctl kernel.perf_event_paranoid=1
```

## example_usage.c
Currently not working.
```bash
./example_usage 1000 sample
```

## example_l1_cache.c
Currently not working.
```bash
# HIT_SWEEPS = 200, MISS_STEPS = 200, BIG_SIZE_MB = 64, MODE = sample, SAMPLE_PERIOD = 100
./example_l1_cache 200 200 64 sample 100
```

## example_cache_call.c

```bash
./example_cache_call 1000 
```