### release perf_event_paranoid
```bash
sudo sysctl kernel.perf_event_paranoid=1
```

### build the library
```bash
make
```

### example_cache_call.c

```bash
# specify what you want to measure out of pmc_events.csv
PMC_EVENT_INDICES="0,1,2,3" ./example_cache_call 1000 
```

when using as libpmc

```bash
export PMC_EVENT_INDICES="0,1,2,3"
```

### python collector

```bash
python3 collect_pmc_features.py --target "./example_cache_call" --runs 5 --total 10 --name example_cache_call --start 1 > result.log

```