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

## How to Enable Debug Mode

Set the `PMC_DEBUG` environment variable to `1`:

```bash
# Enable debug mode
export PMC_DEBUG=1
./your_program

# Or as a one-liner
PMC_DEBUG=1 ./your_program
```

### python collector

```bash
python3 collect_pmc_features.py --target "./example_cache_call" --runs 5 --total 10 --name example_cache_call --start 1 > result.log
```

### file explanation

pmc_events.csv : sample period is longer than pmc_events_chill.csv
collect_pmc_features.py : it will record events even if they don't have any samples, which is unlike collect_pmc_features_old.py


