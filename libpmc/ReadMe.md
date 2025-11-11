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
./example_cache_call 1000 
```