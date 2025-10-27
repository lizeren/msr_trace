# Multi-Event Sampling Guide

This guide explains how to sample multiple performance events simultaneously with different sample periods using the PMC library.

## Overview

The PMC library allows you to create multiple PMC contexts measuring different events concurrently. Each context can have its own configuration, including different sample periods.

## Key Concept

**Run the workload ONCE** while multiple PMC contexts measure simultaneously. This ensures:
1. All events are measured on the same execution
2. Timestamps are comparable across events
3. Lower overhead than running multiple separate measurements

## Basic Pattern

```c
// Create multiple PMC contexts with different events and periods
pmc_config_t config1 = pmc_get_default_config(PMC_EVENT_NEAR_CALL, PMC_MODE_SAMPLING, 100);
pmc_ctx_t *pmc1 = pmc_create(&config1);

pmc_config_t config2 = pmc_get_default_config(PMC_EVENT_L1_DCACHE_HIT, PMC_MODE_SAMPLING, 1000);
config2.precise_ip = 0;  // Cache events may need this
pmc_ctx_t *pmc2 = pmc_create(&config2);

// Start both measurements
pmc_start(pmc1);
pmc_start(pmc2);

// Run workload ONCE - both PMCs measure simultaneously
workload();

// Stop both measurements
pmc_stop(pmc1);
pmc_stop(pmc2);

// Read samples from both
pmc_sample_t *samples1, *samples2;
size_t num_samples1, num_samples2;
pmc_read_samples(pmc1, &samples1, &num_samples1, 0);
pmc_read_samples(pmc2, &samples2, &num_samples2, 0);

// Analyze results...
free(samples1);
free(samples2);
pmc_destroy(pmc1);
pmc_destroy(pmc2);
```

## Complete Example: `example_cache_call.c`

This example measures **near calls** and **L1 cache hits** simultaneously with different sample periods.

### Build

```bash
gcc -O2 -g -Wall -Wextra example_cache_call.c -L. -lpmc -ldl -Wl,-rpath,'$ORIGIN' -o example_cache_call
```

### Usage

```bash
# Counting mode (both events)
./example_cache_call 100000

# Sampling mode with default periods (100 for calls, 1000 for cache)
./example_cache_call 100000 sample

# Sampling mode with custom periods
./example_cache_call 100000 sample 50 500
#                     ^       ^      ^  ^
#                     |       |      |  cache hit period (500)
#                     |       |      near call period (50)
#                     |       sampling mode
#                     iterations
```

### Example Output

```
=== Multi-Event PMC Example ===
Workload: 50000 iterations
Mode: SAMPLING
  Near call sample period: 50
  Cache hit sample period: 500

Measuring events:
  1. BR_INST_RETIRED.NEAR_CALL (period: 50)
  2. MEM_LOAD_RETIRED.L1_HIT (period: 500)

Results:
========================================

--- Near Call Samples ---
Total samples collected: 203

First 5 near call samples:
SAMPLE #1  pid=721 tid=721 time=29953511668628
  ip=0x557dd096573b  (no symbol)
...

Total near calls: 10150
Calls per sample: 50.0

--- L1 Cache Hit Samples ---
Total samples collected: 203

First 5 cache hit samples:
SAMPLE #1  pid=721 tid=721 time=29953511686237
  ip=0x557dd0965746  (no symbol)
...

Total cache hits: 101504
Hits per sample: 500.0
```

## Understanding Sample Periods

### Why Different Periods?

Different events occur at different rates:
- **Near calls**: ~3 per iteration (lower frequency)
- **Cache hits**: ~7 per iteration (higher frequency)

Using the same sample period for both would give:
- Too few samples for low-frequency events
- Too many samples for high-frequency events (overhead!)

### Choosing Sample Periods

**General Rule**: Aim for 100-1000 samples per execution

```c
// Low-frequency event: smaller period
config_calls.sample_period = 100;      // Get ~300 samples for 30K calls

// High-frequency event: larger period  
config_cache.sample_period = 1000;     // Get ~700 samples for 700K hits
```

### Trade-offs

| Period | Samples | Detail | Overhead |
|--------|---------|--------|----------|
| Small (10-100) | Many | High detail | Higher overhead |
| Medium (100-1000) | Moderate | Good detail | Low overhead |
| Large (1000+) | Few | Less detail | Very low overhead |

## Supported Event Combinations

You can combine any events supported by your CPU:

```c
// Branch events
PMC_EVENT_NEAR_CALL
PMC_EVENT_CONDITIONAL_BRANCH
PMC_EVENT_BRANCH_MISPREDICT

// Cache events (set precise_ip=0)
PMC_EVENT_L1_DCACHE_MISS
PMC_EVENT_L1_DCACHE_HIT

// General events
PMC_EVENT_CYCLES
PMC_EVENT_INSTRUCTIONS
```

### Example: Three Events

```c
// Measure calls, cache hits, and mispredicts simultaneously
pmc_config_t cfg1 = pmc_get_default_config(PMC_EVENT_NEAR_CALL, PMC_MODE_SAMPLING, 100);
pmc_config_t cfg2 = pmc_get_default_config(PMC_EVENT_L1_DCACHE_HIT, PMC_MODE_SAMPLING, 1000);
pmc_config_t cfg3 = pmc_get_default_config(PMC_EVENT_BRANCH_MISPREDICT, PMC_MODE_SAMPLING, 50);

cfg2.precise_ip = 0;  // Cache events
cfg3.precise_ip = 0;  // Mispredict events

pmc_ctx_t *pmc1 = pmc_create(&cfg1);
pmc_ctx_t *pmc2 = pmc_create(&cfg2);
pmc_ctx_t *pmc3 = pmc_create(&cfg3);

// Start all three
pmc_start(pmc1);
pmc_start(pmc2);
pmc_start(pmc3);

// Run workload once
workload();

// Stop all three
pmc_stop(pmc1);
pmc_stop(pmc2);
pmc_stop(pmc3);

// Read and analyze samples from all three...
```

## Correlating Samples

Since all events measure the same execution, you can correlate samples by:

### 1. Timestamp

```c
// Find cache hits that occurred during near call samples
for (size_t i = 0; i < num_call_samples; i++) {
    uint64_t call_time = call_samples[i].time;
    
    // Find cache samples within ±1ms
    for (size_t j = 0; j < num_cache_samples; j++) {
        uint64_t cache_time = cache_samples[j].time;
        if (abs((long long)(cache_time - call_time)) < 1000000) {
            printf("Correlated: call at %llu, cache at %llu\n",
                   call_time, cache_time);
        }
    }
}
```

### 2. Instruction Pointer (IP)

```c
// Find hot code regions with both calls and cache misses
for (size_t i = 0; i < num_call_samples; i++) {
    uint64_t call_ip = call_samples[i].ip;
    
    for (size_t j = 0; j < num_miss_samples; j++) {
        uint64_t miss_ip = miss_samples[j].ip;
        
        // Within same function (rough check: within 4KB)
        if (abs((long long)(call_ip - miss_ip)) < 4096) {
            printf("Hot region: 0x%llx has both calls and misses\n", call_ip);
        }
    }
}
```

## Best Practices

### 1. Check for PMC Creation Errors

```c
pmc_ctx_t *pmc1 = pmc_create(&config1);
if (!pmc1) {
    fprintf(stderr, "Failed to create PMC: %s\n", pmc_get_error());
    // Decide: fail or continue without measurement
}
```

### 2. Set `precise_ip=0` for Cache Events

```c
pmc_config_t config = pmc_get_default_config(PMC_EVENT_L1_DCACHE_HIT, mode, period);
config.precise_ip = 0;  // Required for many CPUs
```

### 3. Clean Up Properly

```c
// Free samples
if (samples1) free(samples1);
if (samples2) free(samples2);

// Destroy contexts
pmc_destroy(pmc1);
pmc_destroy(pmc2);
```

### 4. Limit Number of Concurrent PMCs

Most CPUs support 4-8 hardware counters. Exceeding this causes:
- Time multiplexing (context switching between counters)
- Less accurate measurements
- Potential overhead

**Recommendation**: Measure 2-4 events simultaneously.

## Performance Impact

### Overhead Analysis

| Configuration | Overhead |
|---------------|----------|
| 1 event, count mode | < 1% |
| 1 event, sampling (period 1000) | < 2% |
| 2 events, sampling (periods 100, 1000) | < 3% |
| 4 events, sampling (various periods) | 3-5% |

### Reducing Overhead

1. **Increase sample period**: Fewer samples = lower overhead
2. **Measure fewer events**: Each active PMC adds overhead
3. **Use counting mode** when samples aren't needed

## Debugging Tips

### No Samples Collected

```c
if (num_samples == 0) {
    // Possible causes:
    // 1. Sample period too large
    // 2. Workload too short
    // 3. Event doesn't occur in workload
    // 4. CPU doesn't support PEBS for this event
}
```

**Solution**: Lower sample period or increase workload size

### Symbol Resolution Fails

```bash
# Build with debug symbols
gcc -O2 -g your_program.c -lpmc -ldl

# Verify symbols exist
nm your_program | grep your_function
```

### Different CPUs, Different Support

Some events (especially cache events) vary by CPU:
- Intel: MEM_LOAD_RETIRED.L1_MISS (0xD1:0x08)
- AMD: May use different codes
- ARM: Different architecture entirely

**Check**: `perf list | grep -i cache`

## See Also

- `ReadMe.md` - Main library documentation
- `INTEGRATION_GUIDE.md` - Integration examples
- `L1_CACHE_GUIDE.md` - L1 cache specific guide
- `example_cache_call.c` - Full source code
- `example_l1_cache.c` - Single-event sampling example

