# L1 Data Cache Event Guide

Guide for measuring L1 data cache hits and misses using the PMC library.

## Quick Reference

### Events Available

| Event | Description | Intel Event Code |
|-------|-------------|------------------|
| `PMC_EVENT_L1_DCACHE_MISS` | L1 data cache load misses | MEM_LOAD_RETIRED.L1_MISS (0xD1:0x08) |
| `PMC_EVENT_L1_DCACHE_HIT` | L1 data cache load hits | MEM_LOAD_RETIRED.L1_HIT (0xD1:0x01) |

## Basic Usage

### Count L1 Cache Misses

```c
#include "pmc.h"

pmc_config_t config = pmc_get_default_config(PMC_EVENT_L1_DCACHE_MISS, PMC_MODE_COUNTING);
pmc_ctx_t *pmc = pmc_create(&config);

pmc_start(pmc);
// Your memory-intensive code here
pmc_stop(pmc);

uint64_t misses;
pmc_read_count(pmc, &misses);
printf("L1 D-cache misses: %llu\n", misses);
pmc_destroy(pmc);
```

### Count L1 Cache Hits

```c
pmc_config_t config = pmc_get_default_config(PMC_EVENT_L1_DCACHE_HIT, PMC_MODE_COUNTING);
pmc_ctx_t *pmc = pmc_create(&config);

pmc_start(pmc);
// Your memory-intensive code here
pmc_stop(pmc);

uint64_t hits;
pmc_read_count(pmc, &hits);
printf("L1 D-cache hits: %llu\n", hits);
pmc_destroy(pmc);
```

### Calculate Miss Rate

```c
uint64_t measure_miss_rate(void (*workload)(void)) {
    uint64_t misses = 0, hits = 0;
    
    // Measure misses
    pmc_config_t config_miss = pmc_get_default_config(
        PMC_EVENT_L1_DCACHE_MISS, 
        PMC_MODE_COUNTING
    );
    pmc_ctx_t *pmc_miss = pmc_create(&config_miss);
    pmc_start(pmc_miss);
    workload();
    pmc_stop(pmc_miss);
    pmc_read_count(pmc_miss, &misses);
    pmc_destroy(pmc_miss);
    
    // Measure hits
    pmc_config_t config_hit = pmc_get_default_config(
        PMC_EVENT_L1_DCACHE_HIT, 
        PMC_MODE_COUNTING
    );
    pmc_ctx_t *pmc_hit = pmc_create(&config_hit);
    pmc_start(pmc_hit);
    workload();
    pmc_stop(pmc_hit);
    pmc_read_count(pmc_hit, &hits);
    pmc_destroy(pmc_hit);
    
    // Calculate miss rate
    uint64_t total = misses + hits;
    double miss_rate = total > 0 ? (double)misses / total * 100.0 : 0.0;
    
    printf("L1 Misses: %llu, Hits: %llu, Total: %llu, Miss Rate: %.2f%%\n",
           misses, hits, total, miss_rate);
    
    return misses;
}
```

## Example Program

A complete example is provided in `example_l1_cache.c` that demonstrates:

- Sequential access patterns (cache-friendly)
- Strided access patterns (variable cache behavior)
- Random access patterns (cache-unfriendly)
- Comparison of small vs large arrays
- Miss rate calculation

### Build and Run

```bash
# Build the example
make example_l1_cache

# Run it
./example_l1_cache
```

### Expected Output

```
=== L1 Data Cache Behavior Analysis ===
Using PMC Library to measure cache hits and misses

PMC initialized successfully

Test Configurations:
  Small array: 32 KB (fits in L1 cache)
  Large array: 16 MB (exceeds L1 cache)

Results:
Test                           L1 Miss         L1 Hit          Miss Rate
--------------------------------------------------------------------------------
Sequential (small, 32KB)       L1 Miss:         1234  L1 Hit:   1234567  Miss Rate:   0.10%
Sequential (large, 16MB)       L1 Miss:       123456  L1 Hit:   9876543  Miss Rate:   1.23%
Strided (small, stride=1)      L1 Miss:         1234  L1 Hit:   1234567  Miss Rate:   0.10%
Strided (large, stride=16)     L1 Miss:       234567  L1 Hit:   8765432  Miss Rate:   2.60%
Strided (large, stride=64)     L1 Miss:       345678  L1 Hit:   7654321  Miss Rate:   4.32%
Random access (100K)           L1 Miss:        98765  L1 Hit:      1235  Miss Rate:  98.77%
```

## Understanding Cache Behavior

### What Causes L1 Cache Misses?

1. **Cold Misses** (Compulsory)
   - First access to data
   - Cannot be avoided

2. **Capacity Misses**
   - Working set exceeds L1 cache size
   - Data is evicted before reuse
   - Example: Accessing arrays larger than L1 cache

3. **Conflict Misses**
   - Multiple addresses map to same cache set
   - Less common with modern set-associative caches

### Memory Access Patterns

**Cache-Friendly Patterns:**
```c
// Sequential access - excellent spatial locality
for (int i = 0; i < n; i++) {
    sum += array[i];
}
```

**Cache-Unfriendly Patterns:**
```c
// Random access - poor spatial locality
for (int i = 0; i < n; i++) {
    sum += array[random_index[i]];
}

// Large stride - wastes cache lines
for (int i = 0; i < n; i += 64) {
    sum += array[i];
}
```

## Optimizing for L1 Cache

### Technique 1: Loop Tiling/Blocking

Break large arrays into cache-sized chunks:

```c
// Before: Cache-unfriendly for large N
for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
        C[i][j] = A[i][j] + B[i][j];
    }
}

// After: Cache-friendly with tiling
#define TILE 32
for (int ii = 0; ii < N; ii += TILE) {
    for (int jj = 0; jj < N; jj += TILE) {
        for (int i = ii; i < ii + TILE && i < N; i++) {
            for (int j = jj; j < jj + TILE && j < N; j++) {
                C[i][j] = A[i][j] + B[i][j];
            }
        }
    }
}
```

### Technique 2: Data Structure Reorganization

```c
// Bad: Array of structures (AoS) - poor cache usage
struct Point {
    float x, y, z;
    float nx, ny, nz;  // normals (rarely used)
};
Point points[N];
for (int i = 0; i < N; i++) {
    sum += points[i].x;  // Loads unnecessary data
}

// Good: Structure of arrays (SoA) - better cache usage
struct Points {
    float x[N];
    float y[N];
    float z[N];
    float nx[N];
    float ny[N];
    float nz[N];
};
Points points;
for (int i = 0; i < N; i++) {
    sum += points.x[i];  // Only loads needed data
}
```

### Technique 3: Prefetching

```c
// Manual prefetch hints (compiler/CPU specific)
for (int i = 0; i < N; i++) {
    __builtin_prefetch(&array[i + 8], 0, 3);  // Prefetch ahead
    sum += array[i];
}
```

## Measuring Optimization Impact

Use PMC library to validate your optimizations:

```c
void compare_implementations() {
    uint64_t before_misses = 0, after_misses = 0;
    
    // Measure original
    pmc_config_t config = pmc_get_default_config(
        PMC_EVENT_L1_DCACHE_MISS, 
        PMC_MODE_COUNTING
    );
    pmc_ctx_t *pmc = pmc_create(&config);
    
    pmc_start(pmc);
    original_implementation();
    pmc_stop(pmc);
    pmc_read_count(pmc, &before_misses);
    
    // Measure optimized
    pmc_start(pmc);
    optimized_implementation();
    pmc_stop(pmc);
    pmc_read_count(pmc, &after_misses);
    
    pmc_destroy(pmc);
    
    printf("Original:  %llu cache misses\n", before_misses);
    printf("Optimized: %llu cache misses\n", after_misses);
    printf("Reduction: %.1f%%\n", 
           (double)(before_misses - after_misses) / before_misses * 100);
}
```

## Typical L1 Cache Sizes

| Processor | L1 Data Cache Size | Cache Line Size |
|-----------|-------------------|-----------------|
| Intel Core (recent) | 32-48 KB | 64 bytes |
| AMD Ryzen | 32 KB | 64 bytes |
| ARM Cortex-A | 32-64 KB | 64 bytes |

To check your system:
```bash
lscpu | grep -i cache
# or
cat /sys/devices/system/cpu/cpu0/cache/index0/size
```

## Sampling Mode for Hot Spot Analysis

Find which code locations cause most cache misses:

```c
pmc_config_t config = pmc_get_default_config(
    PMC_EVENT_L1_DCACHE_MISS, 
    PMC_MODE_SAMPLING
);
config.sample_period = 100;  // Sample every 100 misses

pmc_ctx_t *pmc = pmc_create(&config);
pmc_start(pmc);

// Run your workload
complex_algorithm();

pmc_stop(pmc);

// Analyze samples
pmc_sample_t *samples;
size_t num_samples;
pmc_read_samples(pmc, &samples, &num_samples, 0);

printf("Top locations causing cache misses:\n");
for (size_t i = 0; i < num_samples && i < 20; i++) {
    pmc_print_sample(&samples[i], i + 1);
}

free(samples);
pmc_destroy(pmc);
```

## Troubleshooting

### High Miss Rates

If you see unexpectedly high miss rates:

1. **Check working set size**: Is it larger than L1 cache?
2. **Check access patterns**: Random? Strided?
3. **Check data structures**: Too much padding? Poor layout?
4. **Check alignment**: Misaligned data can cause extra loads

### Zero Counts

If counters show zero:

1. Ensure code is actually running (not optimized away)
2. Check permissions: `sudo sysctl kernel.perf_event_paranoid=1`
3. Try `precise_ip = 0` if PEBS fails
4. Verify CPU supports the event

### Inconsistent Results

If measurements vary significantly:

1. Run multiple iterations
2. Warm up caches first
3. Disable CPU frequency scaling
4. Pin to specific CPU core

## References

- Intel® 64 and IA-32 Architectures Software Developer's Manual, Volume 3B
- "What Every Programmer Should Know About Memory" by Ulrich Drepper
- Linux perf_event_open man page

## See Also

- `ReadMe.md` - Main library documentation
- `INTEGRATION_GUIDE.md` - Integration examples
- `QUICK_REFERENCE.md` - Quick code snippets
- `example_l1_cache.c` - Complete working example

