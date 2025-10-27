/*
 * Example: Measuring L1 Data Cache Hits and Misses with PMC Library
 * 
 * This demonstrates how to use the PMC library to measure L1 data cache
 * behavior with different memory access patterns, including pointer chasing
 * to defeat hardware prefetchers.
 * 
 * Build: gcc -O2 -Wall example_l1_cache.c -L. -lpmc -ldl -Wl,-rpath,'$ORIGIN' -o example_l1_cache
 * Run:   ./example_l1_cache [hit_sweeps] [miss_steps] [big_buffer_mb] [mode] [sample_period]
 *        mode: count (default) or sample
 *        sample_period: sample every N events (default: 1000)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "pmc.h"

#define L1_SIZE (32 * 1024)            // 32KB (typical L1 data cache size)

// Memory barrier to prevent optimization
static void touch_barrier(void) { 
    asm volatile("" ::: "memory"); 
}

// HIT phase: repeatedly sweep a small buffer that fits in L1D cache
__attribute__((noinline))
static uint64_t l1_hit_phase(uint8_t *buf, size_t sz, uint64_t iters) {
    volatile uint64_t acc = 0;
    for (uint64_t k = 0; k < iters; k++) {
        for (size_t i = 0; i < sz; i += 64) {  // 64B per cache line
            acc += buf[i];
        }
        touch_barrier();
    }
    return acc;
}

// Build a cache-line-granularity random permutation for pointer chasing
// This creates a ring of pointers that jump randomly through the buffer
static void build_ptr_chase(uint8_t *buf, size_t bytes) {
    const size_t step = 64;  // Cache line size
    size_t n = bytes / step;
    size_t *idx = (size_t*)malloc(n * sizeof(size_t));
    
    // Initialize indices
    for (size_t i = 0; i < n; i++) {
        idx[i] = i;
    }
    
    // Fisher-Yates shuffle to randomize
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = (size_t)(rand() % (i + 1));
        size_t t = idx[i];
        idx[i] = idx[j];
        idx[j] = t;
    }
    
    // Each cache line stores a pointer to the next random cache line
    for (size_t i = 0; i < n; i++) {
        size_t next = idx[(i + 1) % n];
        *(size_t*)(buf + idx[i] * step) = (size_t)(buf + next * step);
    }
    
    free(idx);
}

// MISS phase: dependent pointer chasing over a large array
// This defeats hardware prefetchers because each load depends on the previous one
__attribute__((noinline))
static uint64_t l1_miss_phase(uint8_t *buf, size_t sz_bytes, uint64_t steps) {
    volatile uint64_t acc = 0;
    uint8_t *p = buf;
    
    for (uint64_t i = 0; i < steps; i++) {
        p = *(uint8_t * volatile *)p;  // Dependent load → likely L1 miss
        acc += (uintptr_t)p & 1;
        touch_barrier();
    }
    
    return acc;
}

// Measure and print results for a single phase
static void run_phase(const char *name,
                     uint64_t (*phase_fn)(uint8_t*, size_t, uint64_t),
                     uint8_t *buf, size_t sz, uint64_t iters,
                     int do_sampling, uint64_t sample_period) 
{
    printf("\n[%s]\n", name);
    pmc_mode_t mode = do_sampling ? PMC_MODE_SAMPLING : PMC_MODE_COUNTING;
    
    // ===== Measure L1 MISSES =====
    pmc_config_t config_miss = pmc_get_default_config(
        PMC_EVENT_L1_DCACHE_MISS, 
        mode,
        sample_period
    );
    config_miss.precise_ip = 0;  // Use best-effort mode for compatibility
    
    pmc_ctx_t *pmc_miss = pmc_create(&config_miss);
    if (!pmc_miss) {
        fprintf(stderr, "ERROR: Cannot create PMC for L1 miss: %s\n", pmc_get_error());
        return;
    }
    
    pmc_start(pmc_miss);
    volatile uint64_t sink = phase_fn(buf, sz, iters);
    pmc_stop(pmc_miss);
    (void)sink;  // Prevent optimization
    
    uint64_t l1_miss = 0;
    pmc_sample_t *miss_samples = NULL;
    size_t num_miss_samples = 0;
    
    if (do_sampling) {
        if (pmc_read_samples(pmc_miss, &miss_samples, &num_miss_samples, 0) != 0) {
            fprintf(stderr, "ERROR: Failed to read miss samples: %s\n", pmc_get_error());
            pmc_destroy(pmc_miss);
            return;
        }
        // Also read count for summary
        pmc_read_count(pmc_miss, &l1_miss);
    } else {
        pmc_read_count(pmc_miss, &l1_miss);
    }
    pmc_destroy(pmc_miss);
    
    // ===== Measure L1 HITS =====
    pmc_config_t config_hit = pmc_get_default_config(
        PMC_EVENT_L1_DCACHE_HIT, 
        mode,
        sample_period
    );
    config_hit.precise_ip = 0;  // Use best-effort mode for compatibility
    
    pmc_ctx_t *pmc_hit = pmc_create(&config_hit);
    if (!pmc_hit) {
        fprintf(stderr, "ERROR: Cannot create PMC for L1 hit: %s\n", pmc_get_error());
        if (miss_samples) free(miss_samples);
        return;
    }
    
    pmc_start(pmc_hit);
    sink = phase_fn(buf, sz, iters);
    pmc_stop(pmc_hit);
    (void)sink;
    
    uint64_t l1_hit = 0;
    pmc_sample_t *hit_samples = NULL;
    size_t num_hit_samples = 0;
    
    if (do_sampling) {
        if (pmc_read_samples(pmc_hit, &hit_samples, &num_hit_samples, 0) != 0) {
            fprintf(stderr, "ERROR: Failed to read hit samples: %s\n", pmc_get_error());
            pmc_destroy(pmc_hit);
            if (miss_samples) free(miss_samples);
            return;
        }
        // Also read count for summary
        pmc_read_count(pmc_hit, &l1_hit);
    } else {
        pmc_read_count(pmc_hit, &l1_hit);
    }
    pmc_destroy(pmc_hit);
    
    // ===== Print Results =====
    uint64_t total = l1_miss + l1_hit;
    double miss_rate = total > 0 ? (double)l1_miss / total * 100.0 : 0.0;
    
    printf("MEM_LOAD_RETIRED.L1_MISS : %llu\n", (unsigned long long)l1_miss);
    printf("MEM_LOAD_RETIRED.L1_HIT  : %llu\n", (unsigned long long)l1_hit);
    printf("Total loads              : %llu\n", (unsigned long long)total);
    printf("Miss rate                : %.2f%%\n", miss_rate);
    
    if (do_sampling) {
        printf("\n--- Sampling Results ---\n");
        printf("L1 Miss samples collected: %zu (period: %llu)\n", 
               num_miss_samples, (unsigned long long)sample_period);
        printf("L1 Hit samples collected:  %zu (period: %llu)\n", 
               num_hit_samples, (unsigned long long)sample_period);
        
        // Print first few miss samples
        if (num_miss_samples > 0) {
            printf("\nFirst L1 MISS sample locations:\n");
            size_t show = num_miss_samples < 5 ? num_miss_samples : 5;
            for (size_t i = 0; i < show; i++) {
                pmc_print_sample(&miss_samples[i], (int)i + 1);
            }
            if (num_miss_samples > 5) {
                printf("  ... (%zu more miss samples)\n", num_miss_samples - 5);
            }
        }
        
        // Print first few hit samples
        if (num_hit_samples > 0) {
            printf("\nFirst L1 HIT sample locations:\n");
            size_t show = num_hit_samples < 5 ? num_hit_samples : 5;
            for (size_t i = 0; i < show; i++) {
                pmc_print_sample(&hit_samples[i], (int)i + 1);
            }
            if (num_hit_samples > 5) {
                printf("  ... (%zu more hit samples)\n", num_hit_samples - 5);
            }
        }
    }
    
    // Cleanup
    if (miss_samples) free(miss_samples);
    if (hit_samples) free(hit_samples);
}

int main(int argc, char **argv) {
    printf("=== L1 Data Cache Behavior Analysis ===\n");
    printf("Using PMC Library to measure cache hits and misses\n\n");
    
    // Parse command line arguments
    const uint64_t HIT_SWEEPS = (argc > 1) ? strtoull(argv[1], NULL, 0) : 2000ULL;
    const uint64_t MISS_STEPS = (argc > 2) ? strtoull(argv[2], NULL, 0) : 2000ULL;
    const size_t BIG_SIZE_MB = (argc > 3) ? strtoull(argv[3], NULL, 0) : 64ULL;
    const size_t BIG_SIZE = BIG_SIZE_MB * 1024ULL * 1024ULL;
    
    int do_sampling = 0;
    uint64_t sample_period = 1000;  // default
    if (argc > 4 && strcmp(argv[4], "sample") == 0) {
        do_sampling = 1;
        if (argc > 5) {
            sample_period = strtoull(argv[5], NULL, 0);
            if (sample_period == 0) sample_period = 1;
        }
    }
    
    printf("Configuration:\n");
    printf("  L1 cache size     : %d KB (small buffer)\n", (int)(L1_SIZE / 1024));
    printf("  Large buffer size : %zu MB\n", BIG_SIZE_MB);
    printf("  HIT phase sweeps  : %llu\n", (unsigned long long)HIT_SWEEPS);
    printf("  MISS phase steps  : %llu\n", (unsigned long long)MISS_STEPS);
    printf("  Mode              : %s\n", do_sampling ? "SAMPLING" : "COUNTING");
    if (do_sampling) {
        printf("  Sample period     : %llu events\n", (unsigned long long)sample_period);
    }
    printf("\n");
    
    // Check if PMC is available - use COUNTING mode for simple test
    pmc_config_t test_config = pmc_get_default_config(
        PMC_EVENT_L1_DCACHE_MISS, 
        PMC_MODE_COUNTING,
        0  // Not used in counting mode
    );
    
    // Try with precise_ip=0 first (some CPUs don't support PEBS for all events)
    test_config.precise_ip = 0;
    
    pmc_ctx_t *test_pmc = pmc_create(&test_config);
    if (!test_pmc) {
        fprintf(stderr, "ERROR: Cannot initialize PMC: %s\n", pmc_get_error());
        fprintf(stderr, "Tip: sudo sysctl kernel.perf_event_paranoid=1\n");
        fprintf(stderr, "Note: Your CPU may not support MEM_LOAD_RETIRED events.\n");
        fprintf(stderr, "      Check with: perf list | grep MEM_LOAD\n");
        return 1;
    }
    pmc_destroy(test_pmc);
    printf("PMC initialized successfully\n");
    
    // Allocate aligned buffers (cache-aligned)
    uint8_t *small_buf = aligned_alloc(64, L1_SIZE);
    uint8_t *big_buf = aligned_alloc(64, BIG_SIZE);
    
    if (!small_buf || !big_buf) {
        fprintf(stderr, "ERROR: Memory allocation failed\n");
        perror("aligned_alloc");
        return 1;
    }
    
    // Initialize buffers
    srand(0xBADC0DE);  // Deterministic seed
    memset(small_buf, 1, L1_SIZE);
    memset(big_buf, 0, BIG_SIZE);
    
    // Build pointer-chase ring for the large buffer
    printf("Building pointer-chase structure for MISS phase...\n");
    build_ptr_chase(big_buf, BIG_SIZE);
    printf("Done.\n");
    
    // === Phase 1: L1 HITs ===
    // Small buffer that fits entirely in L1 cache, swept repeatedly
    run_phase("HIT phase (small L1-friendly sweeps)",
              l1_hit_phase, small_buf, L1_SIZE, HIT_SWEEPS,
              do_sampling, sample_period);
    
    // === Phase 2: L1 MISSes ===
    // Pointer chasing through large buffer defeats prefetcher
    // Each load depends on previous load → true cache misses
    run_phase("MISS phase (pointer chase over large buffer)",
              l1_miss_phase, big_buf, BIG_SIZE, MISS_STEPS,
              do_sampling, sample_period);
    
    printf("\n");
    printf("Key Observations:\n");
    printf("  - HIT phase: High hit rate (data fits in L1, good spatial locality)\n");
    printf("  - MISS phase: High miss rate (pointer chasing defeats prefetcher)\n");
    printf("  - The miss rate difference demonstrates cache effectiveness\n");
    if (do_sampling) {
        printf("\nSampling Mode Benefits:\n");
        printf("  - Identifies exact code locations causing cache misses/hits\n");
        printf("  - Useful for hot spot analysis and optimization\n");
        printf("  - Sample period controls overhead vs. detail trade-off\n");
    }
    printf("\n");
    
    // Cleanup
    free(small_buf);
    free(big_buf);
    
    printf("=== Test Complete ===\n");
    return 0;
}

