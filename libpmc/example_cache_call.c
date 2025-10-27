/*
 * Example: Multi-Event Sampling with PMC Library
 * 
 * This demonstrates how to sample multiple events simultaneously with different
 * sample periods. We measure both near calls and L1 cache hits at the same time.
 * 
 * Build: gcc -O2 -g -Wall -Wextra example_cache_call.c -L. -lpmc -ldl -Wl,-rpath,'$ORIGIN' -o example_cache_call
 * Run:   ./example_cache_call [iterations] [mode] [near_call_period] [cache_period]
 *        mode: count (default) or sample
 *        near_call_period: sample every N near calls (default: 100)
 *        cache_period: sample every N cache hits (default: 1000)
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <stdint.h>
 #include "pmc.h"
 
 // ===== Example workload functions =====
 __attribute__((noinline)) 
 static uint64_t helper_func1(uint64_t x) { 
     return x * 2 + 1; 
 }
 
 __attribute__((noinline)) 
 static uint64_t helper_func2(uint64_t x) { 
     return x * 3 + 2; 
 }
 
 __attribute__((noinline)) 
 static uint64_t helper_func3(uint64_t x) { 
     return x * 5 + 3; 
 }
 
 typedef uint64_t (*func_ptr_t)(uint64_t);
 
 __attribute__((noinline)) 
 static uint64_t workload(unsigned iters) {
     volatile uint64_t acc = 0;
     uint32_t x = 0x12345678u;
     func_ptr_t funcs[] = { helper_func1, helper_func2, helper_func3 };
 
     for (unsigned i = 0; i < iters; i++) {
         acc += helper_func1(i);        // direct near call
         acc += helper_func2(i);        // direct near call
 
         x = x * 1103515245u + 12345u;  // LCG
         func_ptr_t selected_func = funcs[x % 3];
         acc += selected_func(i);       // indirect near call
     }
     return acc;
 }
 
// ===== Main =====
int main(int argc, char **argv) {
    unsigned iters = (argc > 1) ? (unsigned)strtoul(argv[1], NULL, 0) : 1000000u;
    
    int do_sample = 0;
    uint64_t near_call_period = 100;
    uint64_t cache_period = 1000;
    
    if (argc > 2 && strcmp(argv[2], "sample") == 0) {
        do_sample = 1;
        if (argc > 3) {
            near_call_period = strtoull(argv[3], NULL, 0);
            if (near_call_period == 0) near_call_period = 1;
        }
        if (argc > 4) {
            cache_period = strtoull(argv[4], NULL, 0);
            if (cache_period == 0) cache_period = 1;
        }
    }

    printf("=== Multi-Event PMC Example ===\n");
    printf("Workload: %u iterations\n", iters);
    printf("Mode: %s\n", do_sample ? "SAMPLING" : "COUNTING");
    if (do_sample) {
        printf("  Near call sample period: %llu\n", (unsigned long long)near_call_period);
        printf("  Cache hit sample period: %llu\n", (unsigned long long)cache_period);
    }
    printf("\n");

    pmc_mode_t mode = do_sample ? PMC_MODE_SAMPLING : PMC_MODE_COUNTING;
    
    // ===== Create PMC context for NEAR_CALL =====
    pmc_config_t config_call = pmc_get_default_config(PMC_EVENT_NEAR_CALL, mode, near_call_period);
    pmc_ctx_t *pmc_call = pmc_create(&config_call);
    if (!pmc_call) {
        fprintf(stderr, "ERROR: Failed to create PMC for near calls: %s\n", pmc_get_error());
        return 1;
    }

    // ===== Create PMC context for L1_DCACHE_HIT =====
    pmc_config_t config_cache = pmc_get_default_config(PMC_EVENT_L1_DCACHE_HIT, mode, cache_period);
    config_cache.precise_ip = 0;  // Better compatibility for cache events
    
    pmc_ctx_t *pmc_cache = pmc_create(&config_cache);
    if (!pmc_cache) {
        fprintf(stderr, "ERROR: Failed to create PMC for cache hits: %s\n", pmc_get_error());
        pmc_destroy(pmc_call);
        return 1;
    }

    printf("Measuring events:\n");
    printf("  1. %s (period: %llu)\n", pmc_event_name(config_call.event), 
           (unsigned long long)near_call_period);
    printf("  2. %s (period: %llu)\n", pmc_event_name(config_cache.event),
           (unsigned long long)cache_period);
    printf("\n");

    // ===== Start BOTH measurements =====
    if (pmc_start(pmc_call) != 0) {
        fprintf(stderr, "ERROR: Failed to start near call PMC: %s\n", pmc_get_error());
        pmc_destroy(pmc_call);
        pmc_destroy(pmc_cache);
        return 1;
    }
    
    if (pmc_start(pmc_cache) != 0) {
        fprintf(stderr, "ERROR: Failed to start cache PMC: %s\n", pmc_get_error());
        pmc_destroy(pmc_call);
        pmc_destroy(pmc_cache);
        return 1;
    }

    // ===== Run workload ONCE (both counters measure simultaneously) =====
    volatile uint64_t result = workload(iters);
    (void)result;  // Prevent optimization

    // ===== Stop BOTH measurements =====
    if (pmc_stop(pmc_call) != 0) {
        fprintf(stderr, "ERROR: Failed to stop near call PMC: %s\n", pmc_get_error());
    }
    
    if (pmc_stop(pmc_cache) != 0) {
        fprintf(stderr, "ERROR: Failed to stop cache PMC: %s\n", pmc_get_error());
    }
 
    // ===== Read Results =====
    printf("Results:\n");
    printf("========================================\n\n");
    
    if (do_sample) {
        // ===== NEAR CALL Sampling Results =====
        pmc_sample_t *call_samples = NULL;
        size_t num_call_samples = 0;
        
        if (pmc_read_samples(pmc_call, &call_samples, &num_call_samples, 0) != 0) {
            fprintf(stderr, "ERROR: Failed to read near call samples: %s\n", pmc_get_error());
        } else {
            printf("--- Near Call Samples ---\n");
            printf("Total samples collected: %zu\n", num_call_samples);
            
            // Print first 5 samples
            size_t to_print = num_call_samples < 5 ? num_call_samples : 5;
            if (to_print > 0) {
                printf("\nFirst %zu near call samples:\n", to_print);
                for (size_t i = 0; i < to_print; i++) {
                    pmc_print_sample(&call_samples[i], (int)i + 1);
                }
            }
            if (num_call_samples > 5) {
                printf("  ... (%zu more samples not shown)\n", num_call_samples - 5);
            }
            
            // Read total count
            uint64_t call_count = 0;
            if (pmc_read_count(pmc_call, &call_count) == 0) {
                printf("\nTotal near calls: %llu\n", (unsigned long long)call_count);
                printf("Calls per sample: %.1f\n", (double)call_count / num_call_samples);
            }
            
            free(call_samples);
        }
        
        printf("\n");
        
        // ===== L1 CACHE HIT Sampling Results =====
        pmc_sample_t *cache_samples = NULL;
        size_t num_cache_samples = 0;
        
        if (pmc_read_samples(pmc_cache, &cache_samples, &num_cache_samples, 0) != 0) {
            fprintf(stderr, "ERROR: Failed to read cache samples: %s\n", pmc_get_error());
        } else {
            printf("--- L1 Cache Hit Samples ---\n");
            printf("Total samples collected: %zu\n", num_cache_samples);
            
            // Print first 5 samples
            size_t to_print = num_cache_samples < 5 ? num_cache_samples : 5;
            if (to_print > 0) {
                printf("\nFirst %zu cache hit samples:\n", to_print);
                for (size_t i = 0; i < to_print; i++) {
                    pmc_print_sample(&cache_samples[i], (int)i + 1);
                }
            }
            if (num_cache_samples > 5) {
                printf("  ... (%zu more samples not shown)\n", num_cache_samples - 5);
            }
            
            // Read total count
            uint64_t cache_count = 0;
            if (pmc_read_count(pmc_cache, &cache_count) == 0) {
                printf("\nTotal cache hits: %llu\n", (unsigned long long)cache_count);
                printf("Hits per sample: %.1f\n", (double)cache_count / num_cache_samples);
            }
            
            free(cache_samples);
        }

    } else {
        // ===== Counting Mode Results =====
        uint64_t call_count = 0;
        uint64_t cache_count = 0;
        
        if (pmc_read_count(pmc_call, &call_count) == 0) {
            printf("Total %s: %llu\n", pmc_event_name(config_call.event), 
                   (unsigned long long)call_count);
            printf("  Average per iteration: %.2f\n", (double)call_count / iters);
        }
        
        printf("\n");
        
        if (pmc_read_count(pmc_cache, &cache_count) == 0) {
            printf("Total %s: %llu\n", pmc_event_name(config_cache.event),
                   (unsigned long long)cache_count);
            printf("  Average per iteration: %.2f\n", (double)cache_count / iters);
        }
    }

    // ===== Clean up =====
    pmc_destroy(pmc_call);
    pmc_destroy(pmc_cache);
    
    printf("\n=== Done ===\n");
    return 0;
}
 
 