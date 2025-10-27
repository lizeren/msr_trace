/*
 * Example: Using the PMC library to measure performance events
 * 
 * This demonstrates how to integrate the PMC library into your program
 * to measure hardware performance events like near calls, branches, cache misses, etc.
 * 
 * Build: gcc -O2 -Wall -Wextra example_usage.c -L. -lpmc -ldl -o example
 * Run:   LD_LIBRARY_PATH=. ./example [iterations] [mode]
 *        mode: count (default) or sample
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
    if (argc > 2 && strcmp(argv[2], "sample") == 0) {
        do_sample = 1;
    }

    printf("=== PMC Library Example ===\n");
    printf("Workload: %u iterations\n", iters);
    printf("Mode: %s\n\n", do_sample ? "SAMPLING" : "COUNTING");

    // Create PMC configuration
    pmc_mode_t mode = do_sample ? PMC_MODE_SAMPLING : PMC_MODE_COUNTING;
    pmc_config_t config = pmc_get_default_config(PMC_EVENT_NEAR_CALL, mode, 100);
    

    // Create PMC context
    pmc_ctx_t *pmc = pmc_create(&config);
    if (!pmc) {
        fprintf(stderr, "ERROR: Failed to create PMC context: %s\n", pmc_get_error());
        return 1;
    }

    printf("Measuring event: %s\n", pmc_event_name(config.event));
    if (do_sample) {
        printf("Sample period: %llu events\n", (unsigned long long)config.sample_period);
    }
    printf("\n");

    // Start measurement
    if (pmc_start(pmc) != 0) {
        fprintf(stderr, "ERROR: Failed to start PMC: %s\n", pmc_get_error());
        pmc_destroy(pmc);
        return 1;
    }

    // Run workload
    volatile uint64_t result = workload(iters);

    // Stop measurement
    if (pmc_stop(pmc) != 0) {
        fprintf(stderr, "ERROR: Failed to stop PMC: %s\n", pmc_get_error());
        pmc_destroy(pmc);
        return 1;
    }

    // printf("Workload result (sink): %llu\n\n", (unsigned long long)result);

    // Read results
    if (do_sample) {
        // ===== Sampling mode =====
        pmc_sample_t *samples = NULL;
        size_t num_samples = 0;
        
        if (pmc_read_samples(pmc, &samples, &num_samples, 0) != 0) {
            fprintf(stderr, "ERROR: Failed to read samples: %s\n", pmc_get_error());
            pmc_destroy(pmc);
            return 1;
        }

        printf("Total samples collected: %zu\n", num_samples);
        
        /*
        // Print first 20 samples
        size_t to_print = num_samples < 20 ? num_samples : 20;
        if (to_print > 0) {
            printf("\nFirst %zu samples:\n", to_print);
            for (size_t i = 0; i < to_print; i++) {
                pmc_print_sample(&samples[i], (int)i + 1);
            }
        }

        if (num_samples > 20) {
            printf("... (%zu more samples not shown)\n", num_samples - 20);
        }
        */      

        free(samples);

        // Also read total count
        uint64_t count = 0;
        if (pmc_read_count(pmc, &count) == 0 && count > 0) {
            printf("\nTotal counter value: %llu\n", (unsigned long long)count);
            // double calls_per_iter = (double)count / iters;
            // printf("Average near calls per iteration: %.2f\n", calls_per_iter);
        }

    } else {
        // ===== Counting mode =====
        uint64_t count = 0;
        
        if (pmc_read_count(pmc, &count) != 0) {
            fprintf(stderr, "ERROR: Failed to read count: %s\n", pmc_get_error());
            pmc_destroy(pmc);
            return 1;
        }

        printf("Total %s count: %llu\n", pmc_event_name(config.event), 
               (unsigned long long)count);
        
        double calls_per_iter = (double)count / iters;
        printf("Average near calls per iteration: %.2f\n", calls_per_iter);
    }

    // Clean up
    pmc_destroy(pmc);
    
    printf("\n=== Done ===\n");
    return 0;
}

