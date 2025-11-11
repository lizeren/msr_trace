/*
 * Example: CSV-Based Multi-Event Measurement with PMC Library
 * 
 * This demonstrates the SIMPLIFIED CSV-based API for measuring multiple events
 * simultaneously with mixed counting/sampling modes.
 * 
 * The new API requires only 2 function calls:
 *   1. pmc_measure_begin_csv() at function entry (reads events from CSV)
 *   2. pmc_measure_end() at function exit
 * 
 * Build: gcc -O2 -g -Wall -Wextra example_cache_call.c -L. -lpmc -ldl -Wl,-rpath,'$ORIGIN' -o example_cache_call
 * Run:   ./example_cache_call [iterations]
 * 
 * Requires: pmc_events.csv in the same directory
 */

#include <stdio.h>
#include <stdlib.h>
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
    unsigned iters = (argc > 1) ? (unsigned)strtoul(argv[1], NULL, 0) : 100000u;
    
    printf("=== CSV-Based Multi-Event PMC Example ===\n");
    printf("Workload: %u iterations\n", iters);
    printf("Reading events from: pmc_events.csv\n\n");
    
    // ===== THIS IS ALL YOU NEED! =====
    // Start measurement (loads events from CSV, creates contexts, starts all counters)
    pmc_multi_handle_t *pmc = pmc_measure_begin_csv("workload", NULL);  // NULL = use default "pmc_events.csv"
    
    if (!pmc) {
        fprintf(stderr, "ERROR: Failed to start PMC: %s\n", pmc_get_error());
        fprintf(stderr, "Tip: sudo sysctl kernel.perf_event_paranoid=1\n");
        fprintf(stderr, "Tip: Make sure pmc_events.csv exists in the current directory\n");
        return 1;
    }
    
    // Run workload
    volatile uint64_t result = workload(iters);
    (void)result;
    
    // Stop measurement and get automatic batch report
    pmc_measure_end(pmc, 1);  // 1 = auto-report results and export to JSON
    
    
    return 0;
}
