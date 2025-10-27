/*
 * Example: Simplified Multi-Event Measurement with PMC Library
 * 
 * This demonstrates the NEW simplified API for measuring multiple events
 * simultaneously with mixed counting/sampling modes.
 * 
 * The new API requires only 2 function calls:
 *   1. pmc_measure_begin() at function entry
 *   2. pmc_measure_end() at function exit
 * 
 * Build: gcc -O2 -g -Wall -Wextra example_cache_call.c -L. -lpmc -ldl -Wl,-rpath,'$ORIGIN' -o example_cache_call
 * Run:   ./example_cache_call [iterations]
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
    unsigned iters = (argc > 1) ? (unsigned)strtoul(argv[1], NULL, 0) : 100000u;
    
    printf("=== Simplified Multi-Event PMC Example ===\n");
    printf("Workload: %u iterations\n\n", iters);
    
    // ===== Define events to measure (mixed counting + sampling) =====
    pmc_event_request_t events[] = {
        // Event 1: Count near calls
        {
            .event = PMC_EVENT_NEAR_CALL,
            .mode = PMC_MODE_COUNTING,
            .sample_period = 0,  // Ignored in counting mode
            .precise_ip = 0
        },
        // Event 2: Sample cache hits every 1000 events
        {
            .event = PMC_EVENT_L1_DCACHE_HIT,
            .mode = PMC_MODE_SAMPLING,
            .sample_period = 1000,
            .precise_ip = 0  // Required for cache events
        },
        // Event 3: Sample cache misses every 500 events
        {
            .event = PMC_EVENT_L1_DCACHE_MISS,
            .mode = PMC_MODE_SAMPLING,
            .sample_period = 500,
            .precise_ip = 0
        },
        // Event 4: Count CPU cycles
        {
            .event = PMC_EVENT_CYCLES,
            .mode = PMC_MODE_COUNTING,
            .sample_period = 0,
            .precise_ip = 0
        }
    };
    size_t num_events = sizeof(events) / sizeof(events[0]);
    
    printf("Measuring %zu events:\n", num_events);
    for (size_t i = 0; i < num_events; i++) {
        printf("  [%zu] %s (%s", 
               i + 1,
               pmc_event_name(events[i].event),
               events[i].mode == PMC_MODE_COUNTING ? "count" : "sample");
        if (events[i].mode == PMC_MODE_SAMPLING) {
            printf(", period=%llu", (unsigned long long)events[i].sample_period);
        }
        printf(")\n");
    }
    printf("\n");
    
    // ===== THIS IS ALL YOU NEED! =====
    // Start measurement (creates contexts, starts all counters)
    pmc_multi_handle_t *pmc = pmc_measure_begin("workload", events, num_events);
    
    if (!pmc) {
        fprintf(stderr, "ERROR: Failed to start PMC: %s\n", pmc_get_error());
        fprintf(stderr, "Tip: sudo sysctl kernel.perf_event_paranoid=1\n");
        return 1;
    }
    
    // Run workload
    volatile uint64_t result = workload(iters);
    (void)result;
    
    // Stop measurement and get automatic batch report
    pmc_measure_end(pmc, 1);  // 1 = auto-report results
    
    printf("=== Done ===\n");
    printf("\nThis example shows the NEW simplified API:\n");
    printf("  - Define events once (array of pmc_event_request_t)\n");
    printf("  - Call pmc_measure_begin() at function entry\n");
    printf("  - Call pmc_measure_end() at function exit\n");
    printf("  - Automatic batch report with all results!\n\n");
    
    printf("Perfect for LLVM auto-injection:\n");
    printf("  void* h = pmc_measure_begin(__func__, events, n);\n");
    printf("  // ... function body ...\n");
    printf("  pmc_measure_end(h, 1);\n");
    
    return 0;
}
