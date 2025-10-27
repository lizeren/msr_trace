/*
 * PMC (Performance Monitoring Counter) Library
 * 
 * A flexible shared library for measuring hardware performance events
 * using Linux perf_event_open. Supports both counting and sampling modes.
 * 
 * Usage:
 *   1. Create a PMC context with pmc_create()
 *   2. Start measurement with pmc_start()
 *   3. Run your workload
 *   4. Stop measurement with pmc_stop()
 *   5. Read results with pmc_read_count() or pmc_read_samples()
 *   6. Destroy context with pmc_destroy()
 */

#ifndef PMC_H
#define PMC_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct pmc_ctx pmc_ctx_t;

// ===== Event Types =====
typedef enum {
    PMC_EVENT_NEAR_CALL,           // BR_INST_RETIRED.NEAR_CALL
    PMC_EVENT_CONDITIONAL_BRANCH,  // BR_INST_RETIRED.CONDITIONAL
    PMC_EVENT_BRANCH_MISPREDICT,   // BR_MISP_RETIRED.ALL_BRANCHES
    PMC_EVENT_L1_DCACHE_MISS,      // MEM_LOAD_RETIRED.L1_MISS
    PMC_EVENT_L1_DCACHE_HIT,       // MEM_LOAD_RETIRED.L1_HIT
    PMC_EVENT_CYCLES,              // CPU_CLK_UNHALTED.THREAD
    PMC_EVENT_INSTRUCTIONS,        // INST_RETIRED.ANY
} pmc_event_type_t;

// ===== Mode Types =====
typedef enum {
    PMC_MODE_COUNTING,   // Just count events
    PMC_MODE_SAMPLING,   // Sample every N events
} pmc_mode_t;

// ===== Configuration =====
typedef struct {
    pmc_event_type_t event;        // Event to monitor
    pmc_mode_t mode;               // Counting or sampling
    uint64_t sample_period;        // For sampling: sample every N events (ignored in counting mode)
    int exclude_kernel;            // Exclude kernel events (1=yes, 0=no)
    int exclude_hv;                // Exclude hypervisor events
    int precise_ip;                // PEBS precision level (0-3), 2 recommended for Intel.
                                   // 0 = best effort, 1 = requested, 2 = requested + skid constrained, 3 = must be precise
    unsigned int ring_buffer_pages;// Number of data pages for ring buffer (power of 2, default 128)
} pmc_config_t;

// ===== Sample Data =====
typedef struct {
    uint64_t ip;       // Instruction pointer
    uint32_t pid;      // Process ID
    uint32_t tid;      // Thread ID
    uint64_t time;     // Timestamp
} pmc_sample_t;

// ===== API Functions =====

/**
 * Get default configuration for an event type.
 * 
 * @param event Event type to configure
 * @param mode Counting or sampling mode
 * @return Default configuration structure
 */
pmc_config_t pmc_get_default_config(pmc_event_type_t event, pmc_mode_t mode, uint64_t sample_period);

/**
 * Create a PMC context with the given configuration.
 * 
 * @param config Configuration for the performance counter
 * @return Context handle on success, NULL on failure
 */
pmc_ctx_t* pmc_create(const pmc_config_t *config);

/**
 * Start measuring performance events.
 * Resets the counter and begins collection.
 * 
 * @param ctx PMC context
 * @return 0 on success, -1 on failure
 */
int pmc_start(pmc_ctx_t *ctx);

/**
 * Stop measuring performance events.
 * 
 * @param ctx PMC context
 * @return 0 on success, -1 on failure
 */
int pmc_stop(pmc_ctx_t *ctx);

/**
 * Read the event count (for counting mode).
 * 
 * @param ctx PMC context
 * @param count Output: event count
 * @return 0 on success, -1 on failure
 */
int pmc_read_count(pmc_ctx_t *ctx, uint64_t *count);

/**
 * Read collected samples (for sampling mode).
 * Allocates memory for samples which must be freed by caller.
 * 
 * @param ctx PMC context
 * @param samples Output: pointer to array of samples (caller must free())
 * @param num_samples Output: number of samples collected
 * @param max_samples Maximum number of samples to retrieve (0 = all)
 * @return 0 on success, -1 on failure
 */
int pmc_read_samples(pmc_ctx_t *ctx, pmc_sample_t **samples, 
                     size_t *num_samples, size_t max_samples);

/**
 * Print a sample with symbol information (requires -ldl).
 * 
 * @param sample Sample to print
 * @param index Sample index for display
 */
void pmc_print_sample(const pmc_sample_t *sample, int index);

/**
 * Get a human-readable name for an event type.
 * 
 * @param event Event type
 * @return Event name string
 */
const char* pmc_event_name(pmc_event_type_t event);

/**
 * Destroy a PMC context and free all resources.
 * 
 * @param ctx PMC context
 */
void pmc_destroy(pmc_ctx_t *ctx);

/**
 * Get the last error message.
 * 
 * @return Error message string (thread-local)
 */
const char* pmc_get_error(void);

#ifdef __cplusplus
}
#endif

#endif /* PMC_H */

