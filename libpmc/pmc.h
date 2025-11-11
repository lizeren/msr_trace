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

// ===== Mode Types =====
typedef enum {
    PMC_MODE_COUNTING,   // Just count events
    PMC_MODE_SAMPLING,   // Sample every N events
} pmc_mode_t;

// ===== Configuration =====
typedef struct {
    const char *event;             // Event name (e.g., "BR_INST_RETIRED.NEAR_CALL", "CPU_CLK_UNHALTED.THREAD")
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

// ===== Multi-Event Request =====
typedef struct {
    const char *event;         // Event name (e.g., "BR_INST_RETIRED.NEAR_CALL")
    pmc_mode_t mode;           // Counting or sampling
    uint64_t sample_period;    // For sampling mode (ignored in counting)
    int precise_ip;            // PEBS precision (0-3), 0 recommended for compatibility
} pmc_event_request_t;

// Forward declaration for multi-event handle
typedef struct pmc_multi_handle pmc_multi_handle_t;

// ===== Simplified API for LLVM Injection =====

/**
 * Begin measurement session with multiple events (manual event specification).
 * 
 * @param label Identifier for this measurement (e.g., function name)
 * @param events Array of events to measure
 * @param num_events Number of events in array
 * @return Handle for this measurement session
 */
pmc_multi_handle_t* pmc_measure_begin(const char *label, 
                                       const pmc_event_request_t *events,
                                       size_t num_events);

/**
 * Begin measurement session with events loaded from CSV file.
 * This is the SIMPLIFIED function for LLVM injection - just needs a label!
 * 
 * CSV format (with header):
 *   event_name,mode,sample_period
 *   BR_INST_RETIRED.NEAR_CALL,counting,0
 *   MEM_LOAD_RETIRED.L1_HIT,sampling,1000
 *   CPU_CLK_UNHALTED.THREAD,counting,0
 * 
 * @param label Identifier for this measurement (e.g., function name)
 * @param csv_path Path to CSV file (NULL or empty for default "pmc_events.csv")
 * @return Handle for this measurement session
 */
pmc_multi_handle_t* pmc_measure_begin_csv(const char *label, const char *csv_path);

/**
 * End measurement session and optionally report results.
 * This is the ONLY function needed at function exit.
 * 
 * @param handle Handle from pmc_measure_begin
 * @param report If true, print results to stdout
 */
void pmc_measure_end(pmc_multi_handle_t *handle, int report);

/**
 * Batch report all results (counting + sampling summary).
 * 
 * @param handle Handle from pmc_measure_begin
 */
void pmc_report_all(pmc_multi_handle_t *handle);

/**
 * Export measurement results to JSON file.
 * 
 * @param handle Handle from pmc_measure_begin
 * @param json_path Path to output JSON file
 * @return 0 on success, -1 on failure
 */
int pmc_export_json(pmc_multi_handle_t *handle, const char *json_path);

/**
 * Get specific event count.
 * 
 * @param handle Handle from pmc_measure_begin
 * @param event_name Event name to query (e.g., "BR_INST_RETIRED.NEAR_CALL")
 * @param count Output: event count
 * @return 0 on success, -1 on failure
 */
int pmc_get_count(pmc_multi_handle_t *handle, const char *event_name, uint64_t *count);

/**
 * Get specific event samples.
 * 
 * @param handle Handle from pmc_measure_begin
 * @param event_name Event name to query (e.g., "MEM_LOAD_RETIRED.L1_MISS")
 * @param samples Output: pointer to array of samples (caller must free())
 * @param num_samples Output: number of samples collected
 * @return 0 on success, -1 on failure
 */
int pmc_get_samples(pmc_multi_handle_t *handle, const char *event_name,
                    pmc_sample_t **samples, size_t *num_samples);

// ===== Original API (kept for backwards compatibility) =====

/**
 * Get default configuration for an event.
 * 
 * @param event_name Event name (e.g., "BR_INST_RETIRED.NEAR_CALL")
 * @param mode Counting or sampling mode
 * @param sample_period Sample period for sampling mode
 * @return Default configuration structure
 */
pmc_config_t pmc_get_default_config(const char *event_name, pmc_mode_t mode, uint64_t sample_period);

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

