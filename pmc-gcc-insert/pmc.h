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
    PMC_MODE_COUNTING,        // Just count events (no samples)
    PMC_MODE_SAMPLING,        // Sample every N event occurrences
    PMC_MODE_SAMPLING_FREQ,   // Sample at N Hz frequency (time-based)
} pmc_mode_t;

// ===== Configuration =====
// For perf_event_open()
typedef struct {
    const char *event;             // Event name (e.g., "BR_INST_RETIRED.NEAR_CALL", "CPU_CLK_UNHALTED.THREAD")
    pmc_mode_t mode;               // Counting, event-based sampling, or frequency-based sampling
    uint64_t sample_period;        // For PMC_MODE_SAMPLING: sample every N events
                                   // For PMC_MODE_SAMPLING_FREQ: sample at N Hz
                                   // Ignored in PMC_MODE_COUNTING
    int exclude_kernel;            // Exclude kernel events (1=yes, 0=no)
    int exclude_hv;                // Exclude hypervisor events
    int precise_ip;                // PEBS precision level (0-3), 2 recommended for Intel.
                                   // 0 = best effort, 1 = requested, 2 = requested + skid constrained, 3 = must be precise
    unsigned int ring_buffer_pages;// Number of data pages for ring buffer (power of 2, default 128)
    uint32_t event_number;         // Event number (e.g., 0xC4) - from CSV or lookup
    uint32_t umask;                // Unit mask (e.g., 0x02) - from CSV or lookup
    int is_raw;                    // 1 for raw events, 0 for fixed counters - from CSV or lookup
    int pinned;                    // 1 for pinned events, 0 for non-pinned events
} pmc_config_t;

// ===== Sample Data =====
typedef struct {
    uint64_t ip;       // Instruction pointer
    uint32_t pid;      // Process ID
    uint32_t tid;      // Thread ID
    uint64_t time;     // Timestamp
    uint64_t count;    // Counter value at this sample point
    uint32_t cpu;      // CPU number where sample was taken
    uint32_t res;      // Reserved (padding)
    uint64_t period;   // Sample period value
} pmc_sample_t;

// ===== Multi-Event Request =====
// Come from csv file
typedef struct {
    const char *event;         // Event name (e.g., "BR_INST_RETIRED.NEAR_CALL")
    pmc_mode_t mode;           // PMC_MODE_COUNTING, PMC_MODE_SAMPLING, or PMC_MODE_SAMPLING_FREQ
    uint64_t sample_period;    // For SAMPLING: events per sample; For SAMPLING_FREQ: Hz frequency
    int precise_ip;            // PEBS precision (0-3), 0 recommended for compatibility
    uint32_t event_number;     // Event number from CSV (e.g., 0xC4)
    uint32_t umask;            // Unit mask from CSV (e.g., 0x02)
    int is_raw;                // 1 for raw events, 0 for fixed counters
} pmc_event_request_t;

// Forward declaration for multi-event handle
typedef struct pmc_multi_handle pmc_multi_handle_t;

// ===== Simplified API for Instrumentation =====

/**
 * Begin measurement session with events loaded from CSV file.
 * This is the SIMPLIFIED function for LLVM injection - just needs a label!
 * 
 * CSV format (with header):
 *   index,event_name,event_number,umask,mode,sample_period,type
 *   0,BR_INST_RETIRED.NEAR_CALL,0xC4,0x02,counting,0,raw
 *   1,MEM_LOAD_RETIRED.L1_HIT,0xD1,0x01,sampling,1000,raw        # Sample every 1000 events
 *   2,MEM_LOAD_RETIRED.L1_MISS,0xD1,0x08,sampling_freq,4000,raw  # Sample at 4000 Hz (4kHz)
 *   24,CPU_CLK_UNHALTED.THREAD,0x00,0x00,counting,0,fixed        # Fixed counter (always included)
 *   25,INST_RETIRED.ANY,0x00,0x00,counting,0,fixed               # Fixed counter (always included)
 * 
 * Columns:
 *   - index: Event index for selection
 *   - event_name: Human-readable event name
 *   - event_number: Hardware event number (hex, e.g., 0xC4)
 *   - umask: Unit mask (hex, e.g., 0x02)
 *   - mode: counting, sampling, or sampling_freq
 *   - sample_period: For sampling modes (ignored in counting)
 *   - type: raw (programmable counter) or fixed (fixed counter)
 * 
 * Modes:
 *   - counting: Just count events (sample_period ignored)
 *   - sampling: Sample every N event occurrences
 *   - sampling_freq: Sample at N Hz frequency (time-based)
 * 
 * Type:
 *   - fixed: Always included regardless of PMC_EVENT_INDICES (uses fixed counters)
 *   - raw: Included only if index is in PMC_EVENT_INDICES (uses programmable counters)
 * 
 * Event Selection (REQUIRED):
 *   You MUST set PMC_EVENT_INDICES environment variable to select events:
 *   PMC_EVENT_INDICES="0,1,2,3" ./my_program      # Events 0-3 + all fixed
 *   PMC_EVENT_INDICES="0,1,2,3,24,25" ./my_program # Same (fixed always included anyway)
 *   
 *   If PMC_EVENT_INDICES is not set, measurement will fail with an error.
 * 
 * @param label Identifier for this measurement (e.g., function name)
 * @param csv_path Path to CSV file (NULL or empty for default "pmc_events.csv")
 * @return Handle for this measurement session
 */
pmc_multi_handle_t* pmc_measure_begin_csv(const char *label, const char *csv_path);

/**
 * End measurement session and export results to JSON.
 * This is the ONLY function needed at function exit.
 * 
 * @param handle Handle from pmc_measure_begin_csv
 * @param report If true, export to JSON (always true in instrumentation mode)
 */
void pmc_measure_end(pmc_multi_handle_t *handle, int report);

/**
 * Export measurement results to JSON file.
 * 
 * @param handle Handle from pmc_measure_begin_csv
 * @param json_path Path to output JSON file
 * @return 0 on success, -1 on failure
 */
int pmc_export_json(pmc_multi_handle_t *handle, const char *json_path);

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

