/*
 * PMC Library Implementation
 */

#define _GNU_SOURCE
#include "pmc.h"

#include <asm/unistd.h>
#include <linux/perf_event.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <dlfcn.h>
#include <stdarg.h>

#ifndef PAGE_SIZE
#  define PAGE_SIZE 4096
#endif

// ===== Thread-local error handling =====
static __thread char pmc_error_buf[256] = {0};

static void pmc_set_error(const char *fmt, ...) {
    /*
    If called as:
    pmc_set_error("Failed to open fd=%d, error=%s", fd, strerror(errno));

    vsnprintf formats it into pmc_error_buf:
    pmc_error_buf = "Failed to open fd=3, error=Permission denied"

    https://hackernoon.com/what-is-va_list-in-c-exploring-the-secrets-of-ft_printf
    */

    va_list args; // an iterator over the ... arguments.
    va_start(args, fmt); // Start reading arguments right after the fmt parameter
    vsnprintf(pmc_error_buf, sizeof(pmc_error_buf), fmt, args);
    va_end(args); // Cleans up the va_list (required by C standard).
}

const char* pmc_get_error(void) {
    return pmc_error_buf;
}

// ===== Ring buffer for sampling =====
struct ring {
    void *base;
    size_t mmap_len;
    size_t data_size;
    struct perf_event_mmap_page *meta;
};

// ===== PMC Context =====
struct pmc_ctx {
    pmc_config_t config;
    int fd;
    struct ring ring;
    int is_started;
};

// ===== Multi-Event Handle =====
struct pmc_multi_handle {
    const char *label;
    pmc_ctx_t **contexts;          // Array of individual PMC contexts
    pmc_event_request_t *requests; // Copy of event requests
    size_t num_events;
    int all_started;
};

// ===== Event configuration table =====
typedef struct {
    pmc_event_type_t type;
    const char *name;
    uint32_t event;
    uint32_t umask;
    int is_raw;  // 1 for raw events, 0 for generic
} event_config_entry_t;

static const event_config_entry_t event_table[] = {
    { PMC_EVENT_NEAR_CALL,          "BR_INST_RETIRED.NEAR_CALL",      0xC4, 0x02, 1 },
    { PMC_EVENT_CONDITIONAL_BRANCH, "BR_INST_RETIRED.CONDITIONAL",    0xC4, 0x01, 1 },
    { PMC_EVENT_BRANCH_MISPREDICT,  "BR_MISP_RETIRED.ALL_BRANCHES",   0xC5, 0x00, 1 },
    { PMC_EVENT_L1_DCACHE_MISS,     "MEM_LOAD_RETIRED.L1_MISS",       0xD1, 0x08, 1 },
    { PMC_EVENT_L1_DCACHE_HIT,      "MEM_LOAD_RETIRED.L1_HIT",        0xD1, 0x01, 1 },
    { PMC_EVENT_CYCLES,             "CPU_CLK_UNHALTED.THREAD",        0x3C, 0x00, 1 },
    { PMC_EVENT_INSTRUCTIONS,       "INST_RETIRED.ANY",               0xC0, 0x00, 1 },
};

// ===== Helper functions =====
static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu,
                            int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

static inline void rb_barrier(void) {
    __sync_synchronize();
}

static const event_config_entry_t* get_event_config(pmc_event_type_t event) {
    for (size_t i = 0; i < sizeof(event_table) / sizeof(event_table[0]); i++) {
        if (event_table[i].type == event) {
            return &event_table[i];
        }
    }
    return NULL;
}

const char* pmc_event_name(pmc_event_type_t event) {
    const event_config_entry_t *entry = get_event_config(event);
    return entry ? entry->name : "UNKNOWN";
}

// ===== Ring buffer operations =====
static int ring_mmap(int fd, struct ring *r, unsigned data_pages_pow2) {
    if (data_pages_pow2 < 1) data_pages_pow2 = 1;
    if (data_pages_pow2 > 16) data_pages_pow2 = 16;  // Cap at 256MB
    
    size_t data_pages = 1ULL << data_pages_pow2;
    size_t data_sz    = data_pages * PAGE_SIZE;
    size_t mmap_len   = PAGE_SIZE + data_sz;

    void *base = mmap(NULL, mmap_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        return -1;
    }

    r->base      = base;
    r->mmap_len  = mmap_len;
    r->data_size = data_sz;
    r->meta      = (struct perf_event_mmap_page *)base;
    return 0;
}

static void ring_unmap(struct ring *r) {
    if (r->base && r->base != MAP_FAILED) {
        munmap(r->base, r->mmap_len);
    }
    memset(r, 0, sizeof(*r));
}

static void ring_read_bytes(struct ring *r, uint64_t *tail, void *dst, size_t len) {
    char *data = (char *)r->base + PAGE_SIZE;
    size_t mask = r->data_size - 1;
    size_t pos  = (*tail) & mask;

    if (pos + len <= r->data_size) {
        memcpy(dst, data + pos, len);
    } else {
        size_t first = r->data_size - pos;
        memcpy(dst, data + pos, first);
        memcpy((char*)dst + first, data, len - first);
    }
    *tail += len;
}

// ===== Public API Implementation =====

pmc_config_t pmc_get_default_config(pmc_event_type_t event, pmc_mode_t mode, uint64_t sample_period) {
    pmc_config_t config = {
        .event = event,
        .mode = mode,
        .sample_period = sample_period,
        .exclude_kernel = 1,
        .exclude_hv = 1,
        .precise_ip = 0, // 
        .ring_buffer_pages = 7,  // 128 pages = 512KB
    };
    return config;
}

pmc_ctx_t* pmc_create(const pmc_config_t *config) {
    if (!config) {
        pmc_set_error("NULL configuration");
        return NULL;
    }


    // event_cfg has event number, umask, and if it is raw or not based on the event type provided by the user
    const event_config_entry_t *event_cfg = get_event_config(config->event);
    if (!event_cfg) {
        pmc_set_error("Unknown event type: %d", config->event);
        return NULL;
    }

    // allocate the a PMC event context
    pmc_ctx_t *ctx = calloc(1, sizeof(pmc_ctx_t));
    if (!ctx) {
        pmc_set_error("Failed to allocate context");
        return NULL;
    }

    ctx->config = *config;
    ctx->fd = -1;
    ctx->is_started = 0;

    // Setup perf_event_attr
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.size = sizeof(attr);
    
    if (event_cfg->is_raw) {
        attr.type = PERF_TYPE_RAW;
        attr.config = event_cfg->event | ((uint64_t)event_cfg->umask << 8);
    } else {
        // Generic events (not yet implemented - placeholder)
        pmc_set_error("Generic events not yet implemented");
        free(ctx);
        return NULL;
    }

    attr.disabled = 1;
    attr.exclude_kernel = config->exclude_kernel;
    attr.exclude_hv = config->exclude_hv;
    attr.exclude_idle = 1;
    attr.precise_ip = config->precise_ip;

    // Configure sampling if needed
    if (config->mode == PMC_MODE_SAMPLING) {
        attr.sample_period = config->sample_period;
        attr.freq = 0; // sample only based on event count, not frequency
        attr.sample_type = PERF_SAMPLE_IP | PERF_SAMPLE_TID | PERF_SAMPLE_TIME;
        attr.wakeup_events = 1;
    }

    // Open perf event
    ctx->fd = perf_event_open(&attr, 0, -1, -1, 0);
    if (ctx->fd == -1) {
        pmc_set_error("perf_event_open failed: %s (errno=%d). "
                     "Tip: check perf_event_paranoid or CAP_SYS_ADMIN.", 
                     strerror(errno), errno);
        free(ctx);
        return NULL;
    }

    // Setup ring buffer for sampling
    if (config->mode == PMC_MODE_SAMPLING) {
        if (ring_mmap(ctx->fd, &ctx->ring, config->ring_buffer_pages) == -1) {
            pmc_set_error("Failed to mmap ring buffer: %s", strerror(errno));
            close(ctx->fd);
            free(ctx);
            return NULL;
        }
    }

    return ctx;
}

int pmc_start(pmc_ctx_t *ctx) {
    if (!ctx || ctx->fd == -1) {
        pmc_set_error("Invalid context");
        return -1;
    }

    if (ioctl(ctx->fd, PERF_EVENT_IOC_RESET, 0) == -1) {
        pmc_set_error("PERF_EVENT_IOC_RESET failed: %s", strerror(errno));
        return -1;
    }

    if (ioctl(ctx->fd, PERF_EVENT_IOC_ENABLE, 0) == -1) {
        pmc_set_error("PERF_EVENT_IOC_ENABLE failed: %s", strerror(errno));
        return -1;
    }

    ctx->is_started = 1;
    return 0;
}

int pmc_stop(pmc_ctx_t *ctx) {
    if (!ctx || ctx->fd == -1) {
        pmc_set_error("Invalid context");
        return -1;
    }

    if (ioctl(ctx->fd, PERF_EVENT_IOC_DISABLE, 0) == -1) {
        pmc_set_error("PERF_EVENT_IOC_DISABLE failed: %s", strerror(errno));
        return -1;
    }

    ctx->is_started = 0;
    return 0;
}

int pmc_read_count(pmc_ctx_t *ctx, uint64_t *count) {
    if (!ctx || ctx->fd == -1 || !count) {
        pmc_set_error("Invalid context or NULL count pointer");
        return -1;
    }

    if (read(ctx->fd, count, sizeof(*count)) != sizeof(*count)) {
        pmc_set_error("Failed to read counter: %s", strerror(errno));
        return -1;
    }

    return 0;
}

int pmc_read_samples(pmc_ctx_t *ctx, pmc_sample_t **samples, 
                     size_t *num_samples, size_t max_samples) {
    if (!ctx || ctx->fd == -1 || !samples || !num_samples) {
        pmc_set_error("Invalid parameters");
        return -1;
    }

    if (ctx->config.mode != PMC_MODE_SAMPLING) {
        pmc_set_error("Context not in sampling mode");
        return -1;
    }

    *samples = NULL;
    *num_samples = 0;

    struct ring *r = &ctx->ring;
    uint64_t data_head = r->meta->data_head;
    rb_barrier();
    uint64_t data_tail = r->meta->data_tail;

    // First pass: count samples
    size_t count = 0;
    uint64_t temp_tail = data_tail;
    
    while (temp_tail < data_head) {
        struct perf_event_header h;
        ring_read_bytes(r, &temp_tail, &h, sizeof(h));
        
        if (h.type == PERF_RECORD_SAMPLE) {
            count++;
        }
        
        // Skip the payload
        size_t rem = h.size - sizeof(h);
        temp_tail += rem;
    }

    if (count == 0) {
        return 0;  // No samples
    }

    // Limit samples if requested
    if (max_samples > 0 && count > max_samples) {
        count = max_samples;
    }

    // Allocate sample array
    *samples = malloc(count * sizeof(pmc_sample_t));
    if (!*samples) {
        pmc_set_error("Failed to allocate sample array");
        return -1;
    }

    // Second pass: read samples
    size_t idx = 0;
    temp_tail = data_tail;
    
    while (temp_tail < data_head && idx < count) {
        struct perf_event_header h;
        ring_read_bytes(r, &temp_tail, &h, sizeof(h));
        
        size_t rem = h.size - sizeof(h);
        
        if (h.type == PERF_RECORD_SAMPLE) {
            pmc_sample_t *s = &(*samples)[idx];
            
            if (rem >= sizeof(uint64_t) + 2*sizeof(uint32_t) + sizeof(uint64_t)) {
                ring_read_bytes(r, &temp_tail, &s->ip, sizeof(s->ip));
                ring_read_bytes(r, &temp_tail, &s->pid, sizeof(s->pid));
                ring_read_bytes(r, &temp_tail, &s->tid, sizeof(s->tid));
                ring_read_bytes(r, &temp_tail, &s->time, sizeof(s->time));
                idx++;
                rem -= sizeof(uint64_t) + 2*sizeof(uint32_t) + sizeof(uint64_t);
            }
        }
        
        // Skip remaining payload
        temp_tail += rem;
    }

    *num_samples = idx;

    // Update ring buffer tail
    rb_barrier();
    r->meta->data_tail = data_head;
    rb_barrier();

    return 0;
}

void pmc_print_sample(const pmc_sample_t *sample, int index) {
    if (!sample) return;

    printf("SAMPLE #%d  pid=%u tid=%u time=%llu\n", 
           index, sample->pid, sample->tid, 
           (unsigned long long)sample->time);

    // Try to resolve symbol
    Dl_info info;
    if (dladdr((void*)(uintptr_t)sample->ip, &info) && info.dli_sname) {
        long off = (long)((uintptr_t)sample->ip - (uintptr_t)info.dli_saddr);
        printf("  ip=0x%llx  %s+0x%lx  (%s)\n",
               (unsigned long long)sample->ip, info.dli_sname, off,
               info.dli_fname ? info.dli_fname : "?");
    } else {
        printf("  ip=0x%llx  (no symbol)\n", (unsigned long long)sample->ip);
    }
}

void pmc_destroy(pmc_ctx_t *ctx) {
    if (!ctx) return;

    if (ctx->config.mode == PMC_MODE_SAMPLING) {
        ring_unmap(&ctx->ring);
    }

    if (ctx->fd != -1) {
        close(ctx->fd);
    }

    free(ctx);
}

// ===== CSV Parsing Helpers =====

// Parse event name string to enum
static int parse_event_name(const char *name, pmc_event_type_t *event) {
    if (strcmp(name, "PMC_EVENT_NEAR_CALL") == 0) {
        *event = PMC_EVENT_NEAR_CALL;
    } else if (strcmp(name, "PMC_EVENT_CONDITIONAL_BRANCH") == 0) {
        *event = PMC_EVENT_CONDITIONAL_BRANCH;
    } else if (strcmp(name, "PMC_EVENT_BRANCH_MISPREDICT") == 0) {
        *event = PMC_EVENT_BRANCH_MISPREDICT;
    } else if (strcmp(name, "PMC_EVENT_L1_DCACHE_MISS") == 0) {
        *event = PMC_EVENT_L1_DCACHE_MISS;
    } else if (strcmp(name, "PMC_EVENT_L1_DCACHE_HIT") == 0) {
        *event = PMC_EVENT_L1_DCACHE_HIT;
    } else if (strcmp(name, "PMC_EVENT_CYCLES") == 0) {
        *event = PMC_EVENT_CYCLES;
    } else if (strcmp(name, "PMC_EVENT_INSTRUCTIONS") == 0) {
        *event = PMC_EVENT_INSTRUCTIONS;
    } else {
        return -1;
    }
    return 0;
}

// Parse mode string to enum
static int parse_mode(const char *mode, pmc_mode_t *out_mode) {
    if (strcmp(mode, "counting") == 0 || strcmp(mode, "COUNTING") == 0) {
        *out_mode = PMC_MODE_COUNTING;
    } else if (strcmp(mode, "sampling") == 0 || strcmp(mode, "SAMPLING") == 0) {
        *out_mode = PMC_MODE_SAMPLING;
    } else {
        return -1;
    }
    return 0;
}

// Trim whitespace from string (in-place)
static void trim_whitespace(char *str) {
    if (!str) return;
    
    // Trim leading
    char *start = str;
    while (*start == ' ' || *start == '\t' || *start == '\r' || *start == '\n') {
        start++;
    }
    
    // Trim trailing
    char *end = start + strlen(start) - 1;
    while (end > start && (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n')) {
        *end = '\0';
        end--;
    }
    
    // Move trimmed string to beginning
    if (start != str) {
        memmove(str, start, strlen(start) + 1);
    }
}

// Load events from CSV file
static int load_events_from_csv(const char *csv_path, 
                                pmc_event_request_t **events_out,
                                size_t *num_events_out) {
    FILE *fp = fopen(csv_path, "r");
    if (!fp) {
        pmc_set_error("Failed to open CSV file: %s", csv_path);
        return -1;
    }
    
    // First pass: count events (skip header)
    char line[256];
    size_t count = 0;
    int header_skipped = 0;
    
    while (fgets(line, sizeof(line), fp)) {
        trim_whitespace(line);
        if (strlen(line) == 0 || line[0] == '#') continue;  // Skip empty/comment lines
        if (!header_skipped) {
            header_skipped = 1;
            continue;  // Skip header line
        }
        count++;
    }
    
    if (count == 0) {
        pmc_set_error("No events found in CSV file");
        fclose(fp);
        return -1;
    }
    
    // Allocate event array
    pmc_event_request_t *events = calloc(count, sizeof(pmc_event_request_t));
    if (!events) {
        pmc_set_error("Failed to allocate event array");
        fclose(fp);
        return -1;
    }
    
    // Second pass: parse events
    rewind(fp);
    header_skipped = 0;
    size_t idx = 0;
    
    while (fgets(line, sizeof(line), fp) && idx < count) {
        trim_whitespace(line);
        if (strlen(line) == 0 || line[0] == '#') continue;
        if (!header_skipped) {
            header_skipped = 1;
            continue;
        }
        
        // Parse CSV: event_name,mode,sample_period
        char event_name[64] = {0};
        char mode_str[32] = {0};
        uint64_t sample_period = 0;
        
        // Simple CSV parsing (doesn't handle quoted strings, but we don't need it)
        char *token1 = strtok(line, ",");
        char *token2 = strtok(NULL, ",");
        char *token3 = strtok(NULL, ",");
        
        if (!token1 || !token2 || !token3) {
            pmc_set_error("Malformed CSV line at event %zu", idx + 1);
            free(events);
            fclose(fp);
            return -1;
        }
        
        strncpy(event_name, token1, sizeof(event_name) - 1);
        strncpy(mode_str, token2, sizeof(mode_str) - 1);
        sample_period = strtoull(token3, NULL, 10);
        
        trim_whitespace(event_name);
        trim_whitespace(mode_str);
        
        // Parse event type
        if (parse_event_name(event_name, &events[idx].event) != 0) {
            pmc_set_error("Unknown event name: %s", event_name);
            free(events);
            fclose(fp);
            return -1;
        }
        
        // Parse mode
        if (parse_mode(mode_str, &events[idx].mode) != 0) {
            pmc_set_error("Unknown mode: %s", mode_str);
            free(events);
            fclose(fp);
            return -1;
        }
        
        events[idx].sample_period = sample_period;
        events[idx].precise_ip = 0;  // Default to 0 for compatibility
        
        idx++;
    }
    
    fclose(fp);
    
    *events_out = events;
    *num_events_out = idx;
    return 0;
}

// ===== Multi-Event API Implementation =====

// Helper function to initialize measurement from event array (internal)
static pmc_multi_handle_t* pmc_measure_begin_internal(const char *label,
                                                       const pmc_event_request_t *events,
                                                       size_t num_events,
                                                       int should_free_events) {
    if (!label || !events || num_events == 0) {
        pmc_set_error("Invalid parameters to pmc_measure_begin");
        return NULL;
    }
    
    // Allocate multi-handle
    pmc_multi_handle_t *handle = calloc(1, sizeof(pmc_multi_handle_t));
    if (!handle) {
        pmc_set_error("Failed to allocate multi-handle");
        return NULL;
    }
    
    handle->label = label;
    handle->num_events = num_events;
    handle->all_started = 0;
    
    // Allocate arrays
    handle->contexts = calloc(num_events, sizeof(pmc_ctx_t*));
    handle->requests = calloc(num_events, sizeof(pmc_event_request_t));
    
    if (!handle->contexts || !handle->requests) {
        pmc_set_error("Failed to allocate arrays");
        free(handle->contexts);
        free(handle->requests);
        free(handle);
        if (should_free_events) free((void*)events);
        return NULL;
    }
    
    // Copy requests
    memcpy(handle->requests, events, num_events * sizeof(pmc_event_request_t));
    
    // Free the input events array if needed (came from CSV load)
    if (should_free_events) {
        free((void*)events);
    }
    
    // Create individual PMC contexts for each event
    for (size_t i = 0; i < num_events; i++) {
        pmc_config_t config = pmc_get_default_config(
            handle->requests[i].event,
            handle->requests[i].mode,
            handle->requests[i].sample_period
        );
        config.precise_ip = handle->requests[i].precise_ip;
        
        handle->contexts[i] = pmc_create(&config);
        if (!handle->contexts[i]) {
            // Cleanup on failure
            for (size_t j = 0; j < i; j++) {
                if (handle->contexts[j]) {
                    pmc_destroy(handle->contexts[j]);
                }
            }
            free(handle->contexts);
            free(handle->requests);
            free(handle);
            return NULL;
        }
    }
    
    // Start all measurements
    for (size_t i = 0; i < num_events; i++) {
        if (pmc_start(handle->contexts[i]) != 0) {
            // Continue with other events even if one fails
            fprintf(stderr, "Warning: Failed to start event %s\n", 
                    pmc_event_name(handle->requests[i].event));
        }
    }
    
    handle->all_started = 1;
    return handle;
}

// Original API - manually specify events array
pmc_multi_handle_t* pmc_measure_begin(const char *label, 
                                       const pmc_event_request_t *events,
                                       size_t num_events) {
    return pmc_measure_begin_internal(label, events, num_events, 0);
}

// Simplified API - load events from CSV file
pmc_multi_handle_t* pmc_measure_begin_csv(const char *label, const char *csv_path) {
    if (!label) {
        pmc_set_error("NULL label");
        return NULL;
    }
    
    // Use default CSV path if not provided
    const char *path = csv_path ? csv_path : "pmc_events.csv";
    
    // Load events from CSV
    pmc_event_request_t *events = NULL;
    size_t num_events = 0;
    
    if (load_events_from_csv(path, &events, &num_events) != 0) {
        return NULL;  // Error already set by load_events_from_csv
    }
    
    // Initialize measurement (will free events array)
    return pmc_measure_begin_internal(label, events, num_events, 1);
}

void pmc_measure_end(pmc_multi_handle_t *handle, int report) {
    if (!handle) return;
    
    // Stop all measurements
    if (handle->all_started) {
        for (size_t i = 0; i < handle->num_events; i++) {
            if (handle->contexts[i]) {
                pmc_stop(handle->contexts[i]);
            }
        }
    }
    
    // Report if requested
    if (report) {
        pmc_report_all(handle);
    }
    
    // Cleanup
    for (size_t i = 0; i < handle->num_events; i++) {
        if (handle->contexts[i]) {
            pmc_destroy(handle->contexts[i]);
        }
    }
    
    free(handle->contexts);
    free(handle->requests);
    free(handle);
}

void pmc_report_all(pmc_multi_handle_t *handle) {
    if (!handle) return;
    
    printf("\n========== PMC Report: %s ==========\n", handle->label);
    
    for (size_t i = 0; i < handle->num_events; i++) {
        pmc_ctx_t *ctx = handle->contexts[i];
        if (!ctx) continue;
        
        const char *event_name = pmc_event_name(handle->requests[i].event);
        
        if (handle->requests[i].mode == PMC_MODE_COUNTING) {
            // Counting mode - just print count
            uint64_t count = 0;
            if (pmc_read_count(ctx, &count) == 0) {
                printf("  [%zu] %s: %llu\n", 
                       i + 1,
                       event_name,
                       (unsigned long long)count);
            }
        } else {
            // Sampling mode - print summary
            pmc_sample_t *samples = NULL;
            size_t num_samples = 0;
            
            if (pmc_read_samples(ctx, &samples, &num_samples, 0) == 0) {
                uint64_t count = 0;
                pmc_read_count(ctx, &count);
                
                printf("  [%zu] %s:\n", i + 1, event_name);
                printf("      Total count: %llu\n", (unsigned long long)count);
                printf("      Samples: %zu (period: %llu)\n", 
                       num_samples,
                       (unsigned long long)handle->requests[i].sample_period);
                
                if (num_samples > 0) {
                    printf("      Events per sample: %.1f\n", 
                           (double)count / num_samples);
                    
                    // Print first 3 sample IPs
                    printf("      First samples: ");
                    size_t show = num_samples < 3 ? num_samples : 3;
                    for (size_t j = 0; j < show; j++) {
                        printf("0x%llx", (unsigned long long)samples[j].ip);
                        if (j < show - 1) printf(", ");
                    }
                    if (num_samples > 3) {
                        printf(" ... (+%zu more)", num_samples - 3);
                    }
                    printf("\n");
                }
                
                free(samples);
            }
        }
    }
    
    printf("==========================================\n\n");
}

int pmc_get_count(pmc_multi_handle_t *handle, pmc_event_type_t event, uint64_t *count) {
    if (!handle || !count) {
        pmc_set_error("Invalid parameters");
        return -1;
    }
    
    // Find the context for this event
    for (size_t i = 0; i < handle->num_events; i++) {
        if (handle->requests[i].event == event) {
            return pmc_read_count(handle->contexts[i], count);
        }
    }
    
    pmc_set_error("Event not found in measurement set");
    return -1;
}

int pmc_get_samples(pmc_multi_handle_t *handle, pmc_event_type_t event,
                    pmc_sample_t **samples, size_t *num_samples) {
    if (!handle || !samples || !num_samples) {
        pmc_set_error("Invalid parameters");
        return -1;
    }
    
    // Find the context for this event
    for (size_t i = 0; i < handle->num_events; i++) {
        if (handle->requests[i].event == event) {
            if (handle->requests[i].mode != PMC_MODE_SAMPLING) {
                pmc_set_error("Event is not in sampling mode");
                return -1;
            }
            return pmc_read_samples(handle->contexts[i], samples, num_samples, 0);
        }
    }
    
    pmc_set_error("Event not found in measurement set");
    return -1;
}

