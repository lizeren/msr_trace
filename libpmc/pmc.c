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
#include <time.h>

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
// Read results from perf_event_open()
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


// ===== Helper functions =====
static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu,
                            int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

static inline void rb_barrier(void) {
    __sync_synchronize();
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

pmc_config_t pmc_get_default_config(const char *event_name, pmc_mode_t mode, uint64_t sample_period) {
    pmc_config_t config = {
        .event = event_name,
        .mode = mode,
        .sample_period = sample_period,
        .exclude_kernel = 1, // exclude kernel events
        .exclude_hv = 1, // exclude hypervisor events
        .precise_ip = 0, // most relaxed precision level
        .ring_buffer_pages = 7,  // 128 pages = 512KB
        .event_number = 0,  // To be set by caller
        .umask = 0,  // To be set by caller
        .is_raw = 0,  // To be set by caller
        .pinned = 0,  // To be set by caller
    };
    return config;
}

pmc_ctx_t* pmc_create(const pmc_config_t *config) {
    if (!config) {
        pmc_set_error("NULL configuration");
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
    
    if (config->is_raw) {
        // Raw PMU events (programmable counters)
        attr.type = PERF_TYPE_RAW;
        attr.config = config->event_number | ((uint64_t)config->umask << 8);
    } else {
        // Generic hardware events (uses fixed counters: IA32_FIXED_CTR0/1/2)
        attr.type = PERF_TYPE_HARDWARE;
        if(config->event_number == 0x3C) {
            attr.config = PERF_COUNT_HW_CPU_CYCLES;
        } else if(config->event_number == 0xC0) {
            attr.config = PERF_COUNT_HW_INSTRUCTIONS;
        } else {
            pmc_set_error("Unknown event number: %x", config->event_number);
            free(ctx);
            return NULL;
        }
    }

    attr.disabled = 1;
    attr.exclude_kernel = config->exclude_kernel;
    attr.exclude_hv = config->exclude_hv;
    attr.exclude_idle = 1;
    attr.precise_ip = config->precise_ip;
    attr.pinned = 1;
    // Configure sampling if needed
    if (config->mode == PMC_MODE_SAMPLING) {
        // Event-based sampling: sample every N events
        attr.sample_period = config->sample_period;
        attr.freq = 0;
        attr.sample_type = PERF_SAMPLE_IP | PERF_SAMPLE_TID | PERF_SAMPLE_TIME | PERF_SAMPLE_READ;
        // attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
        attr.wakeup_events = 1;
    } else if (config->mode == PMC_MODE_SAMPLING_FREQ) {
        // Frequency-based sampling: sample at N Hz
        attr.sample_freq = config->sample_period;  // Reuse field for frequency (Hz)
        attr.freq = 1;  // Enable frequency mode
        attr.sample_type = PERF_SAMPLE_IP | PERF_SAMPLE_TID | PERF_SAMPLE_TIME | PERF_SAMPLE_READ;
        // attr.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
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

    // Setup ring buffer for sampling modes
    if (config->mode == PMC_MODE_SAMPLING || config->mode == PMC_MODE_SAMPLING_FREQ) {
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
    // When sampling with read_format set, we get: value, time_enabled, time_running
    // Just read all of it and extract the value (first field)
    // if (ctx->config.mode == PMC_MODE_SAMPLING || ctx->config.mode == PMC_MODE_SAMPLING_FREQ) {
    //     uint64_t read_data[3];  // value, time_enabled, time_running
        
    //     if (read(ctx->fd, read_data, sizeof(read_data)) != sizeof(read_data)) {
    //         pmc_set_error("Failed to read counter: %s", strerror(errno));
    //         return -1;
    //     }
    //     *count = read_data[0];  // Extract just the value (first field)
    // } else {
    //     if (read(ctx->fd, count, sizeof(*count)) != sizeof(*count)) {
    //         pmc_set_error("Failed to read counter: %s", strerror(errno));
    //         return -1;
    //     }
    // }

    return 0;
}

int pmc_read_samples(pmc_ctx_t *ctx, pmc_sample_t **samples, 
                     size_t *num_samples, size_t max_samples) {
    if (!ctx || ctx->fd == -1 || !samples || !num_samples) {
        pmc_set_error("Invalid parameters");
        return -1;
    }

    if (ctx->config.mode != PMC_MODE_SAMPLING && ctx->config.mode != PMC_MODE_SAMPLING_FREQ) {
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
            
            // Expected layout: IP, PID, TID, TIME, READ (value, time_enabled, time_running)
            // size_t expected_size = sizeof(uint64_t) + 2*sizeof(uint32_t) + sizeof(uint64_t) + 3*sizeof(uint64_t);
            
            // Expected layout: IP, PID, TID, TIME, READ (value)
            size_t expected_size = sizeof(uint64_t) + 2*sizeof(uint32_t) + sizeof(uint64_t) + sizeof(uint64_t);

            if (rem >= expected_size) {
                ring_read_bytes(r, &temp_tail, &s->ip, sizeof(s->ip));
                ring_read_bytes(r, &temp_tail, &s->pid, sizeof(s->pid));
                ring_read_bytes(r, &temp_tail, &s->tid, sizeof(s->tid));
                ring_read_bytes(r, &temp_tail, &s->time, sizeof(s->time));
                
                // Read the counter value (PERF_SAMPLE_READ format)
                uint64_t value, time_enabled, time_running;
                ring_read_bytes(r, &temp_tail, &value, sizeof(value));
                // ring_read_bytes(r, &temp_tail, &time_enabled, sizeof(time_enabled));
                // ring_read_bytes(r, &temp_tail, &time_running, sizeof(time_running));
                s->count = value;  // Store the counter value at this sample point
                
                idx++;
                rem -= expected_size;
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

    printf("SAMPLE #%d  pid=%u tid=%u time=%llu count=%llu\n", 
           index, sample->pid, sample->tid, 
           (unsigned long long)sample->time,
           (unsigned long long)sample->count);

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

    if (ctx->config.mode == PMC_MODE_SAMPLING || ctx->config.mode == PMC_MODE_SAMPLING_FREQ) {
        ring_unmap(&ctx->ring);
    }

    if (ctx->fd != -1) {
        close(ctx->fd);
    }

    free(ctx);
}

// ===== CSV Parsing Helpers =====

// Parse mode string to enum
static int parse_mode(const char *mode, pmc_mode_t *out_mode) {
    if (strcmp(mode, "counting") == 0 || strcmp(mode, "COUNTING") == 0) {
        *out_mode = PMC_MODE_COUNTING;
    } else if (strcmp(mode, "sampling") == 0 || strcmp(mode, "SAMPLING") == 0) {
        *out_mode = PMC_MODE_SAMPLING;
    } else if (strcmp(mode, "sampling_freq") == 0 || strcmp(mode, "SAMPLING_FREQ") == 0) {
        *out_mode = PMC_MODE_SAMPLING_FREQ;
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

// Helper: Check if index is in the selection list
static int is_index_selected(int event_index, const int *selected_indices, size_t num_selected) {
    for (size_t i = 0; i < num_selected; i++) {
        if (selected_indices[i] == event_index) {
            return 1;
        }
    }
    return 0;
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
    
    // Check for event index (PMC_EVENT_INDICES environment variable)
    int selected_indices[100];
    size_t num_selected = 0;
    const char *indices_env = getenv("PMC_EVENT_INDICES");
    
    if (!indices_env || strlen(indices_env) == 0) {
        pmc_set_error("PMC_EVENT_INDICES not set. Please specify event indices from the CSV file.\n"
                     "Example: PMC_EVENT_INDICES=\"0,1,2,3\" ./my_program");
        fclose(fp);
        return -1;
    }
    
    // Parse comma-separated list: "0,1,2,3"
    char indices_copy[256];
    strncpy(indices_copy, indices_env, sizeof(indices_copy) - 1);
    indices_copy[sizeof(indices_copy) - 1] = '\0';
    
    char *token = strtok(indices_copy, ","); // get the first token
    while (token && num_selected < 100) {
        trim_whitespace(token); // make sure the string doesn't have any whitespace
        selected_indices[num_selected++] = atoi(token); // store the token in an integer array
        token = strtok(NULL, ","); // get the next token
    }
    
    fprintf(stderr, "PMC: Measuring %zu event indices", num_selected);
    for (size_t i = 0; i < num_selected && i < 10; i++) {
        fprintf(stderr, "%s%d", i == 0 ? ": " : ",", selected_indices[i]);
    }
    if (num_selected > 10) {
        fprintf(stderr, "...");
    }
    fprintf(stderr, "\n");
    
    // First pass: count events that match filter (skip header)
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
        
        // Quick parse to get index and type
        char line_copy[256];
        strncpy(line_copy, line, sizeof(line_copy) - 1);
        char *idx_token = strtok(line_copy, ",");
        if (!idx_token) continue;
        
        trim_whitespace(idx_token);
        int event_idx = atoi(idx_token);
        
        // Skip to type column (7th column: index,event_name,event_number,umask,mode,sample_period,type)
        strtok(NULL, ",");  // event name
        strtok(NULL, ",");  // event_number
        strtok(NULL, ",");  // umask
        strtok(NULL, ",");  // mode
        strtok(NULL, ",");  // sample_period
        strtok(NULL, ",");  // type
        
        // Check if index is in filter (applies to both raw and fixed)
        if (is_index_selected(event_idx, selected_indices, num_selected)) {
            count++;
        }
    }
    
    if (count == 0) {
        pmc_set_error("No events found in CSV file (after filtering)");
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
        
        // Parse CSV: index,event_name,event_number,umask,mode,sample_period,type
        char index_str[32] = {0};
        char event_name_buf[128] = {0};
        char event_number_str[32] = {0};
        char umask_str[32] = {0};
        char mode_str[32] = {0};
        char type_str[32] = {0};
        uint64_t sample_period = 0;
        
        // Simple CSV parsing: index,event_name,event_number,umask,mode,sample_period,type
        char *token1 = strtok(line, ",");  // index
        char *token2 = strtok(NULL, ",");  // event_name
        char *token3 = strtok(NULL, ",");  // event_number
        char *token4 = strtok(NULL, ",");  // umask
        char *token5 = strtok(NULL, ",");  // mode
        char *token6 = strtok(NULL, ",");  // sample_period
        char *token7 = strtok(NULL, ",");  // type
        
        if (!token1 || !token2 || !token3 || !token4 || !token5 || !token6 || !token7) {
            pmc_set_error("Malformed CSV line at event %zu", idx + 1);
            for (size_t j = 0; j < idx; j++) {
                free((void*)events[j].event);
            }
            free(events);
            fclose(fp);
            return -1;
        }
        
        strncpy(index_str, token1, sizeof(index_str) - 1);
        strncpy(event_name_buf, token2, sizeof(event_name_buf) - 1);
        strncpy(event_number_str, token3, sizeof(event_number_str) - 1);
        strncpy(umask_str, token4, sizeof(umask_str) - 1);
        strncpy(mode_str, token5, sizeof(mode_str) - 1);
        sample_period = strtoull(token6, NULL, 10);
        strncpy(type_str, token7, sizeof(type_str) - 1);
        
        trim_whitespace(index_str);
        trim_whitespace(event_name_buf);
        trim_whitespace(event_number_str);
        trim_whitespace(umask_str);
        trim_whitespace(mode_str);
        trim_whitespace(type_str);
        
        int event_idx = atoi(index_str);
        
        // Apply filter: check if index is in selection (applies to both raw and fixed)
        if (!is_index_selected(event_idx, selected_indices, num_selected)) {
            continue;  // Skip this event
        }
        
        // Duplicate event name string (will be freed when handle is destroyed)
        events[idx].event = strdup(event_name_buf);
        if (!events[idx].event) {
            pmc_set_error("Failed to allocate event name");
            for (size_t j = 0; j < idx; j++) {
                free((void*)events[j].event);
            }
            free(events);
            fclose(fp);
            return -1;
        }
        
        // Parse mode
        if (parse_mode(mode_str, &events[idx].mode) != 0) {
            pmc_set_error("Unknown mode: %s", mode_str);
            for (size_t j = 0; j <= idx; j++) {
                free((void*)events[j].event);
            }
            free(events);
            fclose(fp);
            return -1;
        }
        
        // Parse event number and umask (support both hex and decimal)
        events[idx].event_number = (uint32_t)strtoul(event_number_str, NULL, 0);
        events[idx].umask = (uint32_t)strtoul(umask_str, NULL, 0);
        
        // Determine if raw or fixed
        if (strcmp(type_str, "fixed") == 0) {
            events[idx].is_raw = 0;  // Fixed counter
        } else if (strcmp(type_str, "raw") == 0) {
            events[idx].is_raw = 1;  // Raw programmable counter
        } else {
            pmc_set_error("Unknown type: %s (expected 'raw' or 'fixed')", type_str);
            for (size_t j = 0; j <= idx; j++) {
                free((void*)events[j].event);
            }
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
    
    handle->label = label; // name of the function being measured
    handle->num_events = num_events;
    handle->all_started = 0;
    
    // Allocate arrays
    handle->contexts = calloc(num_events, sizeof(pmc_ctx_t*)); // ctx store pmc_config_t, file descriptors, ring buffers
    handle->requests = calloc(num_events, sizeof(pmc_event_request_t));
    
    if (!handle->contexts || !handle->requests) {
        pmc_set_error("Failed to allocate arrays");
        free(handle->contexts);
        free(handle->requests);
        free(handle);
        if (should_free_events) free((void*)events);
        return NULL;
    }
    
    // Copy requests (shallow copy - event pointers are shared)
    memcpy(handle->requests, events, num_events * sizeof(pmc_event_request_t));
    
    // Free the input events array if needed (came from CSV load)
    // Note: We keep the event name strings alive since handle->requests now references them
    if (should_free_events) {
        free((void*)events);  // Free array, but NOT the event name strings
    }
    
    // Create individual PMC contexts for each event
    for (size_t i = 0; i < num_events; i++) {
        pmc_config_t config = pmc_get_default_config(
            handle->requests[i].event,
            handle->requests[i].mode,
            handle->requests[i].sample_period
        );
        config.precise_ip = handle->requests[i].precise_ip;
        config.event_number = handle->requests[i].event_number;
        config.umask = handle->requests[i].umask;
        config.is_raw = handle->requests[i].is_raw;
        
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
                    handle->requests[i].event);
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
        // Check for custom output filename
        const char *output_file = getenv("PMC_OUTPUT_FILE");
        if (!output_file) {
            output_file = "pmc_results.json";
        }
        
        pmc_export_json(handle, output_file);  // Export first to capture samples
        // pmc_report_all(handle);  // Report second (consumes samples from ring buffer)
    }
    
    // Cleanup
    for (size_t i = 0; i < handle->num_events; i++) {
        if (handle->contexts[i]) {
            pmc_destroy(handle->contexts[i]);
        }
        // Free event name strings (allocated by CSV parser)
        if (handle->requests[i].event) {
            free((void*)handle->requests[i].event);
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
        
        const char *event_name = handle->requests[i].event;
        
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
            // Sampling modes - print summary
            pmc_sample_t *samples = NULL;
            size_t num_samples = 0;
            
            if (pmc_read_samples(ctx, &samples, &num_samples, 0) == 0) {
                uint64_t count = 0;
                pmc_read_count(ctx, &count);
                
                printf("  [%zu] %s:\n", i + 1, event_name);
                printf("      Total count: %llu\n", (unsigned long long)count);
                
                if (handle->requests[i].mode == PMC_MODE_SAMPLING) {
                    printf("      Samples: %zu (period: %llu events)\n", 
                           num_samples,
                           (unsigned long long)handle->requests[i].sample_period);
                } else {
                    printf("      Samples: %zu (freq: %llu Hz)\n", 
                           num_samples,
                           (unsigned long long)handle->requests[i].sample_period);
                }
                
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

int pmc_get_count(pmc_multi_handle_t *handle, const char *event_name, uint64_t *count) {
    if (!handle || !event_name || !count) {
        pmc_set_error("Invalid parameters");
        return -1;
    }
    
    // Find the context for this event
    for (size_t i = 0; i < handle->num_events; i++) {
        if (strcmp(handle->requests[i].event, event_name) == 0) {
            return pmc_read_count(handle->contexts[i], count);
        }
    }
    
    pmc_set_error("Event not found in measurement set: %s", event_name);
    return -1;
}

int pmc_get_samples(pmc_multi_handle_t *handle, const char *event_name,
                    pmc_sample_t **samples, size_t *num_samples) {
    if (!handle || !event_name || !samples || !num_samples) {
        pmc_set_error("Invalid parameters");
        return -1;
    }
    
    // Find the context for this event
    for (size_t i = 0; i < handle->num_events; i++) {
        if (strcmp(handle->requests[i].event, event_name) == 0) {
            if (handle->requests[i].mode != PMC_MODE_SAMPLING && 
                handle->requests[i].mode != PMC_MODE_SAMPLING_FREQ) {
                pmc_set_error("Event is not in sampling mode: %s", event_name);
                return -1;
            }
            return pmc_read_samples(handle->contexts[i], samples, num_samples, 0);
        }
    }
    
    pmc_set_error("Event not found in measurement set: %s", event_name);
    return -1;
}

// ===== JSON Export =====

// Helper: Get mode as string
static const char* mode_to_string(pmc_mode_t mode) {
    switch (mode) {
        case PMC_MODE_COUNTING: return "counting";
        case PMC_MODE_SAMPLING: return "sampling";
        case PMC_MODE_SAMPLING_FREQ: return "sampling_freq";
        default: return "unknown";
    }
}

// Helper: Write JSON string with escaping
static void write_json_string(FILE *fp, const char *str) {
    fputc('"', fp);
    for (const char *p = str; *p; p++) {
        switch (*p) {
            case '"': fputs("\\\"", fp); break;
            case '\\': fputs("\\\\", fp); break;
            case '\n': fputs("\\n", fp); break;
            case '\r': fputs("\\r", fp); break;
            case '\t': fputs("\\t", fp); break;
            default: fputc(*p, fp); break;
        }
    }
    fputc('"', fp);
}

// Helper: Read entire file into memory
static char* read_entire_file(const char *path, size_t *out_size) {
    FILE *fp = fopen(path, "r");
    if (!fp) return NULL;
    
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    if (size < 0) {
        fclose(fp);
        return NULL;
    }
    fseek(fp, 0, SEEK_SET);
    
    char *buffer = malloc(size + 1);
    if (!buffer) {
        fclose(fp);
        return NULL;
    }
    
    size_t read_size = fread(buffer, 1, size, fp);
    buffer[read_size] = '\0';
    fclose(fp);
    
    if (out_size) *out_size = read_size;
    return buffer;
}

// Helper: Write a single measurement to file pointer
static void write_measurement_json(FILE *fp, pmc_multi_handle_t *handle, int indent_level) {
    // Get current timestamp
    time_t now = time(NULL);
    struct tm *tm_info = localtime(&now);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y-%m-%dT%H:%M:%S", tm_info);
    
    const char *indent = "    ";
    
    // Write measurement header
    fprintf(fp, "%s{\n", indent);
    fprintf(fp, "%s  \"label\": ", indent);
    write_json_string(fp, handle->label);
    fprintf(fp, ",\n");
    fprintf(fp, "%s  \"timestamp\": \"%s\",\n", indent, timestamp);
    fprintf(fp, "%s  \"num_events\": %zu,\n", indent, handle->num_events);
    fprintf(fp, "%s  \"events\": [\n", indent);
    
    // Write each event
    for (size_t i = 0; i < handle->num_events; i++) {
        pmc_ctx_t *ctx = handle->contexts[i];
        if (!ctx) continue;
        
        const char *event_name = handle->requests[i].event;
        const char *mode = mode_to_string(handle->requests[i].mode);
        
        fprintf(fp, "%s    {\n", indent);
        fprintf(fp, "%s      \"index\": %zu,\n", indent, i + 1);
        fprintf(fp, "%s      \"event_name\": \"%s\",\n", indent, event_name);
        fprintf(fp, "%s      \"mode\": \"%s\",\n", indent, mode);
        
        // Read count
        uint64_t count = 0;
        pmc_read_count(ctx, &count);
        fprintf(fp, "%s      \"count\": %llu", indent, (unsigned long long)count);
        
        // Handle sampling modes
        if (handle->requests[i].mode == PMC_MODE_SAMPLING || 
            handle->requests[i].mode == PMC_MODE_SAMPLING_FREQ) {
            fprintf(fp, ",\n");
            
            // Output appropriate label for period vs frequency
            if (handle->requests[i].mode == PMC_MODE_SAMPLING) {
                fprintf(fp, "%s      \"sample_period\": %llu,\n", indent,
                        (unsigned long long)handle->requests[i].sample_period);
            } else {
                fprintf(fp, "%s      \"sample_freq_hz\": %llu,\n", indent,
                        (unsigned long long)handle->requests[i].sample_period);
            }
            
            // Read samples
            pmc_sample_t *samples = NULL;
            size_t num_samples = 0;
            
            if (pmc_read_samples(ctx, &samples, &num_samples, 0) == 0) {
                fprintf(fp, "%s      \"num_samples\": %zu,\n", indent, num_samples);
                fprintf(fp, "%s      \"samples\": [\n", indent);
                
                for (size_t j = 0; j < num_samples; j++) {
                    fprintf(fp, "%s        {\n", indent);
                    fprintf(fp, "%s          \"ip\": \"0x%llx\",\n", indent,
                            (unsigned long long)samples[j].ip);
                    fprintf(fp, "%s          \"pid\": %u,\n", indent, samples[j].pid);
                    fprintf(fp, "%s          \"tid\": %u,\n", indent, samples[j].tid);
                    fprintf(fp, "%s          \"time\": %llu,\n", indent,
                            (unsigned long long)samples[j].time);
                    fprintf(fp, "%s          \"count\": %llu\n", indent,
                            (unsigned long long)samples[j].count);
                    fprintf(fp, "%s        }", indent);
                    if (j < num_samples - 1) fprintf(fp, ",");
                    fprintf(fp, "\n");
                }
                
                fprintf(fp, "%s      ]\n", indent);
                free(samples);
            } else {
                fprintf(fp, "%s      \"num_samples\": 0,\n", indent);
                fprintf(fp, "%s      \"samples\": []\n", indent);
            }
        } else {
            fprintf(fp, "\n");
        }
        
        fprintf(fp, "%s    }", indent);
        if (i < handle->num_events - 1) fprintf(fp, ",");
        fprintf(fp, "\n");
    }
    
    // Write measurement footer
    fprintf(fp, "%s  ]\n", indent);
    fprintf(fp, "%s}", indent);
}

int pmc_export_json(pmc_multi_handle_t *handle, const char *json_path) {
    if (!handle || !json_path) {
        pmc_set_error("Invalid parameters");
        return -1;
    }
    
    // Check if file exists and read existing measurements
    size_t existing_size = 0;
    char *existing_content = read_entire_file(json_path, &existing_size);
    int has_existing = (existing_content != NULL && existing_size > 0);
    
    // Open file for writing
    FILE *fp = fopen(json_path, "w");
    if (!fp) {
        pmc_set_error("Failed to open JSON file: %s", strerror(errno));
        free(existing_content);
        return -1;
    }
    
    // Write file header
    fprintf(fp, "{\n");
    fprintf(fp, "  \"measurements\": [\n");
    
    // If existing measurements, extract and write them first
    if (has_existing) {
        // Find the start of measurements array
        char *measurements_start = strstr(existing_content, "\"measurements\"");
        if (measurements_start) {
            // Find the opening bracket of the array
            char *array_start = strchr(measurements_start, '[');
            if (array_start) {
                array_start++; // Move past '['
                
                // Find the closing bracket (look from end)
                char *array_end = strrchr(array_start, ']');
                if (array_end && array_end > array_start) {
                    // Extract the measurements content (between [ and ])
                    size_t content_len = array_end - array_start;
                    
                    // Skip leading whitespace
                    while (content_len > 0 && (*array_start == ' ' || *array_start == '\n' || *array_start == '\t')) {
                        array_start++;
                        content_len--;
                    }
                    
                    // Skip trailing whitespace
                    while (content_len > 0 && (array_start[content_len-1] == ' ' || 
                           array_start[content_len-1] == '\n' || array_start[content_len-1] == '\t')) {
                        content_len--;
                    }
                    
                    // Write existing measurements if not empty
                    if (content_len > 0) {
                        fwrite(array_start, 1, content_len, fp);
                        fprintf(fp, ",\n");
                    }
                }
            }
        }
    }
    
    // Write new measurement
    write_measurement_json(fp, handle, 0);
    fprintf(fp, "\n");
    
    // Write file footer
    fprintf(fp, "  ]\n");
    fprintf(fp, "}\n");
    
    fclose(fp);
    free(existing_content);
    return 0;
}

