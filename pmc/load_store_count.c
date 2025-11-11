/*
This program demonstrates three memory instruction retired events:
  - MEM_INST_RETIRED.ALL_LOADS  : event=0xD0, umask=0x81
  - MEM_INST_RETIRED.ALL_STORES : event=0xD0, umask=0x82
  - MEM_INST_RETIRED.ANY        : event=0xD0, umask=0x83
  gcc -O2 -o load_store_count load_store_count.c
  ./load_store_count 10000000
*/

#define _GNU_SOURCE
#include <asm/unistd.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

static long
perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags)
{
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

// Workload with explicit loads and stores
__attribute__((noinline)) static uint64_t workload(unsigned iters)
{
    // Allocate arrays to ensure real memory operations
    uint64_t *array = malloc(1024 * sizeof(uint64_t));
    if (!array) return 0;
    
    volatile uint64_t acc = 0;
    
    for (unsigned i = 0; i < iters; i++) {
        unsigned idx = (i * 7) & 1023;  // pseudo-random index
        
        // LOAD operations
        uint64_t val = array[idx];      // load 1
        uint64_t val2 = array[(idx + 1) & 1023]; // load 2
        
        // STORE operations
        array[idx] = val + val2 + i;    // store 1
        array[(idx + 511) & 1023] = acc; // store 2
        
        // Accumulate to prevent optimization
        acc += val;
    }
    
    free(array);
    return acc;
}

int main(int argc, char **argv)
{
    unsigned iters = (argc > 1) ? (unsigned)strtoul(argv[1], NULL, 0) : 10 * 1000 * 1000u;

    // Setup three counters for the three events
    struct perf_event_attr attr_loads, attr_stores, attr_any;
    
    // MEM_INST_RETIRED.ALL_LOADS: event=0xD0, umask=0x81
    memset(&attr_loads, 0, sizeof(attr_loads));
    attr_loads.size = sizeof(attr_loads);
    attr_loads.type = PERF_TYPE_RAW;
    attr_loads.config = 0xD0 | (0x81ULL << 8);
    attr_loads.disabled = 1;
    attr_loads.exclude_kernel = 1;
    attr_loads.exclude_hv = 1;
    attr_loads.exclude_idle = 1;

    // MEM_INST_RETIRED.ALL_STORES: event=0xD0, umask=0x82
    memset(&attr_stores, 0, sizeof(attr_stores));
    attr_stores.size = sizeof(attr_stores);
    attr_stores.type = PERF_TYPE_RAW;
    attr_stores.config = 0xD0 | (0x82ULL << 8);
    attr_stores.disabled = 1;
    attr_stores.exclude_kernel = 1;
    attr_stores.exclude_hv = 1;
    attr_stores.exclude_idle = 1;

    // MEM_INST_RETIRED.ANY: event=0xD0, umask=0x83
    memset(&attr_any, 0, sizeof(attr_any));
    attr_any.size = sizeof(attr_any);
    attr_any.type = PERF_TYPE_RAW;
    attr_any.config = 0xD0 | (0x83ULL << 8);
    attr_any.disabled = 1;
    attr_any.exclude_kernel = 1;
    attr_any.exclude_hv = 1;
    attr_any.exclude_idle = 1;

    // Open all three counters
    int fd_loads = perf_event_open(&attr_loads, 0, -1, -1, 0);
    if (fd_loads == -1) {
        fprintf(stderr, "perf_event_open (loads) failed: %s (errno=%d)\n", strerror(errno), errno);
        return 1;
    }

    int fd_stores = perf_event_open(&attr_stores, 0, -1, -1, 0);
    if (fd_stores == -1) {
        fprintf(stderr, "perf_event_open (stores) failed: %s (errno=%d)\n", strerror(errno), errno);
        close(fd_loads);
        return 1;
    }

    int fd_any = perf_event_open(&attr_any, 0, -1, -1, 0);
    if (fd_any == -1) {
        fprintf(stderr, "perf_event_open (any) failed: %s (errno=%d)\n", strerror(errno), errno);
        close(fd_loads);
        close(fd_stores);
        return 1;
    }

    // Reset all counters
    ioctl(fd_loads, PERF_EVENT_IOC_RESET, 0);
    ioctl(fd_stores, PERF_EVENT_IOC_RESET, 0);
    ioctl(fd_any, PERF_EVENT_IOC_RESET, 0);

    // Enable all counters
    ioctl(fd_loads, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(fd_stores, PERF_EVENT_IOC_ENABLE, 0);
    ioctl(fd_any, PERF_EVENT_IOC_ENABLE, 0);

    // Run the workload
    volatile uint64_t sink = workload(iters);

    // Disable all counters
    ioctl(fd_loads, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(fd_stores, PERF_EVENT_IOC_DISABLE, 0);
    ioctl(fd_any, PERF_EVENT_IOC_DISABLE, 0);

    // Read results
    uint64_t count_loads = 0, count_stores = 0, count_any = 0;
    read(fd_loads, &count_loads, sizeof(count_loads));
    read(fd_stores, &count_stores, sizeof(count_stores));
    read(fd_any, &count_any, sizeof(count_any));

    close(fd_loads);
    close(fd_stores);
    close(fd_any);

    printf("Workload iters: %u, sink=%llu\n", iters, (unsigned long long)sink);
    printf("MEM_INST_RETIRED.ALL_LOADS  : %llu\n", (unsigned long long)count_loads);
    printf("MEM_INST_RETIRED.ALL_STORES : %llu\n", (unsigned long long)count_stores);
    printf("MEM_INST_RETIRED.ANY        : %llu\n", (unsigned long long)count_any);
    printf("\nExpected: loads + stores ≈ any\n");
    printf("Actual: %llu + %llu = %llu (measured any: %llu)\n", 
           (unsigned long long)count_loads, 
           (unsigned long long)count_stores,
           (unsigned long long)(count_loads + count_stores),
           (unsigned long long)count_any);

    return 0;
}