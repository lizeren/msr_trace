/*

This event counts conditional branch instructions retired.

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

// Raw event encoding for Intel: config = event | (umask << 8)
static long
perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags)
{
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

__attribute__((noinline)) static uint64_t workload(unsigned iters)
{
    volatile uint64_t acc = 0; // volatile to avoid if-conversion (cmov)
    uint32_t x = 0x12345678u;
    for (unsigned i = 0; i < iters; i++) {
        // Make a hard-to-predict condition to force real conditional branches
        x = x * 1103515245u + 12345u;        // LCG
        if (x & 1u) acc++;                   // conditional branch
        if ((x & 0x80000000u) == 0) acc++;   // another conditional branch
    }
    return acc;
}

int main(int argc, char **argv)
{
    unsigned iters = (argc > 1) ? (unsigned)strtoul(argv[1], NULL, 0) : 50 * 1000 * 1000u;

    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.size = sizeof(attr);
    attr.type = PERF_TYPE_RAW;
    // BR_INST_RETIRED.CONDITIONAL  -> event=0xC4, umask=0x01
    attr.config = 0xC4 | (0x01ULL << 8);
    attr.disabled = 1;
    attr.exclude_kernel = 1;   // count user space only
    attr.exclude_hv = 1;
    attr.exclude_idle = 1;

    // Optional: make it precise sampling-capable (not needed for counting)
    // attr.precise_ip = 2; // PEBS level (if you switch to sampling)

    int fd = perf_event_open(&attr, /*pid=*/0, /*cpu=*/-1, /*group_fd=*/-1, /*flags=*/0);
    if (fd == -1) {
        fprintf(stderr, "perf_event_open failed: %s (errno=%d)\n", strerror(errno), errno);
        return 1;
    }

    if (ioctl(fd, PERF_EVENT_IOC_RESET, 0) == -1) { perror("RESET"); return 1; }
    if (ioctl(fd, PERF_EVENT_IOC_ENABLE, 0) == -1) { perror("ENABLE"); return 1; }

    // Run the branchy workload
    volatile uint64_t sink = workload(iters);

    if (ioctl(fd, PERF_EVENT_IOC_DISABLE, 0) == -1) { perror("DISABLE"); return 1; }

    uint64_t count = 0;
    if (read(fd, &count, sizeof(count)) != sizeof(count)) {
        perror("read");
        return 1;
    }
    close(fd);

    printf("Workload iters: %u, sink=%llu\n", iters, (unsigned long long)sink);
    printf("BR_INST_RETIRED.CONDITIONAL count: %llu\n", (unsigned long long)count);
    return 0;
}
    