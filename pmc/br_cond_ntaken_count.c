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

static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                            int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

#ifndef LIKELY
#define LIKELY(x)   __builtin_expect(!!(x), 0)  // predict false (not taken)
#endif

// Make the data volatile so compiler can't fold the branch away.
static volatile unsigned char *g_data;

__attribute__((noinline))
static uint64_t workload(size_t n, size_t true_every) {
    volatile uint64_t acc = 0;
    for (size_t i = 0; i < n; i++) {
        unsigned char v = g_data[i];
        // Branch is almost always NOT taken (v == 1 rarely true)
        if (LIKELY(v == 1)) {
            acc++;               // ensure side-effect
        }
    }
    return acc;
}

int main(int argc, char **argv) {
    size_t iters      = (argc > 1) ? strtoull(argv[1], NULL, 0) : 20000000ULL;
    size_t true_every = (argc > 2) ? strtoull(argv[2], NULL, 0) : 0; // 0 = never true

    // Allocate and initialize data
    unsigned char *buf = (unsigned char*) aligned_alloc(64, iters);
    if (!buf) { perror("alloc"); return 1; }
    memset(buf, 0, iters);
    if (true_every) {
        for (size_t i = true_every-1; i < iters; i += true_every) buf[i] = 1;
    }
    g_data = buf;

    // Set up raw event: BR_INST_RETIRED.COND_NTAKEN => event=0xC4, umask=0x10
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.size = sizeof(attr);
    attr.type = PERF_TYPE_RAW;
    attr.config = 0xC4 | (0x10ULL << 8);
    attr.disabled = 1;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;
    attr.exclude_idle = 1;

    int fd = perf_event_open(&attr, /*pid=*/0, /*cpu=*/-1, /*group_fd=*/-1, /*flags=*/0);
    if (fd == -1) {
        fprintf(stderr, "perf_event_open failed: %s (errno=%d)\n", strerror(errno), errno);
        return 1;
    }

    ioctl(fd, PERF_EVENT_IOC_RESET, 0);
    ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);

    volatile uint64_t sink = workload(iters, true_every);

    ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);

    uint64_t count = 0;
    if (read(fd, &count, sizeof(count)) != sizeof(count)) {
        perror("read");
        return 1;
    }
    close(fd);

    printf("iters=%zu, true_every=%zu, sink=%llu\n",
           iters, true_every, (unsigned long long)sink);
    printf("BR_INST_RETIRED.COND_NTAKEN count: %llu\n",
           (unsigned long long)count);

    free((void*)buf);
    return 0;
}
