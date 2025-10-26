/*
This program counts the number of conditional branches mispredicted.
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

// perf_event_open wrapper
static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                            int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

static int open_counter_raw(uint64_t config, int group_fd, int leader) {
    struct perf_event_attr a; memset(&a, 0, sizeof(a));
    a.size = sizeof(a);
    a.type = PERF_TYPE_RAW;
    a.config = config;                  // raw event encoding
    a.disabled = leader ? 1 : 0;        // leader starts disabled; followers enabled with group
    a.exclude_kernel = 1;
    a.exclude_hv = 1;
    a.exclude_idle = 1;
    return perf_event_open(&a, /*pid=*/0, /*cpu=*/-1, group_fd, 0);
}

static int open_counter_hw(uint64_t hw_id, int group_fd) {
    struct perf_event_attr a; memset(&a, 0, sizeof(a));
    a.size = sizeof(a);
    a.type = PERF_TYPE_HARDWARE;
    a.config = hw_id;
    a.disabled = 0;
    a.exclude_kernel = 1;
    a.exclude_hv = 1;
    a.exclude_idle = 1;
    return perf_event_open(&a, /*pid=*/0, /*cpu=*/-1, group_fd, 0);
}

static inline int64_t read_u64(int fd, uint64_t *val) {
    return read(fd, val, sizeof(*val));
}

// ---------------- RNG & workloads ----------------
static inline uint32_t xorshift32(uint32_t *s) {
    uint32_t x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return *s = x;
}

__attribute__((noinline))
static uint64_t workload_rand(uint64_t iters, uint32_t seed) {
    volatile uint64_t acc = 0;
    uint32_t state = seed ? seed : 0x12345678u;
    for (uint64_t i = 0; i < iters; i++) {
        uint32_t r = xorshift32(&state);
        if (r & 1u)      acc++;    else acc += 2;   // ~50/50
        if (r & 0x8000)  acc += 3; else acc += 4;   // ~50/50
    }
    return acc;
}

static volatile unsigned char *g_data;

__attribute__((noinline))
static uint64_t workload_easy(uint64_t iters) {
    volatile uint64_t acc = 0;
    for (uint64_t i = 0; i < iters; i++) {
        unsigned char v = g_data[i];   // all zeros
        if (v == 1) acc++;             // almost never taken → highly predictable not-taken
    }
    return acc;
}

// --------------- Run one phase ---------------
static void run_phase(const char *name, int fd_lead, int fd_miss, int fd_br,
                      uint64_t (*fn)(uint64_t, uint32_t),
                      uint64_t iters, uint32_t seed,
                      uint64_t *out_misp_cond, uint64_t *out_miss, uint64_t *out_br)
{
    // Reset + enable group
    if (ioctl(fd_lead, PERF_EVENT_IOC_RESET,  PERF_IOC_FLAG_GROUP) == -1) { perror("RESET"); exit(1); }
    if (ioctl(fd_lead, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) == -1) { perror("ENABLE"); exit(1); }

    volatile uint64_t sink = fn ? fn(iters, seed) : 0;
    (void)sink;

    if (ioctl(fd_lead, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) == -1) { perror("DISABLE"); exit(1); }

    uint64_t a=0,b=0,c=0;
    if (read_u64(fd_lead, &a) != sizeof(a)) { perror("read lead"); exit(1); }
    if (fd_miss >= 0 && read_u64(fd_miss, &b) != sizeof(b)) { perror("read miss"); }
    if (fd_br   >= 0 && read_u64(fd_br,   &c) != sizeof(c)) { perror("read br"); }

    *out_misp_cond = a; *out_miss = b; *out_br = c;

    printf("\n[%s] BR_MISP_RETIRED.CONDITIONAL: %llu\n", name, (unsigned long long)a);
    if (fd_miss >= 0 && fd_br >= 0) {
        double rate = c ? (double)b / (double)c * 100.0 : 0.0;
        printf("[%s] branch-misses (all): %llu, branches (all): %llu, miss rate ≈ %.2f%%\n",
               name, (unsigned long long)b, (unsigned long long)c, rate);
    }
}

int main(int argc, char **argv) {
    uint64_t iters = (argc > 1) ? strtoull(argv[1], NULL, 0) : 50ULL*1000*1000ULL;
    uint32_t seed  = (argc > 2) ? (uint32_t)strtoul(argv[2], NULL, 0) : 0xC0FFEEu;

    // Prepare easy-mode data: all zeros
    unsigned char *buf = (unsigned char*) aligned_alloc(64, iters ? iters : 64);
    if (!buf) { perror("alloc"); return 1; }
    memset(buf, 0, iters);
    g_data = buf;

    // Leader: BR_MISP_RETIRED.CONDITIONAL => event=0xC5, umask=0x01
    uint64_t raw_cfg = 0xC5 | (0x01ULL << 8);
    int fd_lead = open_counter_raw(raw_cfg, -1, /*leader=*/1);
    if (fd_lead < 0) { fprintf(stderr, "open leader failed: %s\n", strerror(errno)); return 1; }

    // Followers for context
    int fd_br_miss = open_counter_hw(PERF_COUNT_HW_BRANCH_MISSES, fd_lead);
    int fd_br_all  = open_counter_hw(PERF_COUNT_HW_BRANCH_INSTRUCTIONS, fd_lead);

    // ---- Phase 1: EASY (predictable) ----
    uint64_t misp_easy=0, miss_easy=0, br_easy=0;
    run_phase("easy", fd_lead, fd_br_miss, fd_br_all,
              /*fn=*/(uint64_t (*)(uint64_t,uint32_t))workload_easy, iters, 0,
              &misp_easy, &miss_easy, &br_easy);

    // ---- Phase 2: RAND (unpredictable) ----
    uint64_t misp_rand=0, miss_rand=0, br_rand=0;
    run_phase("rand", fd_lead, fd_br_miss, fd_br_all,
              workload_rand, iters, seed,
              &misp_rand, &miss_rand, &br_rand);

    close(fd_lead); if (fd_br_miss>=0) close(fd_br_miss); if (fd_br_all>=0) close(fd_br_all);
    free(buf);

    // Summary
    printf("\n=== Summary (iters=%llu) ===\n", (unsigned long long)iters);
    printf("easy: mispred_cond=%llu, branches=%llu\n",
           (unsigned long long)misp_easy, (unsigned long long)br_easy);
    printf("rand: mispred_cond=%llu, branches=%llu\n",
           (unsigned long long)misp_rand, (unsigned long long)br_rand);
    return 0;
}
