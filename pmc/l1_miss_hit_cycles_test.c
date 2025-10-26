/*
This program tests the number of L1 misses and hits.
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
#include <time.h>

// perf_event_open wrapper
static long perf_event_open(struct perf_event_attr *attr, pid_t pid, int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}
typedef uint64_t (*phase_fn_t)(uint8_t *buf, size_t sz, uint64_t iters);

static int open_raw(uint64_t cfg, int group_fd, int leader) {
    struct perf_event_attr a; memset(&a, 0, sizeof(a));
    a.size = sizeof(a);
    a.type = PERF_TYPE_RAW;
    a.config = cfg;
    a.disabled = leader ? 1 : 0;
    a.exclude_kernel = 1;
    a.exclude_hv = 1;
    a.exclude_idle = 1;
    return perf_event_open(&a, 0, -1, group_fd, 0);
}

static inline ssize_t read_u64(int fd, uint64_t *v) {
    return read(fd, v, sizeof(*v));
}

// ----------------- Workloads -----------------
static void touch_barrier(void) { asm volatile("" ::: "memory"); }

// HIT phase: repeatedly sweep a small buffer that fits in L1D.
__attribute__((noinline))
static uint64_t l1_hit_phase(uint8_t *buf, size_t sz, uint64_t iters) {
    volatile uint64_t acc = 0;
    for (uint64_t k = 0; k < iters; k++) {
        for (size_t i = 0; i < sz; i += 64) {  // 64B per cache line
            acc += buf[i];
        }
        touch_barrier();
    }
    return acc;
}

// Build a cache-line-granularity random permutation for pointer chasing.
static void build_ptr_chase(uint8_t *buf, size_t bytes) {
    const size_t step = 64;
    size_t n = bytes / step;
    size_t *idx = (size_t*)malloc(n * sizeof(size_t));
    for (size_t i = 0; i < n; i++) idx[i] = i;
    // Fisher–Yates shuffle
    for (size_t i = n - 1; i > 0; i--) {
        size_t j = (size_t) (rand() % (i + 1));
        size_t t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }
    // Each node stores next pointer (as offset) at start of its cache line
    for (size_t i = 0; i < n; i++) {
        size_t next = idx[(i + 1) % n];
        *(size_t*)(buf + idx[i] * step) = (size_t)(buf + next * step);
    }
    free(idx);
}

// MISS phase: dependent pointer chasing over a large array (thwarts prefetchers).
__attribute__((noinline))
static uint64_t l1_miss_phase(uint8_t *buf, size_t sz_bytes, uint64_t steps) {
    volatile uint64_t acc = 0;
    uint8_t *p = buf;
    for (uint64_t i = 0; i < steps; i++) {
        p = *(uint8_t * volatile *)p;  // dependent load → likely L1 miss
        acc += (uintptr_t)p & 1;
        touch_barrier();
    }
    return acc;
}

// --------------- One run helper ---------------
static void run_phase(const char *name,
    int fd_lead, int fd_cycles_l1dm, int fd_l1miss, int fd_l1hit,
    phase_fn_t fn,
    uint8_t *buf, size_t sz, uint64_t iters)
{
if (ioctl(fd_lead, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) == -1) { perror("RESET"); exit(1); }
if (ioctl(fd_lead, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) == -1) { perror("ENABLE"); exit(1); }

volatile uint64_t sink = fn(buf, sz, iters);
(void)sink;

if (ioctl(fd_lead, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) == -1) { perror("DISABLE"); exit(1); }

uint64_t cyc=0, m1=0, h1=0;
if (read(fd_lead,  &cyc, sizeof(cyc)) != sizeof(cyc)) { perror("read cycles_l1d_miss"); exit(1); }
if (read(fd_l1miss,&m1,  sizeof(m1))  != sizeof(m1))  { perror("read l1miss");        exit(1); }
if (read(fd_l1hit, &h1,  sizeof(h1))  != sizeof(h1))  { perror("read l1hit");         exit(1); }

printf("\n[%s]\n", name);
printf("CYCLE_ACTIVITY.CYCLES_L1D_MISS : %llu\n", (unsigned long long)cyc);
printf("MEM_LOAD_RETIRED.L1_MISS       : %llu\n", (unsigned long long)m1);
printf("MEM_LOAD_RETIRED.L1_HIT        : %llu\n", (unsigned long long)h1);
}

int main(int argc, char **argv) {
    // Tunables
    const size_t  L1_SIZE   = 32 * 1024;       // fits typical 32KB L1D
    const uint64_t HIT_SWEEPS = (argc > 1) ? strtoull(argv[1], NULL, 0) : 2000ULL;
    // const uint64_t MISS_STEPS = (argc > 2) ? strtoull(argv[2], NULL, 0) : 5ULL * 1000ULL * 1000ULL;
    const uint64_t MISS_STEPS = (argc > 2) ? strtoull(argv[2], NULL, 0) : 2000ULL;

    const size_t  BIG_SIZE = (argc > 3) ? strtoull(argv[3], NULL, 0) : (64ULL * 1024ULL * 1024ULL); // 64MB

    srand(0xBADC0DE);

    // Buffers
    uint8_t *small = aligned_alloc(64, L1_SIZE);
    uint8_t *big   = aligned_alloc(64, BIG_SIZE);
    if (!small || !big) { perror("alloc"); return 1; }
    memset(small, 1, L1_SIZE);
    memset(big,   0, BIG_SIZE);

    // Prepare pointer-chase ring for the big buffer
    build_ptr_chase(big, BIG_SIZE);

    // --- Open counters (group) ---
    // 1) Leader: CYCLE_ACTIVITY.CYCLES_L1D_MISS  event=0xA3, umask=0x08, cmask=0x08
    uint64_t cfg_cycles_l1dm = 0xA3 | (0x08ULL << 8) | (0x08ULL << 24);
    int fd_lead = open_raw(cfg_cycles_l1dm, -1, /*leader=*/1);
    if (fd_lead < 0) { fprintf(stderr, "open leader failed: %s\n", strerror(errno)); return 1; }

    // 2) MEM_LOAD_RETIRED.L1_MISS  event=0xD1, umask=0x08
    uint64_t cfg_l1_miss = 0xD1 | (0x08ULL << 8);
    int fd_l1miss = open_raw(cfg_l1_miss, fd_lead, 0);
    if (fd_l1miss < 0) { fprintf(stderr, "open l1_miss failed: %s\n", strerror(errno)); return 1; }

    // 3) MEM_LOAD_RETIRED.L1_HIT   event=0xD1, umask=0x01
    uint64_t cfg_l1_hit  = 0xD1 | (0x01ULL << 8);
    int fd_l1hit = open_raw(cfg_l1_hit, fd_lead, 0);
    if (fd_l1hit < 0) { fprintf(stderr, "open l1_hit failed: %s\n", strerror(errno)); return 1; }

    // --- Phase 1: HITs on small buffer ---
    run_phase("HIT phase (small L1-friendly sweeps)",
              fd_lead, fd_lead, fd_l1miss, fd_l1hit,
              l1_hit_phase, small, L1_SIZE, HIT_SWEEPS);

    // --- Phase 2: MISSes via pointer chasing on big buffer ---
    run_phase("MISS phase (pointer chase over big buffer)",
              fd_lead, fd_lead, fd_l1miss, fd_l1hit,
              l1_miss_phase, big, BIG_SIZE, MISS_STEPS);

    close(fd_l1hit); close(fd_l1miss); close(fd_lead);
    free(small); free(big);
    return 0;
}