// rdpmc_demo.c
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <x86intrin.h>   // for __rdtsc; we’ll use inline asm for rdpmc
#include <asm/msr.h>


static inline uint64_t rdpmc(uint32_t ecx) {
    uint32_t lo, hi;
    __asm__ volatile ("rdpmc" : "=a"(lo), "=d"(hi) : "c"(ecx));
    return ((uint64_t)hi << 32) | lo;
}

static inline uint64_t nsec_now(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

int main(void) {
    pid_t pid = getpid();
    printf("Process ID: %d\n", pid);
    // ECX encoding: bit30=1 ⇒ fixed-function; 0 ⇒ programmable (PMC)
    const uint32_t PMC0   = 0;             // programmable counter 0
    // Table 20-2. Association of Fixed-Function Performance Counters with Architectural Performance Events
    const uint32_t FIXED0 = 0x40000000u;   // fixed counter 1 = retired instruction

    const uint32_t FIXED1 = 0x40000001u;   // fixed counter 1 = unhalted core cycles


    uint64_t t0 = nsec_now();
    uint64_t branch_before = rdpmc(PMC0);
    
    uint64_t unhalted_core_cycles_set = rdpmc(FIXED1);

    // do some busy work
    volatile double s = 0;

    uint64_t ins_retired_set = rdpmc(FIXED0);

    // for (int i = 0; i < 50*1000*1000; i++) s += i * 0.000001;
    for (int i = 0; i < 100; i++) s += 1;

    uint64_t ins_retired_stop = rdpmc(FIXED0);

    uint64_t branch_taken = rdpmc(PMC0);
    uint64_t unhalted_core_cycles_stop = rdpmc(FIXED1);
    uint64_t t1 = nsec_now();

    printf("fx1_1 is %llu\n", unhalted_core_cycles_stop);
    printf("fx1_0 is %llu\n", unhalted_core_cycles_set);
    printf("FIXED0 delta: %llu (retired instruction)\n", (unsigned long long)(ins_retired_stop - ins_retired_set));
    printf("Elapsed:      %.3f ms\n", (t1 - t0)/1e6);
    printf("FIXED1 delta: %llu (unhalted core cycles)\n", (unsigned long long)(unhalted_core_cycles_stop - unhalted_core_cycles_set));
    printf("branch before %llu\n", branch_before);
    printf("branch taken %llu\n", branch_taken);
    printf("branch delta %llu\n", (unsigned long long)(branch_taken - branch_before));
    (void)s;
    return 0;
}

// Notes
// # Enable fixed1 (unhalted core cycles) in user+kernel:
// sudo wrmsr -p 0 0x38D 0x000000000000000A   # fixed1: USR=1, OS=1  (0b1010 in bits 4..7)
// Intel® 64 and IA-32 Architectures Software Developer’s Manual, Volume 3B: System Programming Guide, Part 2
// Figure 20-2. Layout of IA32_FIXED_CTR_CTRL MSR
// 0x38D is from #define MSR_CORE_PERF_FIXED_CTR_CTRL	0x0000038d of linux/arch/x86/include/asm/msr-index.h
// so another way to rdpmc(FIXED1) is: sudo rdmsr -p 0 0x30a

