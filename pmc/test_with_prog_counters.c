/*
 * Test fixed counters after pinning 5 programmable counters
 * By theory, two fixed counters should be working, the 5th event should be failed as programmable counter limit is reached
 * Build: gcc -O2 -Wall -Wextra test_with_prog_counters.c -o test_with_prog_counters
 * Run: ./test_with_prog_counters
 */

#define _GNU_SOURCE
#include <asm/unistd.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <errno.h>

static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu,
                            int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

int main() {
    int fds[7];
    
    printf("========================================\n");
    printf("Testing Fixed Counters with 5 Programmable Counters Pinned\n");
    printf("========================================\n\n");
    
    // Create 5 programmable counter events (pinned)
    struct perf_event_attr prog_attr;
    uint64_t prog_configs[] = {0x02C4, 0x01C4, 0x00C5, 0x01D1, 0x08D1};
    const char *prog_names[] = {
        "BR_INST_RETIRED.NEAR_CALL",
        "BR_INST_RETIRED.CONDITIONAL",
        "BR_MISP_RETIRED.ALL_BRANCHES",
        "MEM_LOAD_RETIRED.L1_HIT",
        "MEM_LOAD_RETIRED.L1_MISS",
    };
    
    printf("Creating 5 programmable counter events:\n");
    for (int i = 0; i < 5; i++) {
        memset(&prog_attr, 0, sizeof(prog_attr));
        prog_attr.size = sizeof(prog_attr);
        prog_attr.type = PERF_TYPE_RAW;
        prog_attr.config = prog_configs[i];
        prog_attr.disabled = 1;
        prog_attr.pinned = 1;
        prog_attr.exclude_kernel = 1;
        prog_attr.exclude_hv = 1;
        
        fds[i] = perf_event_open(&prog_attr, 0, -1, -1, 0);
        printf("  %d. %s: %s (fd=%d)\n", i+1, prog_names[i],
               fds[i] != -1 ? "SUCCESS" : "FAILED", fds[i]);
    }
    
    printf("\nNow creating fixed counter events:\n");
    
    // Create CYCLES fixed counter
    struct perf_event_attr cycles_attr;
    memset(&cycles_attr, 0, sizeof(cycles_attr));
    cycles_attr.size = sizeof(cycles_attr);
    cycles_attr.type = PERF_TYPE_HARDWARE;
    cycles_attr.config = PERF_COUNT_HW_CPU_CYCLES;
    cycles_attr.disabled = 1;
    cycles_attr.pinned = 1;
    cycles_attr.exclude_kernel = 1;
    cycles_attr.exclude_hv = 1;
    
    fds[5] = perf_event_open(&cycles_attr, 0, -1, -1, 0);
    printf("  6. CPU_CYCLES (FIXED): %s (fd=%d)\n",
           fds[5] != -1 ? "CREATE SUCCESS" : "CREATE FAILED", fds[5]);
    if (fds[5] == -1) {
        printf("     Error: %s\n", strerror(errno));
    }
    
    // Create INSTRUCTIONS fixed counter
    struct perf_event_attr inst_attr;
    memset(&inst_attr, 0, sizeof(inst_attr));
    inst_attr.size = sizeof(inst_attr);
    inst_attr.type = PERF_TYPE_HARDWARE;
    inst_attr.config = PERF_COUNT_HW_INSTRUCTIONS;
    inst_attr.disabled = 1;
    inst_attr.pinned = 1;
    inst_attr.exclude_kernel = 1;
    inst_attr.exclude_hv = 1;
    
    fds[6] = perf_event_open(&inst_attr, 0, -1, -1, 0);
    printf("  7. INSTRUCTIONS (FIXED): %s (fd=%d)\n",
           fds[6] != -1 ? "CREATE SUCCESS" : "CREATE FAILED", fds[6]);
    if (fds[6] == -1) {
        printf("     Error: %s\n", strerror(errno));
    }
    
    printf("\n========================================\n");
    printf("Running workload and reading counters\n");
    printf("========================================\n\n");
    
    // Enable all counters
    for (int i = 0; i < 7; i++) {
        if (fds[i] != -1) {
            ioctl(fds[i], PERF_EVENT_IOC_RESET, 0);
            ioctl(fds[i], PERF_EVENT_IOC_ENABLE, 0);
        }
    }
    
    // Run workload
    volatile uint64_t x = 0;
    for (int i = 0; i < 500000; i++) {
        x += i;
        if (i % 2) x *= 2;
        if (i % 3) x /= 2;
    }
    
    // Disable all counters
    for (int i = 0; i < 7; i++) {
        if (fds[i] != -1) {
            ioctl(fds[i], PERF_EVENT_IOC_DISABLE, 0);
        }
    }
    
    // Read programmable counters
    printf("Reading programmable counters:\n");
    int prog_success = 0, prog_fail = 0;
    for (int i = 0; i < 5; i++) {
        if (fds[i] != -1) {
            uint64_t count = 0;
            ssize_t bytes = read(fds[i], &count, sizeof(count));
            printf("  %d. %s: ", i+1, prog_names[i]);
            if (bytes == sizeof(count)) {
                printf("SUCCESS (count=%llu)\n", (unsigned long long)count);
                prog_success++;
            } else {
                printf("READ FAILED (returned %zd bytes)\n", bytes);
                prog_fail++;
            }
        }
    }
    
    printf("\nReading fixed counters:\n");
    int cycles_ok = 0, inst_ok = 0;
    
    if (fds[5] != -1) {
        uint64_t count = 0;
        ssize_t bytes = read(fds[5], &count, sizeof(count));
        printf("  6. CPU_CYCLES: ");
        if (bytes == sizeof(count) && count > 0) {
            printf("SUCCESS (count=%llu)\n", (unsigned long long)count);
            cycles_ok = 1;
        } else if (bytes == sizeof(count) && count == 0) {
            printf("READ SUCCESS but count=0 (counter not actually counting)\n");
        } else {
            printf("READ FAILED (returned %zd bytes)\n", bytes);
        }
    }
    
    if (fds[6] != -1) {
        uint64_t count = 0;
        ssize_t bytes = read(fds[6], &count, sizeof(count));
        printf("  7. INSTRUCTIONS: ");
        if (bytes == sizeof(count) && count > 0) {
            printf("SUCCESS (count=%llu)\n", (unsigned long long)count);
            inst_ok = 1;
        } else if (bytes == sizeof(count) && count == 0) {
            printf("READ SUCCESS but count=0 (counter not actually counting)\n");
        } else {
            printf("READ FAILED (returned %zd bytes)\n", bytes);
        }
    }
    
    // Cleanup
    for (int i = 0; i < 7; i++) {
        if (fds[i] != -1) close(fds[i]);
    }
    
    printf("\n========================================\n");
    printf("CONCLUSION\n");
    printf("========================================\n");
    printf("Programmable counters: %d working, %d failed\n", prog_success, prog_fail);
    printf("Fixed counters: CPU_CYCLES=%s, INSTRUCTIONS=%s\n",
           cycles_ok ? "WORKING" : "NOT COUNTING",
           inst_ok ? "WORKING" : "NOT COUNTING");
    printf("\n");
    if (prog_success == 4 && prog_fail >= 1) {
        printf("RESULT: 4 programmable counter limit demonstrated!\n");
        printf("- First 4 programmable events counted successfully\n");
        printf("- 5th programmable event failed to count\n");
    }
    if (cycles_ok == 0 && inst_ok == 1) {
        printf("\nCPU_CYCLES ISSUE: Likely being used by NMI watchdog\n");
        printf("- INSTRUCTIONS works (uses Fixed Counter 0)\n");
        printf("- CPU_CYCLES fails (Fixed Counter 1 reserved by kernel)\n");
        printf("- This demonstrates fixed counters are separate from programmable\n");
    }
    printf("========================================\n");
    
    return 0;
}

