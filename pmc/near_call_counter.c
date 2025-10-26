/*
 * Performance counter for BR_INST_RETIRED.NEAR_CALL
 * This event counts both direct and indirect near call instructions retired.
 * EventSel=C4H UMask=02H
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
 
 // Helper functions that will be called (generating NEAR_CALL instructions)
 __attribute__((noinline)) static uint64_t helper_func1(uint64_t x)
 {
     return x * 2 + 1;
 }
 
 __attribute__((noinline)) static uint64_t helper_func2(uint64_t x)
 {
     return x * 3 + 2;
 }
 
 __attribute__((noinline)) static uint64_t helper_func3(uint64_t x)
 {
     return x * 5 + 3;
 }
 
 // Function pointer type for indirect calls

// Without typedef, you'd have to write this every time:
// uint64_t (*my_func)(uint64_t);
// uint64_t (*operations[3])(uint64_t);
 typedef uint64_t (*func_ptr_t)(uint64_t);
 
 __attribute__((noinline)) static uint64_t workload(unsigned iters)
 {
     volatile uint64_t acc = 0;
     uint32_t x = 0x12345678u;
     
     // Array of function pointers for indirect calls
     func_ptr_t funcs[] = { helper_func1, helper_func2, helper_func3 };
     
     for (unsigned i = 0; i < iters; i++) {
         // Direct calls
         acc += helper_func1(i);        // direct near call
         acc += helper_func2(i);        // direct near call
         
         // Indirect call through function pointer
         x = x * 1103515245u + 12345u;  // LCG for pseudo-random selection
         func_ptr_t selected_func = funcs[x % 3];
         acc += selected_func(i);       // indirect near call
         
         // Another direct call
        //  if ((x & 0x100) == 0) {
        //      acc += helper_func3(i);     // direct near call (conditional)
        //  }
     }
     return acc;
 }
 
 int main(int argc, char **argv)
 {
     unsigned iters = (argc > 1) ? (unsigned)strtoul(argv[1], NULL, 0) : 1000000u;
 
     struct perf_event_attr attr;
     memset(&attr, 0, sizeof(attr));
     attr.size = sizeof(attr);
     attr.type = PERF_TYPE_RAW;
     // BR_INST_RETIRED.NEAR_CALL -> event=0xC4, umask=0x02
     attr.config = 0xC4 | (0x02ULL << 8);
     attr.disabled = 1;
     attr.exclude_kernel = 1;   // count user space only
     attr.exclude_hv = 1;
     attr.exclude_idle = 1;
 
     int fd = perf_event_open(&attr, /*pid=*/0, /*cpu=*/-1, /*group_fd=*/-1, /*flags=*/0);
     if (fd == -1) {
         fprintf(stderr, "perf_event_open failed: %s (errno=%d)\n", strerror(errno), errno);
         fprintf(stderr, "Note: This requires an Intel CPU with support for this event.\n");
         fprintf(stderr, "You may need to run with: sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'\n");
         return 1;
     }
 
     if (ioctl(fd, PERF_EVENT_IOC_RESET, 0) == -1) { 
         perror("RESET"); 
         close(fd);
         return 1; 
     }
     if (ioctl(fd, PERF_EVENT_IOC_ENABLE, 0) == -1) { 
         perror("ENABLE"); 
         close(fd);
         return 1; 
     }
 
     // Run the workload with call instructions
     volatile uint64_t sink = workload(iters);
 
     if (ioctl(fd, PERF_EVENT_IOC_DISABLE, 0) == -1) { 
         perror("DISABLE"); 
         close(fd);
         return 1; 
     }
 
     uint64_t count = 0;
     if (read(fd, &count, sizeof(count)) != sizeof(count)) {
         perror("read");
         close(fd);
         return 1;
     }
     close(fd);
 
     printf("Workload iterations: %u\n", iters);
     printf("Workload result (sink): %llu\n", (unsigned long long)sink);
     printf("BR_INST_RETIRED.NEAR_CALL count: %llu\n", (unsigned long long)count);
     
     // Estimate: we expect approximately 3-4 calls per iteration
     // (2 direct always + 1 indirect always + sometimes 1 conditional direct)
     double calls_per_iter = (double)count / iters;
     printf("Average near calls per iteration: %.2f\n", calls_per_iter);
     
     return 0;
 }