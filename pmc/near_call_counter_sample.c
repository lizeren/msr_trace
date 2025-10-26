/*
 * Count + Sample BR_INST_RETIRED.NEAR_CALL using perf_event_open
 *
 * - Counting mode: prints total event count
 * - Sampling mode: samples every N events and prints IP + symbol (user space)
 *
 * Notes:
 *   - Event: BR_INST_RETIRED.NEAR_CALL (event=0xC4, umask=0x02)
 *   - Requires Linux perf_event and sufficient permissions.
 *   - Suggest setting precise_ip=2 for better attribution on modern Intel.
 *
 * Build: gcc -O2 -Wall -Wextra near_call_perf.c -ldl
 */

 #define _GNU_SOURCE
 #include <asm/unistd.h>
 #include <linux/perf_event.h>
 #include <sys/mman.h>
 #include <sys/ioctl.h>
 #include <sys/syscall.h>
 #include <unistd.h>
 #include <string.h>
 #include <stdint.h>
 #include <stdio.h>
 #include <stdlib.h>
 #include <errno.h>
 #include <dlfcn.h>
 #include <inttypes.h>

 #ifndef PAGE_SIZE
 #  define PAGE_SIZE 4096
 #endif
 
 // ---------- perf helpers ----------
 static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu,
                             int group_fd, unsigned long flags) {
     return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
 }
 
 // Minimal user-space rmb/wmb around ring-buffer head/tail updates.
 static inline void rb_barrier(void) { __sync_synchronize(); }
 
 // ---------- workload (same as your code) ----------
 __attribute__((noinline)) static uint64_t helper_func1(uint64_t x) { return x * 2 + 1; }
 __attribute__((noinline)) static uint64_t helper_func2(uint64_t x) { return x * 3 + 2; }
 __attribute__((noinline)) static uint64_t helper_func3(uint64_t x) { return x * 5 + 3; }
 
 typedef uint64_t (*func_ptr_t)(uint64_t);
 
 __attribute__((noinline)) static uint64_t workload(unsigned iters) {
     volatile uint64_t acc = 0;
     uint32_t x = 0x12345678u;
     func_ptr_t funcs[] = { helper_func1, helper_func2, helper_func3 };
 
     for (unsigned i = 0; i < iters; i++) {
         acc += helper_func1(i);        // direct near call
         acc += helper_func2(i);        // direct near call
 
         x = x * 1103515245u + 12345u;  // LCG
         func_ptr_t selected_func = funcs[x % 3];
         acc += selected_func(i);       // indirect near call
     }
     return acc;
 }
 
 // ---------- sampling ring buffer reading ----------
 struct ring {
     void *base;                  // mmap base (metadata page + data pages)
     size_t mmap_len;             // total length
     size_t data_size;            // size of data area (power of two)
     struct perf_event_mmap_page *meta;
 };
 
 static int ring_mmap(int fd, struct ring *r, unsigned data_pages_pow2) {
     if (data_pages_pow2 < 1) data_pages_pow2 = 1;
     size_t data_pages = 1ULL << data_pages_pow2; // must be power-of-two
     size_t data_sz    = data_pages * PAGE_SIZE; // 2^7 * 4KB = 512KB
     size_t mmap_len   = PAGE_SIZE + data_sz; // metadata + data
 
     void *base = mmap(NULL, mmap_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
     if (base == MAP_FAILED) return -1;
 
     r->base      = base;
     r->mmap_len  = mmap_len;
     r->data_size = data_sz;
     r->meta      = (struct perf_event_mmap_page *)base;
     return 0;
 }
 
 static void ring_unmap(struct ring *r) {
     if (r->base && r->base != MAP_FAILED) munmap(r->base, r->mmap_len);
     memset(r, 0, sizeof(*r));
 }
 
 // Read a contiguous span from ring buffer, handling wrap. Copies into 'dst'.
 static void ring_read_bytes(struct ring *r, uint64_t *tail, void *dst, size_t len) {
     char *data = (char *)r->base + PAGE_SIZE; // Skip metadata page

     // since *tail is mononically increasing,
     size_t mask = r->data_size - 1;
     // This extracts the actual position within the data buffer region from the monotonically increasing tail value.
     size_t pos  = (*tail) & mask;
 
     if (pos + len <= r->data_size) {
        // read len amount of bytes starting from pos
         memcpy(dst, data + pos, len);
     } else {
        // Wraparound case: copy in two chunks
         size_t first = r->data_size - pos; // read from pos to end of buffer  
         memcpy(dst, data + pos, first); // Copy first chunk
         memcpy((char*)dst + first, data, len - first);
     }
     *tail += len; //update the tail to the new position, since we have read len amount of bytes
 }
 
 // Pretty-print an address with best-effort symbolization.
 static void print_ip_symbol(uint64_t ip) {
     Dl_info info;
     if (dladdr((void*)(uintptr_t)ip, &info) && info.dli_sname) {
         long off = (long)((uintptr_t)ip - (uintptr_t)info.dli_saddr);
         printf("  ip=0x%llx  %s+0x%lx  (%s)\n",
                (unsigned long long)ip, info.dli_sname, off,
                info.dli_fname ? info.dli_fname : "?");
     } else {
         printf("  ip=0x%llx  (no symbol)\n", (unsigned long long)ip);
     }
 }
 
 // Drain the ring and print PERF_RECORD_SAMPLEs.
 static void drain_samples(struct ring *r, int max_to_print, int *out_seen) {
     *out_seen = 0;

     /*
     Buffer: [SSSSSSSSSSSSSSSS................]
              ^              ^
              data_tail     data_head
     */

     // head is updated by the kernel, which is producer
     uint64_t data_head = r->meta->data_head;
     rb_barrier();
     // tail is updated by the consumer
     uint64_t data_tail = r->meta->data_tail;
 
     while (data_tail < data_head) {
         struct perf_event_header h;
         ring_read_bytes(r, &data_tail, &h, sizeof(h));
 
         size_t rem = h.size - sizeof(h);
         if (h.type == PERF_RECORD_SAMPLE) {
             // Our sample_type below = IP | TID | TIME (8 + 8 + 8 bytes total payload here: ip,u32 pid,u32 tid,u64 time)
             // refer to include/linux/perf_event.h and search for PERF_RECORD_SAMPLE
             uint64_t ip = 0;
             uint32_t pid = 0, tid = 0;
             uint64_t time = 0;
 
             if (rem < sizeof(uint64_t) + sizeof(uint32_t)*2) {
                 // if the sample layout changes, skip gracefully
                 char skip[256];
                 while (rem > 0) {
                     size_t chunk = rem > sizeof(skip) ? sizeof(skip) : rem;
                     ring_read_bytes(r, &data_tail, skip, chunk);
                     rem -= chunk;
                 }
             } else {
                 ring_read_bytes(r, &data_tail, &ip, sizeof(ip));
                 ring_read_bytes(r, &data_tail, &pid, sizeof(pid));
                 ring_read_bytes(r, &data_tail, &tid, sizeof(tid));
                 ring_read_bytes(r, &data_tail, &time, sizeof(time));
                 if (*out_seen < max_to_print) {
                     printf("SAMPLE #%d  pid=%u tid=%u time= %" PRIu64 "\n", *out_seen + 1, pid, tid, time);
                    //  print_ip_symbol(ip);
                 }
                 (*out_seen)++;
             }
         } else {
             // Skip other records (COMM, MMAP, LOST, etc.)
             char skip[256];
             while (rem > 0) {
                 size_t chunk = rem > sizeof(skip) ? sizeof(skip) : rem;
                 ring_read_bytes(r, &data_tail, skip, chunk);
                 rem -= chunk;
             }
         }
     }
 
     rb_barrier();
     r->meta->data_tail = data_head;  // consume all
     rb_barrier();
 }
 
 // ---------- main ----------
 int main(int argc, char **argv) {
     unsigned iters = (argc > 1) ? (unsigned)strtoul(argv[1], NULL, 0) : 1000000u;
 
     int do_sample = 0;
     uint64_t sample_period = 1000; // default
     if (argc > 2 && strcmp(argv[2], "sample") == 0) {
         do_sample = 1;
         if (argc > 3) sample_period = strtoull(argv[3], NULL, 0);
         if (sample_period == 0) sample_period = 1;
     }
 
     struct perf_event_attr attr;
     memset(&attr, 0, sizeof(attr));
     attr.size = sizeof(attr);
     attr.type = PERF_TYPE_RAW;
     attr.config = 0xC4 | (0x02ULL << 8);      // BR_INST_RETIRED.NEAR_CALL
     attr.disabled = 1;
     attr.exclude_kernel = 1;
     attr.exclude_hv = 1;
     attr.exclude_idle = 1;
 
     // Optional but recommended on Intel for better attribution of IP:
     // 0 = best effort, 1 = requested, 2 = requested + skid constrained, 3 = must be precise
     attr.precise_ip = 2;
 
     // Counting vs Sampling
     struct ring r = {0};
     if (do_sample) {
         attr.sample_period = sample_period;
         attr.freq = 0; // treat sample_period as fixed period
         attr.sample_type = PERF_SAMPLE_IP | PERF_SAMPLE_TID | PERF_SAMPLE_TIME;
         attr.wakeup_events = 1;     // wakeup after at least one sample
         // We’ll only drain once after disabling, but wakeup_events helps avoid partial writes.
     }
 
     int fd = perf_event_open(&attr, /*pid=*/0, /*cpu=*/-1, /*group_fd=*/-1, /*flags=*/0);
     if (fd == -1) {
         fprintf(stderr, "perf_event_open failed: %s (errno=%d)\n", strerror(errno), errno);
         fprintf(stderr, "Tip: on some systems you may need lower perf_event_paranoid or CAP_SYS_ADMIN.\n");
         return 1;
     }
 
     // If sampling, set up the ring buffer (1 metadata page + 2^n data pages)
     if (do_sample) {
         if (ring_mmap(fd, &r, /*data_pages_pow2=*/7) == -1) { // 128 pages ( 128 pages × 4KB = 512KB data)
             perror("mmap");
             close(fd);
             return 1;
         }
     }
 
     if (ioctl(fd, PERF_EVENT_IOC_RESET, 0) == -1) { perror("RESET"); goto fail; }
     if (ioctl(fd, PERF_EVENT_IOC_ENABLE, 0) == -1) { perror("ENABLE"); goto fail; }
 
     volatile uint64_t sink = workload(iters);
 
     if (ioctl(fd, PERF_EVENT_IOC_DISABLE, 0) == -1) { perror("DISABLE"); goto fail; }
 
     printf("Workload iterations: %u\n", iters);
     printf("Workload result (sink): %llu\n", (unsigned long long)sink);
 
     if (!do_sample) {
         // ---- Counting path (same as before) ----
         uint64_t count = 0;
         if (read(fd, &count, sizeof(count)) != sizeof(count)) {
             perror("read");
             goto fail;
         }
         printf("BR_INST_RETIRED.NEAR_CALL count: %llu\n", (unsigned long long)count);
         double calls_per_iter = (double)count / iters;
         printf("Average near calls per iteration: %.2f\n", calls_per_iter);
     } else {
         // ---- Sampling path ----
         int seen = 0;
         drain_samples(&r, /*max_to_print=*/50, &seen);
         printf("Sample period: %llu near calls per sample\n", (unsigned long long)sample_period);
         printf("Total samples collected: %d\n", seen);
 
         // Also read total count (handy summary even in sampling mode)
         uint64_t count = 0;
         (void)!read(fd, &count, sizeof(count));
         if (count)
             printf("Counter (approx) after sampling: %llu\n", (unsigned long long)count);
         else
             printf("Counter read unavailable (kernel may not support read after sampling config).\n");
     }
 
     if (do_sample) ring_unmap(&r);
     close(fd);
     return 0;
 
 fail:
     if (do_sample) ring_unmap(&r);
     close(fd);
     return 1;
 }
 