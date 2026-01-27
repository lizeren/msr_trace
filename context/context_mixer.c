#define _GNU_SOURCE
#include "context_mixer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdint.h>

#define LLC_SIZE (256 * 1024 * 1024)
#define L1D_SIZE (32 * 1024)
#define PAGE_SIZE 4096
#define TLB_PAGES 65536

static uint64_t xorshift64(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static inline uint64_t rdtsc(void) {
    uint32_t lo, hi;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

static void mixer_llc_thrash(size_t duration_us) {
    volatile char* buffer = malloc(LLC_SIZE);
    if (!buffer) {
        fprintf(stderr, "Failed to allocate LLC thrash buffer\n");
        return;
    }

    uint64_t start = rdtsc();
    uint64_t duration = duration_us * 2400;
    volatile char sink = 0;

    while (rdtsc() - start < duration) {
        for (size_t i = 0; i < LLC_SIZE; i += 64) {
            sink += buffer[i];
            buffer[i] = (char)(i & 0xFF);
        }
    }

    free((void*)buffer);
}

static void mixer_l1d_hot(size_t duration_us) {
    volatile char* buffer = malloc(L1D_SIZE);
    if (!buffer) {
        fprintf(stderr, "Failed to allocate L1D hot buffer\n");
        return;
    }

    for (size_t i = 0; i < L1D_SIZE; i++) {
        buffer[i] = (char)(i & 0xFF);
    }

    uint64_t start = rdtsc();
    uint64_t duration = duration_us * 2400;
    volatile char sink = 0;

    while (rdtsc() - start < duration) {
        for (size_t i = 0; i < L1D_SIZE; i++) {
            sink += buffer[i];
        }
    }

    free((void*)buffer);
}

__attribute__((noinline)) static int small_func_0(int x) { return x * 3 + 1; }
__attribute__((noinline)) static int small_func_1(int x) { return x * 5 + 7; }
__attribute__((noinline)) static int small_func_2(int x) { return x * 11 + 13; }
__attribute__((noinline)) static int small_func_3(int x) { return x * 17 + 19; }
__attribute__((noinline)) static int small_func_4(int x) { return x * 23 + 29; }
__attribute__((noinline)) static int small_func_5(int x) { return x * 31 + 37; }
__attribute__((noinline)) static int small_func_6(int x) { return x * 41 + 43; }
__attribute__((noinline)) static int small_func_7(int x) { return x * 47 + 53; }
__attribute__((noinline)) static int small_func_8(int x) { return x * 59 + 61; }
__attribute__((noinline)) static int small_func_9(int x) { return x * 67 + 71; }
__attribute__((noinline)) static int small_func_10(int x) { return x * 73 + 79; }
__attribute__((noinline)) static int small_func_11(int x) { return x * 83 + 89; }
__attribute__((noinline)) static int small_func_12(int x) { return x * 97 + 101; }
__attribute__((noinline)) static int small_func_13(int x) { return x * 103 + 107; }
__attribute__((noinline)) static int small_func_14(int x) { return x * 109 + 113; }
__attribute__((noinline)) static int small_func_15(int x) { return x * 127 + 131; }

typedef int (*small_func_t)(int);

static small_func_t small_funcs[16] = {
    small_func_0, small_func_1, small_func_2, small_func_3,
    small_func_4, small_func_5, small_func_6, small_func_7,
    small_func_8, small_func_9, small_func_10, small_func_11,
    small_func_12, small_func_13, small_func_14, small_func_15
};

static void mixer_icache_churn(size_t duration_us) {
    uint64_t state = (uint64_t)time(NULL);
    uint64_t start = rdtsc();
    uint64_t duration = duration_us * 2400;
    volatile int result = 0;

    while (rdtsc() - start < duration) {
        for (int i = 0; i < 1000; i++) {
            uint64_t idx = xorshift64(&state) & 0xF;
            result = small_funcs[idx](result);
        }
    }
}

static void mixer_branch_chaos(size_t duration_us) {
    uint64_t state = (uint64_t)time(NULL);
    uint64_t start = rdtsc();
    uint64_t duration = duration_us * 2400;
    volatile int result = 0;

    while (rdtsc() - start < duration) {
        for (int i = 0; i < 10000; i++) {
            uint64_t rnd = xorshift64(&state);
            
            if (rnd & 0x1) result++;
            else result--;
            
            if (rnd & 0x2) result *= 3;
            else result /= 2;
            
            if (rnd & 0x4) result ^= 0x5A5A;
            else result &= 0xFFFF;
            
            if (rnd & 0x8) result += (int)(rnd >> 32);
            else result -= (int)(rnd >> 16);
            
            if (rnd & 0x10) {
                result = (result << 3) | (result >> 29);
            } else if (rnd & 0x20) {
                result = (result << 7) | (result >> 25);
            } else if (rnd & 0x40) {
                result = (result << 11) | (result >> 21);
            } else {
                result = (result << 13) | (result >> 19);
            }
        }
    }
}

static void mixer_tlb_thrash(size_t duration_us) {
    size_t total_size = TLB_PAGES * PAGE_SIZE;
    
    void* buffer = mmap(NULL, total_size, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (buffer == MAP_FAILED) {
        fprintf(stderr, "Failed to mmap for TLB thrash\n");
        return;
    }

    uint64_t start = rdtsc();
    uint64_t duration = duration_us * 2400;
    volatile char sink = 0;

    while (rdtsc() - start < duration) {
        for (size_t i = 0; i < TLB_PAGES; i++) {
            char* page = (char*)buffer + (i * PAGE_SIZE);
            sink += page[0];
            page[0] = (char)(i & 0xFF);
        }
    }

    munmap(buffer, total_size);
}

static void mixer_alu_spin(size_t duration_us) {
    uint64_t start = rdtsc();
    uint64_t duration = duration_us * 2400;
    volatile uint64_t result = 1;

    while (rdtsc() - start < duration) {
        for (int i = 0; i < 10000; i++) {
            result = result * 6364136223846793005ULL + 1442695040888963407ULL;
            result += result * 3;
            result ^= result >> 13;
            result *= 0x9e3779b97f4a7c15ULL;
        }
    }
}

static void mixer_memory_mixed(size_t duration_us) {
    size_t buf_size = 128 * 1024 * 1024;
    volatile char* buffer = malloc(buf_size);
    if (!buffer) {
        fprintf(stderr, "Failed to allocate memory mixed buffer\n");
        return;
    }

    uint64_t state = (uint64_t)time(NULL);
    uint64_t start = rdtsc();
    uint64_t duration = duration_us * 2400;
    volatile char sink = 0;

    while (rdtsc() - start < duration) {
        for (size_t i = 0; i < buf_size; i += 64) {
            sink += buffer[i];
        }
        
        for (int i = 0; i < 1000; i++) {
            size_t idx = xorshift64(&state) % buf_size;
            sink += buffer[idx];
            buffer[idx] = (char)(idx & 0xFF);
        }
    }

    free((void*)buffer);
}

#define ALLOC_POOL_SIZE (64 * 1024 * 1024)
#define MAX_CHUNKS 1024

typedef struct {
    void* ptr;
    size_t size;
} chunk_info_t;

static void mixer_alloc_chaos(size_t duration_us) {
    void* pool = malloc(ALLOC_POOL_SIZE);
    if (!pool) {
        fprintf(stderr, "Failed to allocate pool for alloc chaos\n");
        return;
    }

    chunk_info_t chunks[MAX_CHUNKS] = {0};
    size_t pool_offset = 0;
    uint64_t state = (uint64_t)time(NULL);
    uint64_t start = rdtsc();
    uint64_t duration = duration_us * 2400;

    while (rdtsc() - start < duration) {
        for (int op = 0; op < 100; op++) {
            uint64_t rnd = xorshift64(&state);
            size_t chunk_idx = rnd % MAX_CHUNKS;
            
            if (chunks[chunk_idx].ptr == NULL && pool_offset < ALLOC_POOL_SIZE) {
                size_t alloc_size = 64 + (rnd % 8192);
                if (pool_offset + alloc_size <= ALLOC_POOL_SIZE) {
                    chunks[chunk_idx].ptr = (char*)pool + pool_offset;
                    chunks[chunk_idx].size = alloc_size;
                    pool_offset += alloc_size;
                    memset(chunks[chunk_idx].ptr, (int)(rnd & 0xFF), alloc_size);
                }
            } else if (chunks[chunk_idx].ptr != NULL) {
                volatile char* p = chunks[chunk_idx].ptr;
                for (size_t i = 0; i < chunks[chunk_idx].size; i += 64) {
                    p[i] = 0;
                }
                chunks[chunk_idx].ptr = NULL;
                chunks[chunk_idx].size = 0;
            }
        }
        
        if (pool_offset > ALLOC_POOL_SIZE * 3 / 4) {
            memset(chunks, 0, sizeof(chunks));
            pool_offset = 0;
        }
    }

    free(pool);
}

int context_mixer_run(size_t duration_us) {
    if (duration_us == 0) {
        duration_us = 10000; // 0.01 second
    }

    // Read mixer type from environment variable
    const char* mixer_env = getenv("MIXER_INDICES");
    if (!mixer_env) {
        fprintf(stderr, "Warning: MIXER_INDICES not set, skipping context mixer\n");
        return 0;
    }

    int mixer_type = atoi(mixer_env);
    
    // Validate mixer type (1-8)
    if (mixer_type < 1 || mixer_type > 8) {
        fprintf(stderr, "Warning: Invalid MIXER_INDICES value %d (valid: 1-8), skipping context mixer\n", mixer_type);
        return 0;
    }

    switch (mixer_type) {
        case 1:
            mixer_llc_thrash(duration_us);
            break;
        case 2:
            mixer_l1d_hot(duration_us);
            break;
        case 3:
            mixer_icache_churn(duration_us);
            break;
        case 4:
            mixer_branch_chaos(duration_us);
            break;
        case 5:
            mixer_tlb_thrash(duration_us);
            break;
        case 6:
            mixer_alu_spin(duration_us);
            break;
        case 7:
            mixer_memory_mixed(duration_us);
            break;
        case 8:
            mixer_alloc_chaos(duration_us);
            break;
        default:
            fprintf(stderr, "Unknown mixer type: %d\n", mixer_type);
            return -1;
    }

    return 0;
}
