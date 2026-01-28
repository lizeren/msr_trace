#define _GNU_SOURCE
#include "context_mixer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdint.h>

#define LLC_SIZE (64 * 1024)
#define L1D_SIZE (64 * 64)
#define PAGE_SIZE 4096
#define TLB_PAGES 64
#define MEMORY_MIXED_SIZE (256 * 1024)
#define FIXED_SEED 0x123456789ABCDEFULL

static uint64_t xorshift64(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static void mixer_llc_thrash(void) {
    volatile char* buffer = malloc(LLC_SIZE);
    if (!buffer) {
        fprintf(stderr, "Failed to allocate LLC thrash buffer\n");
        return;
    }

    volatile char sink = 0;
    
    // Touch 64KB with 64-byte stride, 1 pass (1024 cache lines)
    for (size_t i = 0; i < LLC_SIZE; i += 64) {
        sink += buffer[i];
        buffer[i] = (char)(i & 0xFF);
    }

    free((void*)buffer);
}

static void mixer_l1d_hot(void) {
    volatile char* buffer = malloc(L1D_SIZE);
    if (!buffer) {
        fprintf(stderr, "Failed to allocate L1D hot buffer\n");
        return;
    }

    // Initialize 64 cache lines (4KB)
    for (size_t i = 0; i < L1D_SIZE; i++) {
        buffer[i] = (char)(i & 0xFF);
    }

    volatile char sink = 0;
    
    // Touch 64 cache lines, 5 passes to keep hot in L1
    for (int pass = 0; pass < 5; pass++) {
        for (size_t i = 0; i < L1D_SIZE; i += 64) {
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

typedef int (*small_func_t)(int);

static small_func_t small_funcs[10] = {
    small_func_0, small_func_1, small_func_2, small_func_3,
    small_func_4, small_func_5, small_func_6, small_func_7,
    small_func_8, small_func_9
};

static void mixer_icache_churn(void) {
    uint64_t state = FIXED_SEED;
    volatile int result = 0;

    // Call 10 random functions to gently churn I-cache
    for (int i = 0; i < 10; i++) {
        uint64_t idx = xorshift64(&state) % 10;
        result = small_funcs[idx](result);
    }
}

static void mixer_branch_chaos(void) {
    uint64_t state = FIXED_SEED;
    volatile int result = 0;

    // Execute 250 data-dependent branches (gentle perturbation)
    for (int i = 0; i < 250; i++) {
        uint64_t rnd = xorshift64(&state);
        
        if (rnd & 0x1) result++;
        else result--;
        
        if (rnd & 0x2) result *= 3;
        else result /= 2;
    }
}

static void mixer_tlb_thrash(void) {
    size_t total_size = TLB_PAGES * PAGE_SIZE;
    
    void* buffer = mmap(NULL, total_size, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (buffer == MAP_FAILED) {
        fprintf(stderr, "Failed to mmap for TLB thrash\n");
        return;
    }

    volatile char sink = 0;

    // Touch one byte per page across 64 pages (256KB), 1 pass
    for (size_t i = 0; i < TLB_PAGES; i++) {
        char* page = (char*)buffer + (i * PAGE_SIZE);
        sink += page[0];
        page[0] = (char)(i & 0xFF);
    }

    munmap(buffer, total_size);
}

static void mixer_alu_spin(void) {
    volatile uint64_t result = 1;

    // Execute 1000 ALU operations (gentle computation)
    for (int i = 0; i < 1000; i++) {
        result = result * 6364136223846793005ULL + 1442695040888963407ULL;
        result ^= result >> 13;
    }
}

static void mixer_memory_mixed(void) {
    volatile char* buffer = malloc(MEMORY_MIXED_SIZE);
    if (!buffer) {
        fprintf(stderr, "Failed to allocate memory mixed buffer\n");
        return;
    }

    uint64_t state = FIXED_SEED;
    volatile char sink = 0;

    // Sequential pass over 256KB
    for (size_t i = 0; i < MEMORY_MIXED_SIZE; i += 64) {
        sink += buffer[i];
    }
    
    // 100 random accesses
    for (int i = 0; i < 100; i++) {
        size_t idx = xorshift64(&state) % MEMORY_MIXED_SIZE;
        sink += buffer[idx];
        buffer[idx] = (char)(idx & 0xFF);
    }

    free((void*)buffer);
}

static void mixer_alloc_chaos(void) {
    void* ptrs[50] = {0};
    uint64_t state = FIXED_SEED;

    // Do 50 alloc/free operations with small sizes (64-1024 bytes)
    for (int i = 0; i < 50; i++) {
        uint64_t rnd = xorshift64(&state);
        size_t alloc_size = 64 + (rnd % 961);
        
        ptrs[i] = malloc(alloc_size);
        if (ptrs[i]) {
            memset(ptrs[i], (int)(rnd & 0xFF), alloc_size);
        }
    }

    // Free all allocations
    for (int i = 0; i < 50; i++) {
        if (ptrs[i]) {
            free(ptrs[i]);
        }
    }
}

int context_mixer_run(void) {
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
            mixer_llc_thrash();
            break;
        case 2:
            mixer_l1d_hot();
            break;
        case 3:
            mixer_icache_churn();
            break;
        case 4:
            mixer_branch_chaos();
            break;
        case 5:
            mixer_tlb_thrash();
            break;
        case 6:
            mixer_alu_spin();
            break;
        case 7:
            mixer_memory_mixed();
            break;
        case 8:
            mixer_alloc_chaos();
            break;
        default:
            fprintf(stderr, "Unknown mixer type: %d\n", mixer_type);
            return -1;
    }

    return 0;
}
