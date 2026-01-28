#include "context_mixer.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

static void print_usage(const char* prog) {
    printf("Usage: Set MIXER_INDICES env var, then run:\n");
    printf("  export MIXER_INDICES=1 && %s\n\n", prog);
    printf("Mixer types:\n");
    printf("  1 - LLC Thrash (cache-cold)\n");
    printf("  2 - L1D Hot (cache-hot baseline)\n");
    printf("  3 - Instruction-cache churn\n");
    printf("  4 - Branch chaos\n");
    printf("  5 - TLB thrash\n");
    printf("  6 - ALU spin (power/frequency baseline)\n");
    printf("  7 - Memory mixed (stream + random)\n");
    printf("  8 - Alloc chaos\n");
    printf("\nExample:\n");
    printf("  export MIXER_INDICES=1 && %s\n", prog);
    printf("  (Run LLC thrash)\n\n");
    printf("  export MIXER_INDICES=4 && %s\n", prog);
    printf("  (Run branch chaos)\n");
}

static const char* get_mixer_name(int mixer_type) {
    switch (mixer_type) {
        case 1: return "LLC Thrash (cache-cold)";
        case 2: return "L1D Hot (cache-hot baseline)";
        case 3: return "Instruction-cache churn";
        case 4: return "Branch chaos";
        case 5: return "TLB thrash";
        case 6: return "ALU spin (power/frequency baseline)";
        case 7: return "Memory mixed (stream + random)";
        case 8: return "Alloc chaos";
        default: return "Unknown";
    }
}

static void run_mixer(void) {
    const char* mixer_env = getenv("MIXER_INDICES");
    int mixer_type = mixer_env ? atoi(mixer_env) : 0;
    
    printf("Running: %s\n", get_mixer_name(mixer_type));
    
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    int ret = context_mixer_run();
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    long elapsed_ns = (end.tv_sec - start.tv_sec) * 1000000000L + 
                      (end.tv_nsec - start.tv_nsec);
    long elapsed_us = elapsed_ns / 1000;
    
    if (ret == 0) {
        printf("Completed successfully in %ld us\n\n", elapsed_us);
    } else {
        printf("Failed with error code: %d\n\n", ret);
    }
}

int main(int argc, char* argv[]) {
    const char* mixer_env = getenv("MIXER_INDICES");
    
    if (!mixer_env) {
        printf("Error: MIXER_INDICES environment variable not set\n\n");
        print_usage(argv[0]);
        return 1;
    }

    printf("Context Mixer Example\n");
    printf("=====================\n");
    printf("MIXER_INDICES=%s\n\n", mixer_env);

    run_mixer();

    printf("Done.\n");
    return 0;
}
