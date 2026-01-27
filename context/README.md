# Context Mixer Library

A shared library for generating different microarchitectural contexts to control cache, branch predictor, TLB, and other CPU state before running benchmarks or tests.

## Overview

The Context Mixer library provides 8 different "mixers" that perturb various microarchitectural states:

1. **LLC Thrash (cache-cold)** - Touches a large buffer (64-256MB) to evict all prior cache state
2. **L1D Hot (cache-hot baseline)** - Repeatedly touches a small buffer (~32KB) to create stable L1 cache state  
3. **Instruction-cache churn** - Calls many small functions in pseudo-random order to perturb I-cache/BTB
4. **Branch chaos** - Data-dependent branches driven by PRNG to scramble branch predictor state
5. **TLB thrash** - Touches one byte per page across large mapping to stress TLB/page walks
6. **ALU spin (power/frequency baseline)** - Tight integer arithmetic loop for consistent frequency/power
7. **Memory mixed (stream + random)** - Combines sequential and random accesses to affect prefetchers
8. **Alloc chaos** - Allocates/frees randomized sizes to perturb heap state and alignment

## Building

```bash
make
```

This builds:
- `libcontext_mixer.so` - The shared library
- `example` - Example program with environment variable support

## Installation

```bash
sudo make install
```

Installs the library to `/usr/local/lib` and header to `/usr/local/include`.

## Usage

### C API

```c
#include <context_mixer.h>

int main() {
    // Mixer type is read from MIXER_INDICES environment variable
    
    // Run for 1 second (mixer type determined by env var)
    context_mixer_run(1000000);
    
    // Run for 500ms
    context_mixer_run(500000);
    
    return 0;
}
```

Set environment variable and compile:
```bash
export MIXER_INDICES=1
gcc -o myprogram myprogram.c -lcontext_mixer
./myprogram
```

### Command Line Example

```bash
# Run LLC thrash for 500ms
export MIXER_INDICES=1
./example 500000

# Run branch chaos for 250ms
export MIXER_INDICES=4
./example 250000

# Run L1D hot for 1 second
export MIXER_INDICES=2
./example 1000000
```

The `MIXER_INDICES` environment variable determines which mixer to run (1-8).

## API Reference

### Function

- `int context_mixer_run(size_t duration_us)` - Run mixer for specified duration (microseconds)
  - Mixer type is read from `MIXER_INDICES` environment variable (1-8)
  - Returns 0 on success, -1 on error
  - If `MIXER_INDICES` not set or invalid, prints warning and returns 0

## Use Cases

### Before Performance Testing
Use mixers to establish consistent microarchitectural state before measurements:

```bash
# Create cache-cold state
export MIXER_INDICES=1
./myprogram  # Runs with LLC thrash context

# Create cache-hot state  
export MIXER_INDICES=2
./myprogram  # Runs with L1D hot context
```

### In Side-Channel Research
Establish known states for reproducible side-channel experiments:

```bash
# Clear all branch predictor state
export MIXER_INDICES=4
./experiment

# Establish baseline power/frequency
export MIXER_INDICES=6
./experiment
```

### For Benchmark Stability
Reduce variance by establishing consistent microarchitectural context:

```c
// Set in your program
for (int trial = 0; trial < 100; trial++) {
    context_mixer_run(500000);  // Uses MIXER_INDICES from env
    run_benchmark();
}
```

## Implementation Details

- Uses `rdtsc()` for precise timing control
- Allocates buffers internally with sensible defaults
- TLB thrash uses `mmap()` for large page allocations
- All mixers are single-threaded
- PRNG based on xorshift64 with time-based seed
- No initialization or cleanup needed - just call and go

## Testing

```bash
make test
```

Runs the example program with all mixers.

## Cleaning

```bash
make clean
```

Removes build artifacts.

## License

This is a utility library for MSR trace collection and performance analysis.
