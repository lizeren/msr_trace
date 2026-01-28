# Context Mixer Library

A shared library for generating different microarchitectural contexts to control cache, branch predictor, TLB, and other CPU state before running benchmarks or tests.

## Overview

The Context Mixer library provides 8 different "mixers" that gently perturb various microarchitectural states with minimal overhead:

1. **LLC Thrash (cache-cold)** - Touches 64KB (1K cache lines) with 1 pass
2. **L1D Hot (cache-hot baseline)** - Touches 64 cache lines (4KB), 5 passes to keep hot in L1
3. **Instruction-cache churn** - Calls 10 random small functions to gently perturb I-cache/BTB
4. **Branch chaos** - 250 data-dependent branches to scramble branch predictor state
5. **TLB thrash** - Touches 64 pages (256KB), 1 pass to stress TLB
6. **ALU spin (power/frequency baseline)** - 1,000 integer operations for gentle computation
7. **Memory mixed (stream + random)** - Sequential touch of 256KB + 100 random accesses
8. **Alloc chaos** - 50 malloc/free operations (64-1024 bytes) to perturb heap state

All mixers use deterministic behavior (fixed seed) for reproducibility.

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
    // Runs with fixed iterations (very fast)
    context_mixer_run();
    
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
# Run LLC thrash
export MIXER_INDICES=1
./example

# Run branch chaos
export MIXER_INDICES=4
./example

# Run L1D hot
export MIXER_INDICES=2
./example
```

The `MIXER_INDICES` environment variable determines which mixer to run (1-8).

## API Reference

### Function

- `int context_mixer_run(void)` - Run mixer with gentle, fixed iterations
  - Mixer type is read from `MIXER_INDICES` environment variable (1-8)
  - Returns 0 on success, -1 on error
  - If `MIXER_INDICES` not set or invalid, prints warning and returns 0
  - Execution time: typically microseconds (very fast, minimal overhead)
  - Deterministic behavior using fixed seed for reproducibility
  - Small memory footprint (64KB-256KB max per mixer)

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
    context_mixer_run();  // Uses MIXER_INDICES from env
    run_benchmark();
}
```

## Implementation Details

- Gentle perturbations with minimal overhead (microseconds typically)
- Fixed iteration counts for fast, deterministic execution
- Uses fixed seed (`0x123456789ABCDEFULL`) for reproducibility
- Small buffer allocations (64KB-256KB max)
- TLB thrash uses `mmap()` for 64 pages (256KB)
- Alloc chaos uses real `malloc/free` for 50 small allocations (64-1024 bytes)
- All mixers are single-threaded
- PRNG based on xorshift64 with deterministic seed
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
