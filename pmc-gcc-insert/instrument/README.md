# GCC Plugin for PMC Instrumentation

A GCC plugin that automatically instruments function calls with Performance Monitoring Counter (PMC) measurements using libpmc, with intelligent handling of nested calls to avoid double-counting.

## Overview

This plugin wraps calls to specified "target" functions with PMC measurement calls:

```c
// Original code:
main() {
    addition(5, 3);
}

// After instrumentation:
main() {
    pmc_multi_handle_t *handle = pmc_measure_begin_csv("addition", "pmc_events.csv");
    addition(5, 3);
    pmc_measure_end(handle, 1);
}
```

## Key Feature: Transitive Exclusion

The plugin uses **static call graph analysis** to prevent nested measurements when target functions call each other directly or indirectly.

### The Problem

Without exclusion logic, nested calls would be double-counted: Let's say addition() and subtraction() are targets, and helper_func() is a non-target.

```
main() → addition()          [START measure addition]
main() → subtraction()            [START measure subtraction] 
          → helper_func()          
            → addition()           [START measure addition]      ← NESTED!
```

### The Solution: Exclusion Zones

The plugin builds a **two-pass call graph**:

1. **Pass 1**: Scans all functions to record who calls whom
2. **Pass 2**: Computes an "exclusion zone" and instruments only calls from outside this zone

**Exclusion Zone = Targets + Non-targets transitively reachable from targets**

**Rule**: Don't instrument calls **FROM** any function in the exclusion zone.

### Example Scenarios

#### Scenario 1: Two Targets

```
Targets: addition, subtraction
Call graph:
  main → addition
  main → subtraction
  subtraction → helper_func → addition
```

**Exclusion Zone:**
- Targets: `addition`, `subtraction`
- Non-targets: `helper_func` (called by subtraction)

**Result:**
- `main() → addition()`: ✓ **MEASURED** (main not in zone)
- `main() → subtraction()`: ✓ **MEASURED** (main not in zone)
- `subtraction() → helper_func()`: ✗ Skipped (subtraction in zone)
- `helper_func() → addition()`: ✗ Skipped (helper_func in zone)

#### Scenario 2: All Three are Targets

```
Targets: addition, subtraction, helper_func
```

**Exclusion Zone:**
- Targets: `addition`, `subtraction`, `helper_func`
- Non-targets: (none)

**Result:**
- `main() → addition()`: ✓ **MEASURED**
- `main() → subtraction()`: ✓ **MEASURED**
- `main() → helper_func()`: ✓ **MEASURED**
- All internal calls skipped (callers in zone)

### Trade-off: Over-skipping

If `main()` directly calls a non-target in the exclusion zone:

```
main() → helper_func() → addition()
```

The call to `addition()` is **NOT measured** because `helper_func` is in the exclusion zone. This is by design to prevent the nested measurement problem.

## Build

```bash
cd pmc-gcc-insert/instrument
make
```

This produces `instrument_callsites_plugin.so`.

## Usage

### Basic Command

```bash
gcc -O0 -fno-inline \
    -fplugin=./instrument_callsites_plugin.so \
    -fplugin-arg-instrument_callsites_plugin-include-function-list=func1,func2 \
    your_program.c -o your_program \
    -L/path/to/libpmc -lpmc -ldl -lpthread
```

### Plugin Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `include-function-list` | Comma-separated list of target functions (REQUIRED) | `addition,subtraction` |
| `csv-path` | Path to PMC events CSV file | `pmc_events.csv` (default) |
| `debug` | Enable verbose debug output | (flag, no value) |

### CSV File Format

The CSV file specifies which PMC events to measure. See `pmc-gcc-insert/test/pmc_events.csv` for format.

Must set `PMC_EVENT_INDICES` environment variable:

```bash
export PMC_EVENT_INDICES="0,1,2,3"
```

### Complete Example

```bash
# Build with instrumentation
gcc -O0 -Wall -fno-inline \
    -fplugin=../instrument/instrument_callsites_plugin.so \
    -fplugin-arg-instrument_callsites_plugin-include-function-list=addition,subtraction \
    -fplugin-arg-instrument_callsites_plugin-csv-path=pmc_events.csv \
    -fplugin-arg-instrument_callsites_plugin-debug \
    test.c -o test \
    -L.. -lpmc -ldl -lpthread -Wl,-rpath,'$ORIGIN/..'

# Run with PMC measurement
PMC_EVENT_INDICES="0,1,2,3" ./test

# Results are in pmc_results.json
cat pmc_results.json
```

## Debug Output

With `-fplugin-arg-instrument_callsites_plugin-debug`, the plugin shows:

```
╔════════════════════════════════════════════════════════════════
║ PMC Instrumentation Plugin Configuration
╠════════════════════════════════════════════════════════════════
║ API: pmc_measure_begin_csv() / pmc_measure_end()
║ CSV: pmc_events.csv (default)
║ Output: pmc_results.json (always enabled)
║ Targets (2):
║   • addition
║   • subtraction
╚════════════════════════════════════════════════════════════════

╔════════════════════════════════════════════════════════════════
║ Call Graph Analysis (Direct Calls Only)
╚════════════════════════════════════════════════════════════════
  addition:
    (no calls recorded)
  subtraction:
    └─> addition
    └─> helper_func

╔════════════════════════════════════════════════════════════════
║ Instrumentation Strategy
╠════════════════════════════════════════════════════════════════
║ EXCLUSION ZONE (calls FROM these functions won't be instrumented):
║
║   TARGETS (skip their internal calls): 2
║     • addition
║     • subtraction
║
║   NON-TARGETS (transitively reachable from targets): 1
║     • helper_func
║
║ RESULT:
║   • Targets WILL be measured when called from outside exclusion zone
║   • Calls FROM exclusion zone are NOT instrumented (avoids nesting)
╚════════════════════════════════════════════════════════════════

⊗ SKIP [TARGET]: addition
⊗ SKIP [TRANSITIVE]: helper_func (called by target)
    └─> helper_func() calls addition() [nested - skipped]
⊗ SKIP [TARGET]: subtraction
    └─> subtraction() calls addition() [nested - skipped]
✓ INSTRUMENT: main() → addition()
✓ INSTRUMENT: main() → subtraction()
```

## Output

Results are always exported to JSON (default: `pmc_results.json`):

```json
{
  "measurements": [
    {
      "label": "addition",
      "timestamp": "2026-01-06T23:10:53",
      "num_events": 4,
      "events": [
        {
          "event_name": "BR_MISP_RETIRED.ALL_BRANCHES",
          "mode": "sampling",
          "count": 27,
          "num_samples": 0
        }
      ]
    }
  ]
}
```

Override output file:
```bash
PMC_OUTPUT_FILE="my_results.json" ./test
```

## Implementation Details

### Two-Pass Architecture

1. **Pass 1** (`build_pmc_call_graph`):
   - Runs early (after CFG construction)
   - Scans GIMPLE to record all direct function calls
   - Builds global call graph map

2. **Pass 2** (`instrument_callsites`):
   - Runs later (before `veclower`)
   - Computes transitive closure of reachable functions
   - Instruments calls to targets from outside exclusion zone

### Limitations

- **Direct calls only**: Function pointers and virtual calls are not tracked
- **Compile-time only**: Cannot handle dynamic linking or runtime-loaded code
- **Over-skipping**: Targets called through excluded non-targets won't be measured
- **Per-compilation-unit**: Call graph is built per `.c` file

## Requirements

- GCC with plugin support (tested on GCC 11)
- libpmc (Performance Monitoring Counter library)
- Linux with `perf_event_open` support

## Integration with libpmc

The plugin generates calls to libpmc's simplified API:

```c
// Start measurement
pmc_multi_handle_t* pmc_measure_begin_csv(
    const char *label,      // Function name
    const char *csv_path    // Path to CSV (NULL = default)
);

// End measurement and export to JSON
void pmc_measure_end(
    pmc_multi_handle_t *handle,
    int report              // Always 1 (export enabled)
);
```

See `pmc-gcc-insert/pmc.h` for full libpmc API documentation.

## Troubleshooting

### "undefined symbol: _ZTI8opt_pass"

The plugin must be built with `-fno-rtti`. Check `Makefile` includes this flag.

### No measurements in JSON

1. Check `PMC_EVENT_INDICES` is set
2. Verify CSV file path is correct
3. Run with `debug` flag to see what's being instrumented
4. Check that target functions are actually called from non-excluded functions

### "Label already measured" warnings

libpmc deduplicates by label. Each unique label is only measured once per run. This is expected behavior.

## See Also

- `pmc-gcc-insert/pmc.h` - libpmc API documentation
- `pmc-gcc-insert/test/` - Example usage and test programs
- `libpmc/` - PMC library implementation
