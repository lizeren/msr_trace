# PMC Batching Guide

## Overview

Measure 100+ events across multiple runs when hardware counters are limited (4 programmable + 2 fixed).

## Hardware Constraints

**Intel CPU (typical):**
- ✅ 2-3 **fixed counters**: CYCLES, INSTRUCTIONS (always available)
- ⚠️ 4 **programmable counters**: Shared resource
- ❌ Cannot measure 100+ events simultaneously

**Solution:** Run your test multiple times, each measuring a different batch of 4 events + fixed counters.

## CSV File Format

`pmc_events.csv` contains ALL events:

```csv
index,event_name,mode,sample_period,priority
0,BR_INST_RETIRED.NEAR_CALL,sampling,10,normal
1,BR_INST_RETIRED.CONDITIONAL,counting,0,normal
2,MEM_LOAD_RETIRED.L1_MISS,sampling_freq,2000,normal
...
10,CPU_CLK_UNHALTED.THREAD,counting,0,fixed
11,INST_RETIRED.ANY,counting,0,fixed
```

**Columns:**
- `index`: Numeric index for selection (all events have an index)
- `event_name`: Intel event name (e.g., `BR_INST_RETIRED.NEAR_CALL`)
- `mode`: `counting`, `sampling`, or `sampling_freq`
- `sample_period`: N for sampling modes (0 for counting)
- `priority`: `fixed` = always included, `normal` = selectable

**Note:** Fixed counters (priority=fixed) are always included regardless of PMC_EVENT_INDICES.

## Event Selection

### Environment Variable: `PMC_EVENT_INDICES` (REQUIRED)

**You MUST specify which events to measure:**

```bash
# Measure events 0,1,2,3 + fixed counters (10,11)
PMC_EVENT_INDICES="0,1,2,3" ./my_test

# Explicitly include fixed counters (same result as above)
PMC_EVENT_INDICES="0,1,2,3,10,11" ./my_test

# Measure events 4,5,6,7 + fixed counters
PMC_EVENT_INDICES="4,5,6,7" ./my_test

# ERROR: Not specifying indices will fail
./my_test  # ❌ Error: "need to specify event index from the excel"
```

**Fixed counters (priority=fixed) are ALWAYS included** even if not in the list.

**Why this is required:**
- Prevents accidentally measuring all 100+ events at once
- Ensures explicit control over hardware counter allocation
- Forces proper batching workflow for large event sets

**Benefits of numeric indices for fixed counters:**
- Simpler CSV structure (all rows have numeric indices)
- Easy Excel/spreadsheet navigation by row number
- Can explicitly list them if desired: `"0,1,2,3,10,11"`

## Automated Batching

Use the provided script to automatically batch measurements:

```bash
./run_batched_pmc.sh ./my_instrumented_test output_dir
```

**What it does:**
1. Counts total programmable events in CSV (e.g., 100 events)
2. Divides into batches of 4 (e.g., 25 batches)
3. Runs your test 25 times with different `PMC_EVENT_INDICES`
4. Outputs: `output_dir/pmc_results_batch_0.json`, `batch_1.json`, ...

**Example:**
```bash
$ ./run_batched_pmc.sh ./example_cache_call results

=========================================
PMC Batch Measurement
=========================================
Test binary: ./example_cache_call
Total programmable events: 100
Batch size: 4
Number of batches: 25
Output directory: results
=========================================

========== Batch 0/24 ==========
Measuring events: 0,1,2,3
PMC: Filtering to 4 event indices
...

========== Batch 1/24 ==========
Measuring events: 4,5,6,7
PMC: Filtering to 4 event indices
...
```

## Aggregating Results

Merge all batch JSONs into one comprehensive report:

```bash
python3 aggregate_pmc_results.py results/ final_results.json
```

**Features:**
- Combines all events from all batches
- Detects execution variation (compares CYCLES across batches)
- Warns if variation >5% (indicates unstable workload)

**Example output:**
```
Found 25 batch files
Processing results/pmc_results_batch_0.json...
Processing results/pmc_results_batch_1.json...
...

Cycle counts across batches:
  Batch 0: 1,234,567 cycles (0.12% deviation)
  Batch 1: 1,235,123 cycles (0.16% deviation)
  ...

Maximum variation: 0.82%

============================================================
Aggregation complete!
  Total batches: 25
  Total events: 125  (100 programmable + 25×2 fixed duplicates)
  Output file: final_results.json
============================================================
```

## Integration with LLVM Instrumentation

Your LLVM pass inserts:

```c
void my_function() {
    pmc_measure_begin_csv("my_function", "pmc_events.csv");
    // ... original code ...
    pmc_measure_end(handle, 1);
}
```

**No code changes needed!** The library reads `PMC_EVENT_INDICES` at runtime.

## Workflow Example

```bash
# 1. Create master CSV with ALL 100+ events
vim pmc_events.csv

# 2. LLVM instruments your code (once)
# Your code now has pmc_measure_begin/end calls

# 3. Compile
make

# 4. Run batched measurements
./run_batched_pmc.sh ./my_test batch_results

# 5. Aggregate
python3 aggregate_pmc_results.py batch_results/ comprehensive.json

# 6. Analyze
cat comprehensive.json
# or
python3 analyze.py comprehensive.json
```

## Manual Batching

For custom workflows:

```bash
# Batch 0: Events 0-3
PMC_EVENT_INDICES="0,1,2,3" \
PMC_OUTPUT_FILE="batch_0.json" \
./my_test

# Batch 1: Events 4-7
PMC_EVENT_INDICES="4,5,6,7" \
PMC_OUTPUT_FILE="batch_1.json" \
./my_test

# Aggregate
python3 aggregate_pmc_results.py . final.json
```

## Tips for Stable Measurements

To minimize variation across batches:

```bash
# Pin to specific CPU
taskset -c 0 ./run_batched_pmc.sh ./my_test results

# Disable frequency scaling
sudo cpupower frequency-set --governor performance

# Reduce system noise
# - Close other applications
# - Disable unnecessary services
# - Use dedicated benchmarking machine
```

## Environment Variables Reference

| Variable | Purpose | Example |
|----------|---------|---------|
| `PMC_EVENT_INDICES` | Select which events to measure | `"0,1,2,3"` |
| `PMC_OUTPUT_FILE` | Output JSON filename | `"batch_5.json"` |

## See Also

- `pmc.h` - API documentation
- `pmc_events.csv` - Event definitions
- `run_batched_pmc.sh` - Automated batching script
- `aggregate_pmc_results.py` - Result aggregation

