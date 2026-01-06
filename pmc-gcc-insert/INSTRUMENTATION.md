# Function Instrumentation with -finstrument-functions

This library supports automatic function instrumentation using GCC's `-finstrument-functions` flag. When enabled, the library automatically measures performance counters for functions listed in `api_lists.csv`.

## How It Works

When you compile your application with `-finstrument-functions`, GCC automatically inserts calls to:
- `__cyg_profile_func_enter()` at the beginning of every function
- `__cyg_profile_func_exit()` at the end of every function

The library implements these functions to:
1. Check if the function name is in `api_lists.csv`
2. If yes, call `pmc_measure_begin_csv()` on entry
3. Call `pmc_measure_end()` on exit

## Setup

### 1. Create `api_lists.csv`

Create a file named `api_lists.csv` (or specify a different path via `PMC_API_LIST` environment variable) with one function name per line:

```
addition
subtraction
multiplication
```

Trailing commas are automatically removed.

### 2. Compile Your Application

Compile your application with `-finstrument-functions` and link against `libpmc.so`:

**IMPORTANT**: You must use `-rdynamic` (or `-Wl,-export-dynamic`) to export symbols so the library can resolve function names:

```bash
gcc -finstrument-functions -O2 -o myapp myapp.c -L. -lpmc -ldl -lpthread -Wl,-rpath,'$$ORIGIN' -rdynamic
```

**Note**: If functions are being inlined by the optimizer, instrumentation won't work. Consider using `-O0` or `-fno-inline` for debugging:

```bash
gcc -finstrument-functions -O0 -fno-inline -o myapp myapp.c -L. -lpmc -ldl -lpthread -Wl,-rpath,'$$ORIGIN' -rdynamic
```

### 3. Set Environment Variables

Before running your application, set the required environment variables:

```bash
# Required: Select which events to measure from pmc_events.csv
export PMC_EVENT_INDICES="0,1,2"

# Optional: Path to CSV file with events (default: pmc_events.csv)
export PMC_CSV_PATH="pmc_events.csv"

# Optional: Path to function list (default: api_lists.csv)
export PMC_API_LIST="api_lists.csv"

# Optional: Auto-report results (default: 0/off)
export PMC_AUTO_REPORT=1

# Optional: Output file for JSON results (default: pmc_results.json)
export PMC_OUTPUT_FILE="pmc_results.json"
```

### 4. Run Your Application

```bash
./myapp
```

The library will automatically:
- Measure functions listed in `api_lists.csv`
- Export results to JSON (if `PMC_AUTO_REPORT=1`, also print to stdout)
- Handle nested function calls correctly using a per-thread stack

## Example

See `test/instrumented_test.c` for a complete example:

```bash
cd test
make
export PMC_EVENT_INDICES="0,1,2" && ./instrumented_test
```

## Important Notes

1. **Function Name Matching**: The library uses `dladdr()` to resolve function names. Only functions that can be resolved by `dladdr()` will be instrumented.

2. **Nested Calls**: The library handles nested function calls correctly using a per-thread stack. If a function in the list calls another function in the list, both will be measured independently.

3. **Performance Overhead**: Instrumentation adds overhead to every function call. Only functions in `api_lists.csv` will trigger PMC measurements, but the symbol resolution happens for all functions.

4. **Thread Safety**: The handle stack is thread-local, so instrumentation works correctly in multi-threaded applications.

5. **Stack Overflow Protection**: The handle stack is limited to 64 nested calls. If exceeded, a warning is printed and instrumentation is skipped for that call.

## Troubleshooting

- **No measurements**: 
  - Check that function names in `api_lists.csv` exactly match the function names in your code (use `nm` or `objdump` to verify).
  - Make sure you're using `-rdynamic` when linking your application.
  - Verify that functions aren't being inlined (use `-O0` or `-fno-inline`).
  - Enable debug mode: `export PMC_DEBUG=1` to see what's happening.

- **Missing symbols**: 
  - Make sure your application is compiled with `-rdynamic` to export symbols.
  - Verify symbols exist: `nm myapp | grep function_name`

- **Permission errors**: 
  - Ensure `perf_event_paranoid` allows access or run with appropriate permissions.
  - Check that `PMC_EVENT_INDICES` is set correctly.

- **Functions not being instrumented**:
  - Use `export PMC_DEBUG=1` to see which functions are being checked.
  - Verify `api_lists.csv` is in the current directory or set `PMC_API_LIST` environment variable.

