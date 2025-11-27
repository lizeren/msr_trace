# PMC Feature Collection - Quick Start Guide

## Overview

Automatically collects temporal features from **ALL PMC events** and **ALL workloads** by measuring **1 event at a time**.

## Usage

### Single Command - That's It!

```bash
# Collects ALL events from pmc_events.csv and ALL workloads from target program
# Automatically deletes old results and builds complete feature file
python3 collect_pmc_features.py --target ./example_cache_call --runs 10
```

**What it does:**
1. Deletes old `temporal_features.json`
2. Reads all events from `pmc_events.csv`
3. For each event:
   - Runs target program 10 times
   - Averages the results
   - Appends to `temporal_features.json`
4. Result: One complete JSON file with all events and workloads


## Cleanup

The script automatically deletes `pmc_results.json` at the end of collection.

## Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--target` | Yes | - | Path to target binary |
| `--runs` | No | 10 | Number of runs per batch |

## Important Notes

1. **Both sampling modes work**: Events with `mode=sampling` or `mode=sampling_freq` in CSV are collected
2. **All workloads extracted**: Automatically extracts ALL workloads from target program output
3. **One event at a time**: Events are measured individually to avoid hardware conflicts
4. **Auto-cleanup**: `pmc_results.json` is deleted after collection
5. **Averages runs**: Multiple runs are averaged using median-length run as template
6. **Event index mapping**: Event names are mapped back to their CSV index numbers
7. **Fixed files**: Always reads from `pmc_events.csv` and writes to `temporal_features.json`

## Example Workflow

```bash
# 1. Build target
make example_cache_call

# 2. Collect features - ONE COMMAND
python3 collect_pmc_features.py --target ./example_cache_call --runs 10
```

**That's it!** No need to worry about deleting old files, appending, or multiple commands.
