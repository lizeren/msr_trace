#!/usr/bin/env python3
"""
PMC Temporal Feature Collector

Collects performance counter data by running a target program multiple times
with different PMC events (1 at a time), extracting temporal features for ML classification.

Automatically reads events from pmc_events.csv and measures them individually.

Usage:
    python3 collect_pmc_features.py --target ./example_cache_call --runs 5 --total 2
"""

import json
import subprocess
import os
import sys
import argparse
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


class PMCTemporalCollector:
    def __init__(self, target_binary: str, csv_file: str = "pmc_events.csv", 
                 output_json: str = "pmc_features.json",
                 pmc_output: str = "pmc_results.json"):
        self.target_binary = target_binary
        self.csv_file = csv_file
        self.output_json = output_json
        self.pmc_output = pmc_output
        self.feature_db = {}
        
        # Verify target exists
        if not os.path.exists(target_binary):
            raise FileNotFoundError(f"Target binary not found: {target_binary}")
        
        # Verify CSV exists
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        
        # Cache event name to index mapping (optimization: read CSV only once)
        self.event_name_map = self.build_event_name_to_index_map()
        print(f"✓ Cached {len(self.event_name_map)} event mappings from {csv_file}")
    
    def read_event_indices_from_csv(self) -> List[int]:
        """Read a CSV file and returns a list of integers taken from the "index" column of that file."""
        event_indices = []
        try:
            with open(self.csv_file, 'r') as f:
                reader = csv.DictReader(f) # read each row of the CSV as a Python dictionary.
                for row in reader:
                    if row.get('index'): # checks whether there is a value in the "index" column.
                        idx = int(row['index'])
                        event_indices.append(idx)
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV file: {e}")
        
        return event_indices
    
    def build_event_name_to_index_map(self) -> Dict[str, int]:
        """Build mapping from event name to CSV index."""
        event_map = {}
        try:
            with open(self.csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('index') and row.get('event_name'):
                        idx = int(row['index'])
                        name = row['event_name'].strip()
                        event_map[name] = idx
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV file: {e}")
        
        return event_map
    
    def collect_events(self, event_indices: List[int], num_runs: int = 5) -> int:
        """
        Collect temporal data for multiple events simultaneously across multiple runs.
        Extracts data for ALL workloads found in the target program output.
        
        WORKFLOW:
        1. Run target program multiple times (num_runs)
        2. Each run measures the specified events (event_indices) simultaneously
        3. Extract temporal features from each run's PMC output
        4. Reorganize data: runs->workloads->events becomes workloads->events->runs
        5. Average temporal data across all runs for each workload+event combination
        6. Store averaged results in feature database
        
        Args:
            event_indices: List of event indices to measure simultaneously
            num_runs: Number of times to run the target program
            
        Returns:
            Number of successful runs
        """
        event_str = ','.join(map(str, event_indices))
        print(f"\n{'='*60}")
        print(f"Collecting Events {event_str} for ALL workloads")
        print(f"Target: {self.target_binary}")
        print(f"Runs: {num_runs}")
        print(f"{'='*60}")
        
        # run_data will accumulate temporal features from each successful run
        # Structure: List of dictionaries, one per run
        # [
        #   {workload1: [event_data1, event_data2], workload2: [event_data1]},  # Run 1
        #   {workload1: [event_data1, event_data2], workload2: [event_data1]},  # Run 2
        #   ...
        # ]
        run_data = []
        
        for run_id in range(num_runs):
            print(f"\n[Run {run_id+1}/{num_runs}]", end=" ")
            
            # Delete previous PMC output from libpmc to avoid appending
            if os.path.exists(self.pmc_output):
                os.remove(self.pmc_output)
                # print(f"Cleaned old {self.pmc_output}", end=" -> ")
            
            # Set environment variables (multiple events comma-separated)
            env = os.environ.copy()
            env['PMC_EVENT_INDICES'] = event_str
            env['PMC_OUTPUT_FILE'] = self.pmc_output
            
            # Run target program
            try:
                result = subprocess.run(
                    [self.target_binary],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=30  # 30 second timeout
                )
                
                if result.returncode != 0:
                    print(f"FAILED (exit code {result.returncode})")
                    if result.stderr:
                        print(f"  Error: {result.stderr[:200]}")
                    continue
                
            except subprocess.TimeoutExpired:
                print("TIMEOUT")
                continue
            except Exception as e:
                print(f"ERROR: {e}")
                continue
            
            # Read PMC results from JSON file
            if not os.path.exists(self.pmc_output):
                print(f"FAILED (no output file)")
                continue
            
            try:
                with open(self.pmc_output) as f:
                    pmc_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"FAILED (invalid JSON: {e})")
                continue
            
            # Extract temporal features for all events and all workloads
            # This returns: {workload_label: [event_data1, event_data2, ...]}
            # Each event_data contains timestamps, counts, and other temporal info for one event
            workload_data_dict = self.extract_all_temporal_features(pmc_data, target_label=None)
            
            if workload_data_dict:
                run_data.append(workload_data_dict)
                total_events = sum(len(events) for events in workload_data_dict.values())
                num_workloads = len(workload_data_dict)
                print(f"SUCCESS ({num_workloads} workloads, {total_events} events total)")
            else:
                print("FAILED (no temporal data)")
        
        # We have completed the user-specified number of runs of the target program.
        # By now we should have samples for the current event batch from all runs in the memory
        # Now we need to average the results across all runs and organize by workload and event.
        # In this method we store the averaged results in the feature database (an attribute of the class).
        
        # DATA STRUCTURE at this point:
        # run_data = [
        #   {workload1: [event_data1, event_data2], workload2: [event_data1]},  # Run 1
        #   {workload1: [event_data1, event_data2], workload2: [event_data1]},  # Run 2
        #   ...
        # ]
        # Each run contains multiple workloads, each workload contains data for multiple events
        
        if run_data:
            print(f"\n✓ Collected {len(run_data)}/{num_runs} successful runs")
            
            # STEP 1: Reorganize data structure from "runs->workloads->events" to "workloads->events->runs"
            # This makes it easier to average all runs for the same workload and event combination
            
            # TARGET STRUCTURE:
            # workloads_events = {
            #   "workload1": {
            #     event_idx_0: [event_data_run1, event_data_run2, ...],
            #     event_idx_1: [event_data_run1, event_data_run2, ...],
            #   },
            #   "workload2": {
            #     event_idx_0: [event_data_run1, event_data_run2, ...],
            #   }
            # }
            workloads_events = {}
            
            # Iterate through each run's data
            for run in run_data:
                # Each run contains multiple workloads (e.g., "RSA_encrypt", "RSA_decrypt")
                for workload_label, events_list in run.items():
                    # Create entry for this workload if it doesn't exist yet
                    if workload_label not in workloads_events:
                        workloads_events[workload_label] = {}
                    
                    # Each workload has multiple events (e.g., cache-misses, instructions, etc.)
                    for event_data in events_list:
                        event_idx = event_data['event_index']
                        # Create list for this event if it doesn't exist yet
                        # (sometimes a target function doesn't have data for this event becuase function too short)
                        if event_idx not in workloads_events[workload_label]:
                            workloads_events[workload_label][event_idx] = []
                        # Append this run's data for this workload and event
                        workloads_events[workload_label][event_idx].append(event_data)
            
            # STEP 2: Average temporal data across all runs for each workload+event combination
            # Then store the averaged result in the feature database
            # Sort events by index to ensure consistent ordering in output
            for workload_label, events_by_index in workloads_events.items():
                for event_idx in sorted(events_by_index.keys()):  # Sort by event index
                    event_runs = events_by_index[event_idx]
                    # event_runs is a list of temporal data from all runs for this workload and event
                    # average_temporal_runs() will compute mean timestamps, count statistics, etc.
                    averaged = self.average_temporal_runs(event_runs)
                    # Store the averaged temporal features in the feature database
                    self.add_to_feature_db(workload_label, event_idx, averaged)
        else:
            print(f"\n✗ No successful runs for events {event_str}")
        
        return len(run_data)
    
    def extract_all_temporal_features(self, pmc_data: dict, target_label: str = None) -> Dict[str, List[dict]]:
        """
        Extract temporal information for ALL events from ALL workloads in PMC JSON output.
        Only processes sampling mode events.
        
        INPUT STRUCTURE (pmc_data from pmc_results.json):
        {
          "measurements": [
            {
              "label": "RSA_encrypt",
              "events": [
                {
                  "event_name": "cache-misses",
                  "mode": "sampling",
                  "count": 12345,
                  "samples": [
                    {"time": 1000000, "cpu": 0},
                    {"time": 1500000, "cpu": 0},
                    ...
                  ]
                },
                ...
              ]
            },
            {
              "label": "RSA_decrypt",
              "events": [...]
            }
          ]
        }
        
        OUTPUT STRUCTURE (returned dict):
        {
          "RSA_encrypt": [
            {
              "event_index": 0,
              "event_name": "cache-misses",
              "mode": "sampling",
              "total_count": 12345,
              "num_samples": 50,
              "timestamps_ns": [0, 500000, 1000000, ...],  # normalized to t=0
              "cpus": [0, 0, 1, ...],
              "total_duration_ns": 5000000
            },
            ...  # more events for RSA_encrypt
          ],
          "RSA_decrypt": [...]  # events for RSA_decrypt
        }
        
        Args:
            pmc_data: Parsed JSON from PMC output file
            target_label: Function/workload label to match (None = extract all workloads)
            
        Returns:
            Dictionary mapping workload labels to lists of event data
        """
        # Use cached event name to CSV index mapping (optimization: no file I/O)
        event_name_map = self.event_name_map
        
        workload_data = {}  # {workload_label: [event_data1, event_data2, ...]}
        measurements = pmc_data.get('measurements', [])
        
        for measurement in measurements:
            workload_label = measurement.get('label')
            
            # Filter by target_label if specified
            if target_label and workload_label != target_label:
                continue
            
            # Initialize list for this workload if not exists
            if workload_label not in workload_data:
                workload_data[workload_label] = []
            
            events = measurement.get('events', [])
            for event in events:
                mode = event.get('mode')
                
                # Only process sampling modes (skip counting mode and sampling_freq mode)
                if mode not in ['sampling']:
                    continue
                
                samples = event.get('samples', [])
                if not samples:
                    continue
                
                # Map event name to CSV index
                event_name = event['event_name'] # extract event names from JSON entry "event_name"
                event_idx = event_name_map.get(event_name, -1)
                
                if event_idx == -1:
                    print(f"  Warning: Event '{event_name}' not found in CSV, skipping")
                    continue
                
                # Extract event info
                event_info = {
                    'event_index': event_idx,
                    'event_name': event['event_name'],
                    'mode': mode,
                    'sampling_period': event.get('sample_period', 0),
                    'total_count': event['count'],
                    'num_samples': len(samples)
                }
                
                # Extract temporal sequence (normalize to t=0)
                t0 = samples[0]['time']
                event_info['timestamps_ns'] = [s['time'] - t0 for s in samples]
                event_info['cpus'] = [s['cpu'] for s in samples] # keep track how many CPUs are involved in this workload
                
                # Derived features
                event_info['total_duration_ns'] = event_info['timestamps_ns'][-1]
                
                workload_data[workload_label].append(event_info)
        
        return workload_data
    
    def average_temporal_runs(self, run_data: List[dict]) -> dict:
        """
        Average temporal data across multiple runs.
        Uses the median-length run as template and aligns others.
        
        INPUT STRUCTURE (run_data):
        [
          {  # Run 1
            "event_index": 0,
            "event_name": "cache-misses",
            "total_count": 12345,
            "num_samples": 50,
            "timestamps_ns": [0, 500000, 1000000, ...],
            "total_duration_ns": 5000000
          },
          {  # Run 2 (might have different number of samples)
            "event_index": 0,
            "event_name": "cache-misses",
            "total_count": 12500,
            "num_samples": 48,
            "timestamps_ns": [0, 520000, 1020000, ...],
            "total_duration_ns": 5100000
          },
          ...  # more runs
        ]
        
        OUTPUT STRUCTURE (averaged result):
        {
          "event_name": "cache-misses",
          "mode": "sampling",
          "sampling_period": 10000,
          "num_runs": 5,
          "timestamps_ns": [0, 510000, 1010000, ...],  # averaged timestamps
          "num_samples": 50,  # based on median-length run
          "stats": {
            "total_count_mean": 12400.0,
            "total_count_std": 150.5,
            "duration_mean_ns": 5050000.0,
            "duration_std_ns": 50000.0,
            "num_samples_mean": 49.2,
            "num_samples_std": 1.5
          }
        }
        
        Args:
            run_data: List of temporal feature dictionaries from each run
            
        Returns:
            Averaged temporal features
        """
        # Find run with median number of samples
        sample_counts = [r['num_samples'] for r in run_data]
        median_idx = np.argsort(sample_counts)[len(sample_counts)//2]
        template = run_data[median_idx]
        
        averaged = {
            'event_name': template['event_name'],
            'mode': template['mode'],
            'sampling_period': template['sampling_period'],
            'num_runs': len(run_data)
        }
        
        target_len = template['num_samples']
        
        # Average each sample position across runs
        timestamps_avg = []
        
        for i in range(target_len):
            # Collect i-th sample from each run (if it exists)
            ts_vals = [r['timestamps_ns'][i] for r in run_data if i < len(r['timestamps_ns'])]
            
            if ts_vals:
                timestamps_avg.append(int(np.mean(ts_vals)))
        
        averaged['timestamps_ns'] = timestamps_avg
        averaged['num_samples'] = len(timestamps_avg)
        
        # Statistics across runs
        averaged['stats'] = {
            'total_count_mean': float(np.mean([r['total_count'] for r in run_data])),
            'total_count_std': float(np.std([r['total_count'] for r in run_data])),
            'duration_mean_ns': float(np.mean([r['total_duration_ns'] for r in run_data])),
            'duration_std_ns': float(np.std([r['total_duration_ns'] for r in run_data])),
            'num_samples_mean': float(np.mean(sample_counts)),
            'num_samples_std': float(np.std(sample_counts))
        }
        
        return averaged
    
    def add_to_feature_db(self, function_label: str, event_index: int, temporal_data: dict):
        """
        Add temporal data to the feature database.
        
        FEATURE DATABASE STRUCTURE (self.feature_db):
        {
          "RSA_encrypt": {
            "event_0": {  # cache-misses
              "event_name": "cache-misses",
              "timestamps_ns": [...],
              "num_samples": 50,
              "stats": {...}
            },
            "event_1": {  # instructions
              "event_name": "instructions",
              "timestamps_ns": [...],
              ...
            }
          },
          "RSA_decrypt": {
            "event_0": {...},
            ...
          }
        }
        
        This structure will be saved to pmc_features.json
        """
        if function_label not in self.feature_db:
            self.feature_db[function_label] = {}
        
        event_key = f"event_{event_index}"
        self.feature_db[function_label][event_key] = temporal_data
        
        print(f"\n✓ Added to feature database:")
        print(f"  Function: {function_label}")
        print(f"  Event: {temporal_data['event_name']}")
        print(f"  Samples: {temporal_data['num_samples']}")
        print(f"  Duration: {temporal_data['stats']['duration_mean_ns']/1e6:.2f}ms "
              f"(±{temporal_data['stats']['duration_std_ns']/1e6:.2f}ms)")
        print(f"  Count: {temporal_data['stats']['total_count_mean']:.0f} "
              f"(±{temporal_data['stats']['total_count_std']:.0f})")
    
    def save_features(self):
        """Save feature database to JSON file."""
        with open(self.output_json, 'w') as f:
            json.dump(self.feature_db, f, indent=2)
        
        # Print summary
        total_functions = len(self.feature_db)
        total_events = sum(len(events) for events in self.feature_db.values())
        
        print(f"\n{'='*60}")
        print(f"✓ Saved features to {self.output_json}")
        print(f"  Functions/Workloads: {total_functions}")
        print(f"  Total Events: {total_events}")
        print(f"{'='*60}")
    
    def load_features(self) -> dict:
        """Load existing feature database from JSON file."""
        if os.path.exists(self.output_json):
            with open(self.output_json) as f:
                self.feature_db = json.load(f)
            
            total_functions = len(self.feature_db)
            total_events = sum(len(events) for events in self.feature_db.values())
            
            print(f"✓ Loaded existing features from {self.output_json}")
            print(f"  Functions: {total_functions}, Events: {total_events}")
        
        return self.feature_db


def main():
    """
    MAIN WORKFLOW:
    1. Parse command-line arguments (--target, --runs, --total)
    2. Read event indices from pmc_events.csv
    3. For each iteration (1 to --total):
       a. Create output file: pmc_features_N.json
       b. Group events into batches (batch_size events per batch)
       c. For each batch:
          - Run target program --runs times
          - Each run measures all events in the batch simultaneously
          - Extract temporal features from each run
          - Average features across runs
          - Store in memory (no file I/O yet)
       d. After all batches complete, save once to disk
    4. Output: pmc_features_1.json, pmc_features_2.json, ..., pmc_features_N.json
    
    OPTIMIZATION: One save per iteration (not per batch)
    - All data held in memory during batch collection (~1MB)
    - Single write at end = 5-10x faster than checkpoint approach
    - If script fails, only current iteration is lost
    
    FINAL OUTPUT FILE STRUCTURE (each pmc_features_N.json):
    {
      "RSA_encrypt": {
        "event_0": {temporal_data with averaged timestamps and stats},
        "event_1": {temporal_data with averaged timestamps and stats},
        ...
      },
      "RSA_decrypt": {
        "event_0": {...},
        ...
      }
    }
    """
    parser = argparse.ArgumentParser(
        description='Collect PMC temporal features for ML classification (measures 1 event at a time)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # Collect all events from pmc_events.csv (1 event at a time)
  # Automatically extracts ALL workloads from the target program
  # Deletes old results and builds complete pmc_features.json
  python3 collect_pmc_features.py --target ./example_cache_call --runs 5
  
  # Generate multiple feature files (pmc_features_1.json, pmc_features_2.json, etc.)
  python3 collect_pmc_features.py --target ./rsa_test --runs 5 --total 2
        """
    )
    
    parser.add_argument('--target', required=True, help='Path to target binary')
    parser.add_argument('--runs', type=int, default=5, help='Number of runs per event (default: 5)')
    parser.add_argument('--total', type=int, default=1, help='Number of complete feature collection iterations (default: 1)')
    
    args = parser.parse_args()
    
    # Hardcoded defaults
    csv_file = 'pmc_events.csv'
    batch_size = 4 # collect 4 event at a time
    
    # Loop for multiple iterations if --total is specified
    all_iterations_success = True
    
    for iteration in range(1, args.total + 1):
  
        output_file = f'pmc_features_{iteration}.json'
        
        print(f"\n{'*'*60}")
        print(f"{'*'*60}")
        print(f"ITERATION {iteration}/{args.total}")
        print(f"Output file: {output_file}")
        print(f"{'*'*60}")
        print(f"{'*'*60}\n")
        
        # Create collector
        try:
            collector = PMCTemporalCollector(
                target_binary=args.target,
                csv_file=csv_file,
                output_json=output_file
            )
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return 1
        
        # Read event indices from CSV
        try:
            event_indices = collector.read_event_indices_from_csv()
        except RuntimeError as e:
            print(f"Error: {e}")
            return 1
        
        if not event_indices:
            print(f"Error: No events found in {csv_file}")
            return 1
        
        print(f"Found {len(event_indices)} events in {csv_file}: {event_indices}")
        
        # Delete old output file to start fresh
        if os.path.exists(output_file):
            os.remove(output_file)
            print(f"✓ Deleted old {output_file}")
        
        # Group events into batches (pairs)
        # With batch_size=3, events [0,1,2,3,4,5] become [[0,1,2], [3,4,5]]
        # Each batch will be measured simultaneously in the same run
        event_batches = []
        for i in range(0, len(event_indices), batch_size):
            batch = event_indices[i:i+batch_size]
            event_batches.append(batch)
        
        print(f"Will collect {len(event_batches)} batches ({batch_size} events at a time)")
        print(f"Strategy: Hold all data in memory, save once at end of iteration\n")
        
        # Collect data for each batch
        success_count = 0
        total_batches = len(event_batches)
        
        for batch_idx, event_batch in enumerate(event_batches):
            print(f"\n{'#'*60}")
            print(f"Batch {batch_idx+1}/{total_batches}: Events {event_batch}")
            print(f"{'#'*60}")
            
            num_successful = collector.collect_events(
                event_indices=event_batch,
                num_runs=args.runs
            )
            
            if num_successful > 0:
                success_count += 1
                print(f"  ✓ Batch {batch_idx+1}/{total_batches} complete (holding in memory)")
        
        # Save all collected data once at the end of iteration
        if success_count > 0:
            print(f"\n{'='*60}")
            print(f"💾 Saving all {success_count} batches to {output_file}...")
            collector.save_features()
            print(f"{'='*60}")
        else:
            print(f"\n⚠️  No successful batches to save")

        
        # Iteration summary
        print(f"\n{'='*60}")
        print(f"Iteration {iteration} Complete!")
        print(f"  Total events: {len(event_indices)}")
        print(f"  Successful batches: {success_count}/{total_batches}")
        print(f"  Output: {output_file}")
        print(f"{'='*60}\n")
        
        if success_count == 0:
            all_iterations_success = False
    
    # Cleanup: remove pmc_results.json
    pmc_output = 'pmc_results.json'
    if os.path.exists(pmc_output):
        os.remove(pmc_output)
        print(f"✓ Cleaned up {pmc_output}\n")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"ALL ITERATIONS COMPLETE!")
    print(f"  Total iterations: {args.total}")
    if args.total == 1:
        print(f"  Output: pmc_features.json")
    else:
        print(f"  Output files: pmc_features_1.json to pmc_features_{args.total}.json")
    print(f"{'='*60}\n")
    
    return 0 if all_iterations_success else 1


if __name__ == "__main__":
    sys.exit(main())

