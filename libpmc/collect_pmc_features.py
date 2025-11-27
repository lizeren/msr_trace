#!/usr/bin/env python3
"""
PMC Temporal Feature Collector

Collects performance counter data by running a target program multiple times
with different PMC events (1 at a time), extracting temporal features for ML classification.

Automatically reads events from pmc_events.csv and measures them individually.

Usage:
    python3 collect_pmc_features.py --target ./example_cache_call --runs 10
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
                 output_json: str = "temporal_features.json",
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
    
    def read_event_indices_from_csv(self) -> List[int]:
        """Read all event indices from CSV file."""
        event_indices = []
        try:
            with open(self.csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('index'):
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
    
    def collect_events(self, event_indices: List[int], num_runs: int = 10) -> int:
        """
        Collect temporal data for multiple events simultaneously across multiple runs.
        Extracts data for ALL workloads found in the target program output.
        
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
        
        run_data = []
        
        for run_id in range(num_runs):
            print(f"\n[Run {run_id+1}/{num_runs}]", end=" ")
            
            # Delete previous PMC output to avoid appending
            if os.path.exists(self.pmc_output):
                os.remove(self.pmc_output)
                print(f"Cleaned old {self.pmc_output}", end=" -> ")
            
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
            workload_data_dict = self.extract_all_temporal_features(pmc_data, target_label=None)
            
            if workload_data_dict:
                run_data.append(workload_data_dict)
                total_events = sum(len(events) for events in workload_data_dict.values())
                num_workloads = len(workload_data_dict)
                print(f"SUCCESS ({num_workloads} workloads, {total_events} events total)")
            else:
                print("FAILED (no temporal data)")
        
        # Average across runs and store for each workload and event
        if run_data:
            print(f"\n✓ Collected {len(run_data)}/{num_runs} successful runs")
            
            # Organize data by workload and event index
            # workloads_events = {workload_label: {event_idx: [event_data_per_run]}}
            workloads_events = {}
            
            for run in run_data:
                for workload_label, events_list in run.items():
                    if workload_label not in workloads_events:
                        workloads_events[workload_label] = {}
                    
                    for event_data in events_list:
                        event_idx = event_data['event_index']
                        if event_idx not in workloads_events[workload_label]:
                            workloads_events[workload_label][event_idx] = []
                        workloads_events[workload_label][event_idx].append(event_data)
            
            # Average and store each event for each workload
            for workload_label, events_by_index in workloads_events.items():
                for event_idx, event_runs in events_by_index.items():
                    averaged = self.average_temporal_runs(event_runs)
                    self.add_to_feature_db(workload_label, event_idx, averaged)
        else:
            print(f"\n✗ No successful runs for events {event_str}")
        
        return len(run_data)
    
    def extract_all_temporal_features(self, pmc_data: dict, target_label: str = None) -> Dict[str, List[dict]]:
        """
        Extract temporal information for ALL events from ALL workloads in PMC JSON output.
        Only processes sampling mode events.
        
        Args:
            pmc_data: Parsed JSON from PMC output file
            target_label: Function/workload label to match (None = extract all workloads)
            
        Returns:
            Dictionary mapping workload labels to lists of event data
        """
        # Build mapping from event name to CSV index
        event_name_map = self.build_event_name_to_index_map()
        
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
                
                # Only process sampling modes (skip counting mode)
                if mode not in ['sampling', 'sampling_freq']:
                    continue
                
                samples = event.get('samples', [])
                if not samples:
                    continue
                
                # Map event name to CSV index
                event_name = event['event_name']
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
                event_info['event_counts'] = [s['count'] for s in samples]
                event_info['cpus'] = [s['cpu'] for s in samples]
                
                # Derived features
                event_info['total_duration_ns'] = event_info['timestamps_ns'][-1]
                
                workload_data[workload_label].append(event_info)
        
        return workload_data
    
    def average_temporal_runs(self, run_data: List[dict]) -> dict:
        """
        Average temporal data across multiple runs.
        Uses the median-length run as template and aligns others.
        
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
        counts_avg = []
        
        for i in range(target_len):
            # Collect i-th sample from each run (if it exists)
            ts_vals = [r['timestamps_ns'][i] for r in run_data if i < len(r['timestamps_ns'])]
            count_vals = [r['event_counts'][i] for r in run_data if i < len(r['event_counts'])]
            
            if ts_vals:
                timestamps_avg.append(int(np.mean(ts_vals)))
                counts_avg.append(int(np.mean(count_vals)))
        
        averaged['timestamps_ns'] = timestamps_avg
        averaged['event_counts'] = counts_avg
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
        """Add temporal data to the feature database."""
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
    parser = argparse.ArgumentParser(
        description='Collect PMC temporal features for ML classification (measures 1 event at a time)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # Collect all events from pmc_events.csv (1 event at a time)
  # Automatically extracts ALL workloads from the target program
  # Deletes old results and builds complete temporal_features.json
  python3 collect_pmc_features.py --target ./example_cache_call --runs 10
        """
    )
    
    parser.add_argument('--target', required=True, help='Path to target binary')
    parser.add_argument('--runs', type=int, default=10, help='Number of runs per event (default: 10)')
    
    args = parser.parse_args()
    
    # Hardcoded defaults
    csv_file = 'pmc_events.csv'
    output_file = 'temporal_features.json'
    batch_size = 1
    
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
    event_batches = []
    for i in range(0, len(event_indices), batch_size):
        batch = event_indices[i:i+batch_size]
        event_batches.append(batch)
    
    print(f"Will collect {len(event_batches)} events (1 event at a time)")
    
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
            # Load existing features from file to merge with new batch
            if batch_idx > 0 and os.path.exists(output_file):
                existing_db = {}
                try:
                    with open(output_file) as f:
                        existing_db = json.load(f)
                    # Merge existing data into current feature_db
                    for workload, events in existing_db.items():
                        if workload not in collector.feature_db:
                            collector.feature_db[workload] = {}
                        collector.feature_db[workload].update(events)
                except Exception as e:
                    print(f"Warning: Could not load existing features: {e}")
            
            # Save after each batch (appends if data was loaded above)
            collector.save_features()
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"Collection Complete!")
    print(f"  Total events: {len(event_indices)}")
    print(f"  Successful batches: {success_count}/{total_batches}")
    print(f"  Output: {output_file}")
    print(f"{'='*60}\n")
    
    # Cleanup: remove pmc_results.json
    pmc_output = 'pmc_results.json'
    if os.path.exists(pmc_output):
        os.remove(pmc_output)
        print(f"✓ Cleaned up {pmc_output}\n")
    
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

