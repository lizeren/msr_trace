#!/usr/bin/env python3
"""
Extract a subset of PMC events from JSON feature files based on an events CSV.

For every pmc_features_*.json in the source folder, keep only the events whose
event_name matches a name listed in the CSV, re-index them sequentially as
event_0, event_1, ..., and write the filtered file to the output folder.

All other JSON content (timestamps_ns, stats, num_runs, etc.) is preserved
verbatim. Source files are never modified.

Usage:
    python3 extract_events.py \\
        --input  2024-5991-static-40events_mix \\
        --events-csv pmc_events_deterministic.csv \\
        --output 2024-5991-static-10events

    # --output is optional; if omitted the name is auto-derived by replacing
    # the NN in "NNevents" inside the source folder name with the CSV row count.
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path


def load_event_names(csv_path: str) -> list[str]:
    """Return the ordered list of event_name values from the CSV."""
    names = []
    with open(csv_path, newline='') as fh:
        reader = csv.DictReader(fh)
        if 'event_name' not in (reader.fieldnames or []):
            sys.exit(f"ERROR: CSV '{csv_path}' has no 'event_name' column "
                     f"(found: {reader.fieldnames})")
        for row in reader:
            name = row['event_name'].strip()
            if name:
                names.append(name)
    if not names:
        sys.exit(f"ERROR: No event names found in '{csv_path}'")
    return names


def derive_output_folder(input_folder: str, n_events: int) -> str:
    """Replace the first NNevents token in the folder name with <n_events>events."""
    base = os.path.basename(input_folder.rstrip('/'))
    parent = os.path.dirname(os.path.abspath(input_folder))
    new_base = re.sub(r'\d+events', f'{n_events}events', base, count=1)
    if new_base == base:
        new_base = f"{base}-{n_events}events"
    return os.path.join(parent, new_base)


def extract_file(src_path: str, dst_path: str, allowed: set[str]) -> tuple[int, int]:
    """
    Filter one JSON file.

    Returns (total_events_in, total_events_kept) across all workloads.
    """
    try:
        with open(src_path, 'r') as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"  WARNING: skipping corrupted file: {src_path} ({exc})")
        return 0, 0

    total_in = 0
    total_kept = 0
    out_data = {}

    for workload_label, events in data.items():
        sorted_keys = sorted(events.keys(), key=lambda k: int(k.split('_')[1]))
        new_events: dict = {}
        new_idx = 0
        for key in sorted_keys:
            total_in += 1
            event_name = events[key].get('event_name', '')
            if event_name in allowed:
                new_events[f'event_{new_idx}'] = events[key]
                new_idx += 1
                total_kept += 1
        out_data[workload_label] = new_events

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, 'w') as fh:
        json.dump(out_data, fh, indent=2)

    return total_in, total_kept


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Extract a subset of PMC events from JSON feature files')
    parser.add_argument('--input', required=True,
                        help='Source folder containing pmc_features_*.json files')
    parser.add_argument('--events-csv', required=True,
                        help='CSV file with an event_name column listing events to keep')
    parser.add_argument('--output', default=None,
                        help='Destination folder (auto-derived from --input if omitted)')
    args = parser.parse_args()

    input_folder = args.input.rstrip('/')
    if not os.path.isdir(input_folder):
        sys.exit(f"ERROR: input folder not found: '{input_folder}'")

    event_names = load_event_names(args.events_csv)
    allowed = set(event_names)
    n_events = len(event_names)

    output_folder = args.output or derive_output_folder(input_folder, n_events)

    src_files = sorted(glob.glob(os.path.join(input_folder, 'pmc_features_*.json')))
    if not src_files:
        sys.exit(f"ERROR: no pmc_features_*.json files found in '{input_folder}'")

    print(f"Input  : {input_folder}  ({len(src_files)} files)")
    print(f"CSV    : {args.events_csv}  ({n_events} events)")
    print(f"Output : {output_folder}")
    print(f"Keeping events: {', '.join(event_names)}")
    print()

    os.makedirs(output_folder, exist_ok=True)

    total_files = 0
    total_in = 0
    total_kept = 0

    for src_path in src_files:
        filename = os.path.basename(src_path)
        dst_path = os.path.join(output_folder, filename)
        f_in, f_kept = extract_file(src_path, dst_path, allowed)
        total_in += f_in
        total_kept += f_kept
        total_files += 1
        if total_files % 500 == 0:
            print(f"  Progress: {total_files}/{len(src_files)} files...")

    print(f"Done.")
    print(f"  Files processed : {total_files}")
    print(f"  Events in       : {total_in}")
    print(f"  Events kept     : {total_kept}")
    if total_in > 0:
        dropped = total_in - total_kept
        print(f"  Events dropped  : {dropped} "
              f"({dropped / total_in * 100:.1f}% of input)")
    print(f"  Output folder   : {output_folder}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
