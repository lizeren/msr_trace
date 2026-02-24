#!/usr/bin/env python3
"""
Pre-compute statistical features and cache them to disk.
This dramatically speeds up training by avoiding repeated JSON I/O and computation.

Usage:
    python3 preprocess_features.py
    python3 preprocess_features.py --features "features/pmc_features_*.json"
"""

import json
import numpy as np
import glob
import argparse
import os
from typing import List
import pickle
from pathlib import Path


FEATURES_PER_EVENT = 16  # Number of statistical features computed per PMC event


def compute_statistical_features(timestamps: List[int], sampling_period: int, 
                                total_count_mean: float = 0.0,
                                total_count_std: float = 0.0,
                                duration_mean_ns: float = 0.0,
                                duration_std_ns: float = 0.0,
                                num_samples_mean: float = 0.0,
                                num_samples_std: float = 0.0) -> np.ndarray:
    """
    Compute 16 statistical features: 6 from stats + 10 from timestamp intervals.
    
    Features from stats (1-6):
    1. total_count_mean, 2. total_count_std, 3. duration_mean_ns, 
    4. duration_std_ns, 5. num_samples_mean, 6. num_samples_std
    
    Features from timestamps (7-16):
    7. total_duration, 8. mean_interval, 9. std_interval, 10. min_interval, 11. max_interval
    12. sample_rate, 13. num_samples, 14. q25, 15. q50 (median), 16. q75
    """
    # Features 1-6: From stats (always available)
    features = [
        total_count_mean,
        total_count_std,
        duration_mean_ns,
        duration_std_ns,
        num_samples_mean,
        num_samples_std
    ]
    
    # Compute temporal features if we have enough timestamps
    if len(timestamps) < 2:
        # Not enough timestamps - pad with zeros for temporal features
        features.extend([0.0] * 10)
        return np.array(features, dtype=np.float32)
    
    intervals = np.diff(timestamps)
    
    if len(intervals) == 0 or np.all(intervals == 0):
        features.extend([0.0] * 10)
        return np.array(features, dtype=np.float32)
    
    # Temporal features (7-16)
    features.extend([
        timestamps[-1] - timestamps[0],  # 7: total_duration
        np.mean(intervals),              # 8: mean_interval
        np.std(intervals),               # 9: std_interval
        np.min(intervals),               # 10: min_interval
        np.max(intervals),               # 11: max_interval
        len(timestamps) / (timestamps[-1] - timestamps[0] + 1),  # 12: sample_rate
        len(timestamps),                 # 13: num_samples
        np.percentile(intervals, 25),    # 14: q25
        np.percentile(intervals, 50),    # 15: q50 (median)
        np.percentile(intervals, 75),    # 16: q75
    ])
    
    return np.array(features, dtype=np.float32)


def preprocess_and_cache(features_pattern: str, output_dir: str = 'features_16', 
                        min_samples: int = 20, custom_files: List[str] = None):
    """Load JSON features, compute statistics, and cache to disk.
    
    Args:
        features_pattern: Glob pattern for JSON feature files (ignored if custom_files provided)
        output_dir: Output directory for cached features
        min_samples: Minimum samples required per function (default: 20)
                    - Technical minimum: 4 (for stratified split)
                    - Statistical minimum: 20-30 (basic validity)
                    - Recommended: 50+ (reliable performance)
        custom_files: Custom list of files to process (overrides features_pattern)
    """
    
    if custom_files is not None:
        files = custom_files
        print(f"\nUsing {len(files)} pre-selected files")
    else:
        files = sorted(glob.glob(features_pattern))
        
        if len(files) == 0:
            print(f"❌ No files found matching: {features_pattern}")
            return
        
        print(f"Found {len(files)} files")
    
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    all_samples_raw = []  # Raw (unpadded) event feature lists; padding happens after all files are read
    all_labels = []
    all_groups = []  # Track which file each sample came from
    
    print(f"\n{'='*80}")
    print("Processing files...")
    print(f"{'='*80}\n")
    
    for file_idx, file_path in enumerate(files):
        if file_idx % 200 == 0:
            print(f"  Progress: {file_idx}/{len(files)} files...")
        
        # Use file basename as group identifier
        file_group = os.path.basename(file_path)
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  ⚠️  Warning: Skipping corrupted file: {file_path}")
            continue
        
        for workload_label, events in data.items():
            # Process into statistical features
            event_features = []

            # Sort events by event number
            # without it "event_0", "event_1", "event_10", "event_11", ..., "event_19"
            # with it "event_0", "event_1", "event_2", ..., "event_10", "event_11"
            event_keys = sorted(events.keys(), key=lambda x: int(x.split('_')[1]))
            
            for event_key in event_keys:
                event_data = events[event_key]
                timestamps = event_data.get('timestamps_ns', [])
                sampling_period = event_data.get('sampling_period', 100)
                
                # Get all stats (these are always available in the JSON)
                stats_dict = event_data.get('stats', {})
                total_count_mean = stats_dict.get('total_count_mean', 0.0)
                total_count_std = stats_dict.get('total_count_std', 0.0)
                duration_mean_ns = stats_dict.get('duration_mean_ns', 0.0)
                duration_std_ns = stats_dict.get('duration_std_ns', 0.0)
                num_samples_mean = stats_dict.get('num_samples_mean', 0.0)
                num_samples_std = stats_dict.get('num_samples_std', 0.0)
                
                # Compute stats (now 16 features: 6 from stats + 10 from timestamps)
                stats = compute_statistical_features(
                    timestamps, sampling_period,
                    total_count_mean, total_count_std,
                    duration_mean_ns, duration_std_ns,
                    num_samples_mean, num_samples_std
                )
                event_features.append(stats)

            all_samples_raw.append(event_features)
            all_labels.append(workload_label)
            all_groups.append(file_group)  # Track which file this sample came from
    
    # Determine max events dynamically across all collected samples
    max_events = max(len(ef) for ef in all_samples_raw) if all_samples_raw else 0

    # Pad every sample to max_events and convert to arrays
    all_samples = []
    for event_features in all_samples_raw:
        while len(event_features) < max_events:
            event_features.append(np.zeros(FEATURES_PER_EVENT, dtype=np.float32))
        all_samples.append(np.array(event_features[:max_events], dtype=np.float32))

    X = np.array(all_samples, dtype=np.float32)  # Shape: [N, max_events, FEATURES_PER_EVENT]
    y = np.array(all_labels)
    groups = np.array(all_groups)  # Group identifiers (file names)

    print(f"\n  ✓ Processed {len(all_samples)} samples from {len(np.unique(groups))} unique files")
    
    # Analyze class distribution
    print(f"\n{'='*80}")
    print("Class Distribution Analysis")
    print(f"{'='*80}\n")
    
    from collections import Counter
    label_counts = Counter(all_labels)
    
    # Sort by count (descending) then by name
    sorted_labels = sorted(label_counts.items(), key=lambda x: (-x[1], x[0]))
    
    print(f"Total unique functions: {len(label_counts)}")
    print(f"Total samples: {len(all_labels)}")
    print(f"Minimum samples threshold: {min_samples}")
    print(f"")
    
    # Identify functions with too few samples
    functions_too_few = [label for label, count in label_counts.items() if count < min_samples]
    functions_adequate = [label for label, count in label_counts.items() if count >= min_samples]
    
    if functions_too_few:
        total_samples_too_few = sum(label_counts[label] for label in functions_too_few)
        print(f"⚠️  WARNING: {len(functions_too_few)} functions have < {min_samples} samples")
        print(f"   These will be filtered out during training (losing {total_samples_too_few} samples)")
        print(f"   Functions with adequate samples: {len(functions_adequate)}")
        print(f"")
    
    # Display all function counts
    print(f"Sample count per function:")
    print(f"{'-'*80}")
    print(f"{'Function Name':<50} {'Samples':>10} {'Status':>10}")
    print(f"{'-'*80}")
    
    for label, count in sorted_labels:
        status = "OK" if count >= min_samples else "TOO FEW"
        marker = "⚠️ " if count < min_samples else "  "
        print(f"{marker}{label:<48} {count:>10} {status:>10}")
    
    print(f"{'-'*80}")
    print(f"{'TOTAL':<50} {len(all_labels):>10}")
    print(f"")
    
    # Statistics
    counts_list = list(label_counts.values())
    print(f"Distribution statistics:")
    print(f"  Max samples per function: {max(counts_list)}")
    print(f"  Min samples per function: {min(counts_list)}")
    print(f"  Mean samples per function: {np.mean(counts_list):.1f}")
    print(f"  Median samples per function: {np.median(counts_list):.1f}")
    print(f"  Imbalance ratio (max/min): {max(counts_list)/min(counts_list):.1f}x")
    print(f"")
    
    # Filter out functions with too few samples before caching
    if functions_too_few:
        print(f"\n{'='*80}")
        print(f"Filtering Data")
        print(f"{'='*80}\n")
        
        print(f"Removing {len(functions_too_few)} functions with < {min_samples} samples...")
        
        # Create a mask for samples to keep
        mask = np.array([label not in functions_too_few for label in all_labels])
        
        # Filter features, labels, AND groups
        X_filtered = X[mask]
        y_filtered = y[mask]
        groups_filtered = groups[mask]
        
        samples_removed = len(X) - len(X_filtered)
        print(f"  Samples before: {len(X)}")
        print(f"  Samples after: {len(X_filtered)}")
        print(f"  Removed: {samples_removed} samples ({samples_removed/len(X)*100:.2f}%)")
        print(f"  Files before: {len(np.unique(groups))}")
        print(f"  Files after: {len(np.unique(groups_filtered))}")
        
        # Update X, y, and groups
        X = X_filtered
        y = y_filtered
        groups = groups_filtered
        
        print(f"\n✓ Filtered data ready for training")
        print(f"  Functions: {len(functions_adequate)}")
        print(f"  Samples: {len(X)}")
        print(f"  Files: {len(np.unique(groups))}")
    else:
        print(f"\n✓ All functions meet minimum sample requirement")
    
    # Save to disk
    cache_file = os.path.join(output_dir, 'features_cache.pkl')
    
    print(f"\n{'='*80}")
    print("Saving cache...")
    print(f"{'='*80}\n")
    print(f"  Shape: {X.shape}")
    print(f"  Labels: {len(y)}")
    print(f"  File: {cache_file}")
    
    cache_data = {
        'X': X,
        'y': y,
        'groups': groups,  # Add group information for proper train/test splitting
        'shape': X.shape,
        'num_samples': len(X),
        'num_files': len(np.unique(groups)),
        'num_events': max_events,
        'num_features': FEATURES_PER_EVENT,
        'min_samples_threshold': min_samples,
        'functions_removed': functions_too_few if functions_too_few else [],
        'feature_names': [
            # From stats (1-6)
            'total_count_mean', 'total_count_std', 'duration_mean_ns', 
            'duration_std_ns', 'num_samples_mean', 'num_samples_std',
            # From timestamps (7-16)
            'total_duration', 'mean_interval', 'std_interval', 'min_interval', 
            'max_interval', 'sample_rate', 'num_samples', 'q25', 'q50', 'q75'
        ]
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_size_mb = os.path.getsize(cache_file) / (1024 * 1024)
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"\n  ✓ Cache saved successfully!")
    
    # Save metadata
    metadata_file = os.path.join(output_dir, 'metadata.txt')
    with open(metadata_file, 'w') as f:
        f.write(f"Feature Cache Metadata\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Source pattern: {features_pattern}\n")
        f.write(f"Total samples: {len(X)}\n")
        f.write(f"Shape: {X.shape}\n")
        f.write(f"Data type: {X.dtype}\n")
        f.write(f"Cache file: {cache_file}\n")
        f.write(f"File size: {file_size_mb:.2f} MB\n\n")
        f.write(f"Preprocessing:\n")
        f.write(f"  Min samples threshold: {min_samples}\n")
        if functions_too_few:
            f.write(f"  Functions filtered: {len(functions_too_few)}\n")
            f.write(f"  Functions in cache: {len(functions_adequate)}\n")
        else:
            f.write(f"  Functions filtered: 0\n")
            f.write(f"  Functions in cache: {len(label_counts)}\n")
        f.write(f"\n")
        f.write(f"Feature dimensions:\n")
        f.write(f"  Events per sample: {max_events}\n")
        f.write(f"  Features per event: {FEATURES_PER_EVENT} (6 from stats + 10 from timestamps)\n")
        f.write(f"  Total features: {max_events * FEATURES_PER_EVENT} (when flattened: {max_events}*{FEATURES_PER_EVENT})\n\n")
        f.write(f"Feature names (per event):\n")
        for i, name in enumerate(cache_data['feature_names'], 1):
            f.write(f"  {i}. {name}\n")
    
    print(f"  ✓ Metadata saved to: {metadata_file}")
    
    # Save class distribution report
    distribution_file = os.path.join(output_dir, 'class_distribution.txt')
    with open(distribution_file, 'w') as f:
        f.write(f"Class Distribution Report\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Total unique functions: {len(label_counts)}\n")
        f.write(f"Total samples: {len(all_labels)}\n")
        f.write(f"Min samples required for training: {min_samples}\n\n")
        
        if functions_too_few:
            total_samples_too_few = sum(label_counts[label] for label in functions_too_few)
            f.write(f"WARNING: {len(functions_too_few)} functions have < {min_samples} samples\n")
            f.write(f"These will be filtered out during training (losing {total_samples_too_few} samples)\n")
            f.write(f"Functions with adequate samples: {len(functions_adequate)}\n\n")
        
        f.write(f"\nSample count per function:\n")
        f.write(f"{'-'*80}\n")
        f.write(f"{'Function Name':<50} {'Samples':>10} {'Status':>10}\n")
        f.write(f"{'-'*80}\n")
        
        for label, count in sorted_labels:
            status = "OK" if count >= min_samples else "TOO FEW"
            marker = "[!] " if count < min_samples else "    "
            f.write(f"{marker}{label:<46} {count:>10} {status:>10}\n")
        
        f.write(f"{'-'*80}\n")
        f.write(f"{'TOTAL':<50} {len(all_labels):>10}\n\n")
        
        f.write(f"\nDistribution statistics:\n")
        f.write(f"  Max samples per function: {max(counts_list)}\n")
        f.write(f"  Min samples per function: {min(counts_list)}\n")
        f.write(f"  Mean samples per function: {np.mean(counts_list):.1f}\n")
        f.write(f"  Median samples per function: {np.median(counts_list):.1f}\n")
        f.write(f"  Imbalance ratio (max/min): {max(counts_list)/min(counts_list):.1f}x\n")
        
        if functions_too_few:
            f.write(f"\n\nFunctions with too few samples (< {min_samples}):\n")
            f.write(f"{'-'*60}\n")
            for label in sorted(functions_too_few):
                f.write(f"  {label}: {label_counts[label]} samples\n")
    
    print(f"  ✓ Class distribution saved to: {distribution_file}")
    
    print(f"\n{'='*80}")
    print("✅ Preprocessing complete!")
    print(f"{'='*80}\n")
    print(f"Summary:")
    print(f"  Cached samples: {len(X)}")
    print(f"  Functions: {len(functions_adequate) if functions_too_few else len(label_counts)}")
    print(f"  Filtered out: {len(functions_too_few)} functions")
    print(f"  Cache file: {cache_file}")
    print(f"")
    print(f"To use cached features in training:")
    print(f"  python3 train_xgboost_gpu.py --cache --scale-pos-weight --gpu")
    print(f"  python3 train_dual_stream.py --cache")
    print(f"  python3 train_hybrid_cnn.py --cache")
    print()


def main():
    parser = argparse.ArgumentParser(description='Preprocess and cache statistical features')
    parser.add_argument('--features', default='features/pmc_features_*.json',
                        help='Path pattern to JSON feature files')
    parser.add_argument('--output', default='features_16',
                        help='Output directory for cached features')
    parser.add_argument('--min-samples', type=int, default=20,
                        help='Minimum samples per function (default: 20). '
                             'Functions with fewer samples will be flagged. '
                             'Recommended: 4 (technical min), 20 (statistical min), 50+ (reliable)')
    parser.add_argument('--sample-per-pattern', type=str, default=None,
                        help='Sample N files per pattern. Format: "pattern1:N,pattern2:N" '
                             'Example: "rsa:500,wolfssl:500" takes 500 files containing "rsa" and 500 containing "wolfssl"')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"PMC Feature Preprocessing")
    print(f"{'='*80}")
    print(f"  Features pattern: {args.features}")
    print(f"  Output directory: {args.output}")
    print(f"  Min samples threshold: {args.min_samples}")
    
    # Handle pattern-based sampling
    files = sorted(glob.glob(args.features))
    
    if args.sample_per_pattern:
        print(f"  Sampling mode: Per-pattern")
        print(f"  Pattern spec: {args.sample_per_pattern}")
        print(f"{'='*80}\n")
        
        # Parse pattern specification: "rsa:500,wolfssl:500"
        pattern_specs = {}
        for spec in args.sample_per_pattern.split(','):
            pattern, count = spec.strip().split(':')
            pattern_specs[pattern.strip().lower()] = int(count)
        
        print(f"Pattern-based file sampling:")
        selected_files = []
        
        for pattern, max_count in pattern_specs.items():
            # Find files matching this pattern
            matching = [f for f in files if pattern in os.path.basename(f).lower()]
            sampled = matching[:max_count]
            selected_files.extend(sampled)
            
            print(f"  '{pattern}': Found {len(matching)} files, selected {len(sampled)}")
        
        files = selected_files
        print(f"\nTotal files to process: {len(files)}")
        
        if len(files) == 0:
            print(f"❌ No files matched the patterns!")
            return 1
    else:
        print(f"{'='*80}\n")
        print(f"Processing all {len(files)} files")
    
    preprocess_and_cache(args.features if not args.sample_per_pattern else None, 
                        args.output, args.min_samples, 
                        custom_files=files if args.sample_per_pattern else None)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

