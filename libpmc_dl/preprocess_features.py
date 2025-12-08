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


def compute_statistical_features(timestamps: List[int], sampling_period: int) -> np.ndarray:
    """
    Compute 10 statistical features on timestamp intervals.
    
    Features:
    1. total_duration, 2. mean_interval, 3. std_interval, 4. min_interval, 5. max_interval
    6. sample_rate, 7. num_samples, 8. q25, 9. q50 (median), 10. q75
    """
    if len(timestamps) < 2:
        return np.zeros(10, dtype=np.float32)
    
    intervals = np.diff(timestamps)
    
    if len(intervals) == 0 or np.all(intervals == 0):
        return np.zeros(10, dtype=np.float32)
    
    features = [
        timestamps[-1] - timestamps[0],  # 0: total_duration
        np.mean(intervals),              # 1: mean_interval
        np.std(intervals),               # 2: std_interval
        np.min(intervals),               # 3: min_interval
        np.max(intervals),               # 4: max_interval
        len(timestamps) / (timestamps[-1] - timestamps[0] + 1),  # 5: sample_rate
        len(timestamps),                 # 6: num_samples
        np.percentile(intervals, 25),    # 7: q25
        np.percentile(intervals, 50),    # 8: q50 (median)
        np.percentile(intervals, 75),    # 9: q75
    ]
    
    return np.array(features, dtype=np.float32)


def preprocess_and_cache(features_pattern: str, output_dir: str = 'features_10'):
    """Load JSON features, compute statistics, and cache to disk."""
    
    files = sorted(glob.glob(features_pattern))
    
    if len(files) == 0:
        print(f"❌ No files found matching: {features_pattern}")
        return
    
    print(f"Found {len(files)} files")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    all_samples = []
    all_labels = []
    
    print(f"\n{'='*80}")
    print("Processing files...")
    print(f"{'='*80}\n")
    
    for file_idx, file_path in enumerate(files):
        if file_idx % 200 == 0:
            print(f"  Progress: {file_idx}/{len(files)} files...")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  ⚠️  Warning: Skipping corrupted file: {file_path}")
            continue
        
        for workload_label, events in data.items():
            # Process into statistical features
            event_features = []
            event_keys = sorted(events.keys(), key=lambda x: int(x.split('_')[1]))
            
            for event_key in event_keys:
                event_data = events[event_key]
                timestamps = event_data.get('timestamps_ns', [])
                sampling_period = event_data.get('sampling_period', 100)
                
                # Compute stats
                stats = compute_statistical_features(timestamps, sampling_period)
                event_features.append(stats)  # Keep as [38, 10] for dual-stream
            
            # Ensure exactly 38 events (pad if needed)
            while len(event_features) < 38:
                event_features.append(np.zeros(10, dtype=np.float32))
            
            # Store as [38, 10] array
            sample = np.array(event_features[:38], dtype=np.float32)
            all_samples.append(sample)
            all_labels.append(workload_label)
    
    print(f"\n  ✓ Processed {len(all_samples)} samples")
    
    # Convert to arrays
    X = np.array(all_samples, dtype=np.float32)  # Shape: [N, 38, 10]
    y = np.array(all_labels)
    
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
        'shape': X.shape,
        'num_samples': len(X),
        'num_events': 38,
        'num_features': 10,
        'feature_names': [
            'total_duration', 'mean_interval', 'std_interval', 'min_interval', 'max_interval',
            'sample_rate', 'num_samples', 'q25', 'q50', 'q75'
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
        f.write(f"Feature dimensions:\n")
        f.write(f"  Events per sample: 38\n")
        f.write(f"  Features per event: 10\n")
        f.write(f"  Total features: 380 (when flattened)\n\n")
        f.write(f"Feature names (per event):\n")
        for i, name in enumerate(cache_data['feature_names'], 1):
            f.write(f"  {i}. {name}\n")
    
    print(f"  ✓ Metadata saved to: {metadata_file}")
    
    print(f"\n{'='*80}")
    print("✅ Preprocessing complete!")
    print(f"{'='*80}\n")
    print(f"To use cached features in training:")
    print(f"  python3 train_xgboost_gpu.py --cache")
    print(f"  python3 train_dual_stream.py --cache")
    print(f"  python3 train_hybrid_cnn.py --cache")
    print()


def main():
    parser = argparse.ArgumentParser(description='Preprocess and cache statistical features')
    parser.add_argument('--features', default='features/pmc_features_*.json',
                        help='Path pattern to JSON feature files')
    parser.add_argument('--output', default='features_10',
                        help='Output directory for cached features')
    
    args = parser.parse_args()
    
    preprocess_and_cache(args.features, args.output)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

