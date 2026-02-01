#!/usr/bin/env python3
"""
XGBoost Inference Script for PMC Function Classification

This script loads a trained XGBoost model and runs inference on JSON feature files
from the inference_temp/ directory.

Usage:
    python3 inference_xgboost.py
    python3 inference_xgboost.py --model ../models/xgboost_model.json
    python3 inference_xgboost.py --features pmc_features_wolfssl_1.json
    python3 inference_xgboost.py --output results_all.json
"""

import json
import numpy as np
import argparse
import os
import sys
from typing import List
import pickle
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix


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


def extract_features_from_json(json_file: str) -> tuple:
    """
    Extract features from a JSON file.
    
    Returns:
        (features_array, workload_names): features shape [N, 608], list of workload names
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    samples = []
    workload_names = []
    
    for workload_label, events in data.items():
        # Process into statistical features
        event_features = []
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
            
            # Compute 16 features per event
            stats = compute_statistical_features(
                timestamps, sampling_period,
                total_count_mean, total_count_std,
                duration_mean_ns, duration_std_ns,
                num_samples_mean, num_samples_std
            )
            event_features.extend(stats)  # Flatten: 38 events * 16 features = 608 features
        
        # Ensure exactly 38 events (pad if needed)
        while len(event_features) < 608:  # 38 * 16
            event_features.extend(np.zeros(16, dtype=np.float32))
        
        samples.append(np.array(event_features[:608], dtype=np.float32))
        workload_names.append(workload_label)
    
    return np.array(samples, dtype=np.float32), workload_names


def main():
    parser = argparse.ArgumentParser(description='XGBoost inference for PMC classification')
    parser.add_argument('--model', default='../models/xgboost_model.json',
                        help='Path to trained XGBoost model (default: ../models/xgboost_model.json)')
    parser.add_argument('--metadata', default='../models/xgboost_metadata.pkl',
                        help='Path to model metadata (default: ../models/xgboost_metadata.pkl)')
    parser.add_argument('--features', default='traces/*.json',
                        help='JSON feature file(s) to process. Can be a single file or glob pattern (default: traces/*.json)')
    parser.add_argument('--output', help='Optional: Save results to JSON file')
    
    args = parser.parse_args()
    
    # Resolve paths relative to script directory
    script_dir = Path(__file__).parent
    model_path = script_dir / args.model if not os.path.isabs(args.model) else Path(args.model)
    metadata_path = script_dir / args.metadata if not os.path.isabs(args.metadata) else Path(args.metadata)
    
    # Handle glob patterns for features
    import glob as glob_module
    if '*' in args.features or '?' in args.features:
        # It's a glob pattern
        features_pattern = str(script_dir / args.features) if not os.path.isabs(args.features) else args.features
        feature_files = sorted(glob_module.glob(features_pattern))
    else:
        # Single file
        features_path = script_dir / args.features if not os.path.isabs(args.features) else Path(args.features)
        feature_files = [str(features_path)]
    
    if not feature_files:
        print(f"Error: No feature files found matching: {args.features}")
        return 1
    
    print(f"{'='*80}")
    print("XGBoost Inference for PMC Classification")
    print(f"{'='*80}\n")
    print(f"Found {len(feature_files)} file(s) to process:")
    for f in feature_files:
        print(f"  - {Path(f).name}")
    print()
    
    # Check if files exist
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        print(f"Please train a model first using: python3 train_xgboost_gpu.py --save-model")
        return 1
    
    if not metadata_path.exists():
        print(f"Error: Metadata file not found: {metadata_path}")
        print(f"Please train a model first using: python3 train_xgboost_gpu.py --save-model")
        return 1
    
    # Load model
    print(f"Loading model from: {model_path.name}")
    bst = xgb.Booster()
    bst.load_model(str(model_path))
    print("✓ Model loaded successfully\n")
    
    # Load metadata
    print(f"Loading metadata from: {metadata_path.name}")
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    label_encoder = metadata['label_encoder']
    mean = metadata['mean']
    std = metadata['std']
    print(f"✓ Metadata loaded:")
    print(f"  Classes: {len(label_encoder.classes_)}")
    print(f"  Normalization: mean/std from training data")
    if 'test_accuracy' in metadata:
        print(f"  Test accuracy: {metadata['test_accuracy']:.4f}")
    print()
    
    # Process each file
    all_results = {}
    
    for file_idx, features_path in enumerate(feature_files, 1):
        features_path = Path(features_path)
        file_name = features_path.name
        
        print(f"\n{'='*80}")
        print(f"Processing File {file_idx}/{len(feature_files)}: {file_name}")
        print(f"{'='*80}\n")
        
        # Extract features from JSON
        print(f"Extracting features...")
        X, workload_names = extract_features_from_json(str(features_path))
        print(f"✓ Extracted {len(X)} samples with {X.shape[1]} features\n")
        
        # Normalize features (using training mean/std)
        print("Normalizing features...")
        X_normalized = (X - mean) / (std + 1e-8)
        print("✓ Normalization complete\n")
        
        # Create DMatrix for XGBoost
        dtest = xgb.DMatrix(X_normalized)
        
        # Run inference
        print(f"Running inference...")
        
        # Get class predictions (multi:softmax outputs class indices directly)
        y_pred = bst.predict(dtest)
        
        # Get probabilities
        try:
            # Try to get probabilities by using output_margin and applying softmax
            raw_scores = bst.predict(dtest, output_margin=True)
            # Apply softmax to get probabilities
            exp_scores = np.exp(raw_scores - np.max(raw_scores, axis=1, keepdims=True))
            y_pred_proba = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        except:
            # If that fails, create uniform probabilities
            print("  Warning: Could not extract probabilities, using uniform distribution")
            y_pred_proba = np.ones((len(y_pred), len(label_encoder.classes_))) / len(label_encoder.classes_)
        
        # Convert predictions to class labels
        predicted_labels = label_encoder.inverse_transform(y_pred.astype(int))
        print(f"✓ Inference complete\n")
        
        # Calculate accuracy
        accuracy = None
        balanced_acc = None
        y_true_encoded = None
        
        # Check which labels are unknown
        known_labels = set(label_encoder.classes_)
        unknown_labels = [label for label in workload_names if label not in known_labels]
        
        # Print predictions for samples with unknown ground truth
        if unknown_labels:
            print(f"{'='*80}")
            print("Predictions for Unseen Functions")
            print(f"{'='*80}\n")
            
            # Group by ground truth label
            unknown_predictions = {}
            for i, (gt_label, pred_label, probs) in enumerate(zip(workload_names, predicted_labels, y_pred_proba)):
                if gt_label not in known_labels:
                    if gt_label not in unknown_predictions:
                        unknown_predictions[gt_label] = []
                    unknown_predictions[gt_label].append({
                        'sample_idx': i,
                        'predicted': pred_label,
                        'confidence': float(probs.max()),
                        'probabilities': probs
                    })
            
            # Print predictions grouped by unseen ground truth
            for gt_label in sorted(unknown_predictions.keys()):
                predictions = unknown_predictions[gt_label]
                print(f"Ground Truth (Unseen): {gt_label}")
                print(f"  Number of samples: {len(predictions)}")
                
                # Count predictions
                pred_counts = {}
                for pred in predictions:
                    pred_label = pred['predicted']
                    if pred_label not in pred_counts:
                        pred_counts[pred_label] = 0
                    pred_counts[pred_label] += 1
                
                # Show prediction distribution
                print(f"  Predicted as:")
                for pred_label, count in sorted(pred_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / len(predictions)) * 100
                    print(f"    - {pred_label}: {count}/{len(predictions)} ({percentage:.1f}%)")
                
                # Show top-3 probabilities for first sample
                first_pred = predictions[0]
                print(f"  Example (Sample #{first_pred['sample_idx']}):")
                print(f"    Predicted: {first_pred['predicted']}")
                print(f"    Confidence: {first_pred['confidence']:.4f}")
                print(f"    Top-3 classes:")
                
                # Get top 3 classes
                top3_indices = np.argsort(first_pred['probabilities'])[-3:][::-1]
                for rank, idx in enumerate(top3_indices, 1):
                    class_name = label_encoder.classes_[idx]
                    prob = first_pred['probabilities'][idx]
                    bar_length = int(prob * 30)
                    bar = '█' * bar_length + '░' * (30 - bar_length)
                    print(f"      {rank}. {class_name:<30s} {prob:.4f} |{bar}|")
                
                print()
            
            print(f"{'='*80}\n")
        
        # Print predictions for samples with known ground truth
        known_samples = [i for i, label in enumerate(workload_names) if label in known_labels]
        if known_samples:
            print(f"{'='*80}")
            print("Predictions for Known Functions")
            print(f"{'='*80}\n")
            
            # Group by ground truth label
            known_predictions = {}
            for i in known_samples:
                gt_label = workload_names[i]
                pred_label = predicted_labels[i]
                probs = y_pred_proba[i]
                
                if gt_label not in known_predictions:
                    known_predictions[gt_label] = []
                known_predictions[gt_label].append({
                    'sample_idx': i,
                    'predicted': pred_label,
                    'confidence': float(probs.max()),
                    'probabilities': probs,
                    'correct': gt_label == pred_label
                })
            
            # Print predictions grouped by known ground truth
            for gt_label in sorted(known_predictions.keys()):
                predictions = known_predictions[gt_label]
                correct_count = sum(1 for p in predictions if p['correct'])
                accuracy_pct = (correct_count / len(predictions)) * 100
                
                print(f"Ground Truth (Known): {gt_label}")
                print(f"  Number of samples: {len(predictions)}")
                print(f"  Correctly predicted: {correct_count}/{len(predictions)} ({accuracy_pct:.1f}%)")
                
                # Count predictions
                pred_counts = {}
                for pred in predictions:
                    pred_label = pred['predicted']
                    if pred_label not in pred_counts:
                        pred_counts[pred_label] = 0
                    pred_counts[pred_label] += 1
                
                # Show prediction distribution
                print(f"  Predicted as:")
                for pred_label, count in sorted(pred_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / len(predictions)) * 100
                    marker = "✓" if pred_label == gt_label else "✗"
                    print(f"    {marker} {pred_label}: {count}/{len(predictions)} ({percentage:.1f}%)")
                
                # Show average confidence and example
                avg_confidence = np.mean([p['confidence'] for p in predictions])
                print(f"  Average confidence: {avg_confidence:.4f}")
                
                # Show example with highest confidence
                max_conf_pred = max(predictions, key=lambda x: x['confidence'])
                print(f"  Example with highest confidence (Sample #{max_conf_pred['sample_idx']}):")
                print(f"    Predicted: {max_conf_pred['predicted']} {'✓ CORRECT' if max_conf_pred['correct'] else '✗ WRONG'}")
                print(f"    Confidence: {max_conf_pred['confidence']:.4f}")
                print(f"    Top-3 classes:")
                
                # Get top 3 classes
                top3_indices = np.argsort(max_conf_pred['probabilities'])[-3:][::-1]
                for rank, idx in enumerate(top3_indices, 1):
                    class_name = label_encoder.classes_[idx]
                    prob = max_conf_pred['probabilities'][idx]
                    bar_length = int(prob * 30)
                    bar = '█' * bar_length + '░' * (30 - bar_length)
                    marker = "✓" if class_name == gt_label else " "
                    print(f"      {rank}. {marker} {class_name:<28s} {prob:.4f} |{bar}|")
                
                print()
            
            print(f"{'='*80}\n")
        
        if unknown_labels:
            print(f"⚠️  Warning: {len(unknown_labels)} function(s) not in training data:")
            for label in sorted(set(unknown_labels)):
                count = workload_names.count(label)
                print(f"    - {label} ({count} samples)")
            print(f"\n  These functions were likely filtered out during training (< min_samples threshold)")
            print(f"  or were not present in the training dataset.\n")
        
        try:
            y_true_encoded = label_encoder.transform(workload_names)
            accuracy = accuracy_score(y_true_encoded, y_pred)
            balanced_acc = balanced_accuracy_score(y_true_encoded, y_pred)
            
            # Print accuracy metrics
            print(f"{'='*80}")
            print("Results")
            print(f"{'='*80}")
            print(f"Overall Accuracy: {accuracy*100:.2f}%")
            print(f"Balanced Accuracy: {balanced_acc*100:.2f}%")
            print(f"Total Samples: {len(workload_names)}")
            print(f"Correct Predictions: {int(accuracy * len(workload_names))}")
            print(f"Wrong Predictions: {len(workload_names) - int(accuracy * len(workload_names))}")
            print(f"{'='*80}\n")
            
            # Detailed class info
            unique_gt = np.unique(y_true_encoded)
            unique_pred = np.unique(y_pred.astype(int))
            print(f"Unique ground truth classes: {len(unique_gt)}")
            print(f"Unique predicted classes: {len(unique_pred)}")
            
            # Store results for this file
            all_results[file_name] = {
                'accuracy': float(accuracy),
                'balanced_accuracy': float(balanced_acc),
                'total_samples': len(workload_names),
                'correct': int(accuracy * len(workload_names)),
                'wrong': len(workload_names) - int(accuracy * len(workload_names)),
                'unique_classes_gt': len(unique_gt),
                'unique_classes_pred': len(unique_pred),
                'unknown_functions': len(unknown_labels)
            }
            
        except ValueError as e:
            # Some labels in inference data might not be in training set
            print(f"{'='*80}")
            print("Results")
            print(f"{'='*80}")
            print(f"⚠️  ERROR: Cannot calculate accuracy")
            print(f"  Reason: {e}")
            print(f"  Total samples: {len(workload_names)}")
            print(f"  Unknown functions: {len(unknown_labels)}/{len(set(workload_names))}")
            print(f"{'='*80}\n")
            
            all_results[file_name] = {
                'error': str(e),
                'total_samples': len(workload_names),
                'known_samples': len(workload_names) - len([l for l in workload_names if l in unknown_labels]),
                'unknown_samples': len([l for l in workload_names if l in unknown_labels]),
                'unknown_functions': unknown_labels
            }
    
    # Print summary
    print(f"\n{'='*80}")
    print("Summary Across All Files")
    print(f"{'='*80}\n")
    print(f"{'File':<40} {'Accuracy':<12} {'Balanced':<12} {'Samples':<10}")
    print(f"{'-'*80}")
    
    for file_name, results in all_results.items():
        if 'accuracy' in results:
            acc_str = f"{results['accuracy']*100:.2f}%"
            bal_str = f"{results['balanced_accuracy']*100:.2f}%"
            samples_str = f"{results['total_samples']}"
            print(f"{file_name:<40} {acc_str:<12} {bal_str:<12} {samples_str:<10}")
        else:
            print(f"{file_name:<40} {'ERROR':<12} {'N/A':<12} {results['total_samples']:<10}")
    
    print(f"{'-'*80}")
    
    # Overall statistics
    if all_results:
        valid_results = [r for r in all_results.values() if 'accuracy' in r]
        if valid_results:
            avg_acc = np.mean([r['accuracy'] for r in valid_results])
            avg_bal = np.mean([r['balanced_accuracy'] for r in valid_results])
            total_samples = sum(r['total_samples'] for r in valid_results)
            total_correct = sum(r['correct'] for r in valid_results)
            print(f"{'AVERAGE':<40} {avg_acc*100:.2f}%      {avg_bal*100:.2f}%      {total_samples}")
            print(f"{'TOTAL ACCURACY':<40} {(total_correct/total_samples)*100:.2f}%")
    
    print(f"\n{'='*80}\n")
    
    # Save results if requested
    if args.output:
        output_path = script_dir / args.output if not os.path.isabs(args.output) else Path(args.output)
        
        output_data = {
            'summary': {
                'total_files': len(feature_files),
                'files_processed': len(all_results),
                'model_path': str(model_path),
                'metadata_path': str(metadata_path)
            },
            'per_file_results': all_results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Results saved to: {output_path}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
