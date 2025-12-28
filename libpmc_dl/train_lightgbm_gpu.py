#!/usr/bin/env python3
"""
LightGBM GPU Training for PMC Function Classification

Single file with everything needed:
- Data loading from JSON features or cache
- Statistical feature computation
- LightGBM training with GPU acceleration
- Class imbalance handling
- Comprehensive evaluation with balanced accuracy

LightGBM is often faster and more accurate than XGBoost!
"""

import json
import numpy as np
import glob
import argparse
import os
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, confusion_matrix
import lightgbm as lgb


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


def load_data(features_pattern: str) -> Tuple[np.ndarray, List[str]]:
    """Load data and compute statistical features."""
    files = sorted(glob.glob(features_pattern))
    samples = []
    labels = []
    
    print(f"Loading {len(files)} files and computing statistical features...")
    
    for file_idx, file_path in enumerate(files):
        if file_idx % 200 == 0:
            print(f"  Progress: {file_idx}/{len(files)} files...")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  Warning: Skipping corrupted file: {file_path}")
            continue
        
        for workload_label, events in data.items():
            # Process into statistical features
            event_features = []
            event_keys = sorted(events.keys(), key=lambda x: int(x.split('_')[1]))
            
            for event_key in event_keys:
                event_data = events[event_key]
                timestamps = event_data.get('timestamps_ns', [])
                sampling_period = event_data.get('sampling_period', 100)
                
                # Get all stats
                stats_dict = event_data.get('stats', {})
                total_count_mean = stats_dict.get('total_count_mean', 0.0)
                total_count_std = stats_dict.get('total_count_std', 0.0)
                duration_mean_ns = stats_dict.get('duration_mean_ns', 0.0)
                duration_std_ns = stats_dict.get('duration_std_ns', 0.0)
                num_samples_mean = stats_dict.get('num_samples_mean', 0.0)
                num_samples_std = stats_dict.get('num_samples_std', 0.0)
                
                # Compute stats (16 features)
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
            labels.append(workload_label)
    
    print(f"Loaded {len(samples)} samples with {len(samples[0])} features each")
    return np.array(samples), labels


def main():
    parser = argparse.ArgumentParser(description='LightGBM GPU training for PMC classification')
    parser.add_argument('--features', default='features/pmc_features_*.json',
                        help='Path to feature files')
    parser.add_argument('--n-estimators', type=int, default=2000,
                        help='Number of boosting rounds')
    parser.add_argument('--num-leaves', type=int, default=63,
                        help='Maximum number of leaves (2^depth - 1)')
    parser.add_argument('--learning-rate', type=float, default=0.05,
                        help='Learning rate')
    parser.add_argument('--feature-fraction', type=float, default=0.8,
                        help='Feature sampling ratio')
    parser.add_argument('--bagging-fraction', type=float, default=0.8,
                        help='Data sampling ratio')
    parser.add_argument('--bagging-freq', type=int, default=5,
                        help='Bagging frequency')
    parser.add_argument('--min-child-samples', type=int, default=20,
                        help='Minimum samples per leaf')
    parser.add_argument('--reg-alpha', type=float, default=0.1,
                        help='L1 regularization')
    parser.add_argument('--reg-lambda', type=float, default=1.0,
                        help='L2 regularization')
    parser.add_argument('--scale-pos-weight', action='store_true',
                        help='Auto-scale class weights for imbalanced data')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for training')
    parser.add_argument('--save-model', action='store_true',
                        help='Save trained model')
    parser.add_argument('--early-stopping', type=int, default=50,
                        help='Early stopping rounds')
    parser.add_argument('--cache', action='store_true',
                        help='Use pre-computed cached features from features_16/')
    
    args = parser.parse_args()
    
    # Load data
    if args.cache:
        print("\n📦 Loading cached features...")
        cache_file = 'features_16/features_cache.pkl'
        if not os.path.exists(cache_file):
            print(f"❌ Cache file not found: {cache_file}")
            print(f"   Run: python3 preprocess_features.py")
            return 1
        
        import pickle
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        X = cache_data['X'].reshape(len(cache_data['X']), -1)  # Flatten [N, 38, 10] -> [N, 380]
        y = cache_data['y']
        print(f"✓ Loaded {len(X)} samples from cache (shape: {X.shape})")
    else:
        X, y = load_data(args.features)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"\nDataset info:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: {len(label_encoder.classes_)}")
    print(f"  Class names: {label_encoder.classes_[:5]}... (showing first 5)")
    
    # Check for class imbalance
    unique, counts = np.unique(y_encoded, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = max_count / min_count
    
    print(f"\nClass distribution:")
    print(f"  Max samples per class: {max_count}")
    print(f"  Min samples per class: {min_count}")
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}x")
    
    if imbalance_ratio > 3:
        print(f"  ⚠️  Significant class imbalance detected!")
        if not args.scale_pos_weight:
            print(f"  💡 Consider using --scale-pos-weight flag")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")
    print(f"  Test:  {len(X_test)} samples")
    
    # Normalize features
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8
    
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # LightGBM parameters
    params = {
        'objective': 'multiclass',
        'num_class': len(label_encoder.classes_),
        'metric': ['multi_logloss', 'multi_error'],
        'boosting_type': 'gbdt',
        'num_leaves': args.num_leaves,
        'learning_rate': args.learning_rate,
        'feature_fraction': args.feature_fraction,
        'bagging_fraction': args.bagging_fraction,
        'bagging_freq': args.bagging_freq,
        'min_child_samples': args.min_child_samples,
        'lambda_l1': args.reg_alpha,
        'lambda_l2': args.reg_lambda,
        'verbose': -1,
        'seed': 42,
        'force_col_wise': True,  # Faster for many features
    }
    
    # GPU support
    if args.gpu:
        params['device'] = 'gpu'
        params['gpu_platform_id'] = 0
        params['gpu_device_id'] = 0
        print("\n🚀 GPU acceleration enabled (device=gpu)")
    else:
        params['device'] = 'cpu'
        print("\n💻 Using CPU (device=cpu)")
    
    # Class weighting
    if args.scale_pos_weight:
        # For multiclass, we use sample weights
        class_counts = np.bincount(y_train)
        class_weights = len(y_train) / (len(class_counts) * class_counts)
        sample_weights = class_weights[y_train]
        train_data.set_weight(sample_weights)
        print("✓ Class weighting enabled (balanced sample weights)")
    
    print(f"\nLightGBM parameters:")
    for key, val in params.items():
        print(f"  {key}: {val}")
    
    # Train
    print(f"\n{'='*80}")
    print("Training LightGBM...")
    print(f"{'='*80}\n")
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=args.early_stopping),
        lgb.log_evaluation(period=10)
    ]
    
    bst = lgb.train(
        params,
        train_data,
        num_boost_round=args.n_estimators,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=callbacks
    )
    
    print(f"\n✓ Training completed!")
    print(f"Best iteration: {bst.best_iteration}")
    
    # Evaluate on test set
    print(f"\n{'='*80}")
    print("Test Set Evaluation")
    print(f"{'='*80}\n")
    
    y_pred_proba = bst.predict(X_test, num_iteration=bst.best_iteration)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Metrics
    test_acc = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Balanced Accuracy: {balanced_acc*100:.2f}%")
    
    gap = abs(test_acc - balanced_acc) * 100
    if gap > 10:
        print(f"\n⚠️  Large gap ({gap:.1f}%) between accuracy and balanced accuracy!")
        print(f"    This indicates class imbalance issues.")
        if not args.scale_pos_weight:
            print(f"    Try using --scale-pos-weight flag")
    elif gap > 5:
        print(f"\n⚠️  Moderate gap ({gap:.1f}%) between accuracy and balanced accuracy.")
    else:
        print(f"\n✓ Small gap ({gap:.1f}%) - good balance across classes!")
    
    print(f"\n{classification_report(y_test, y_pred, target_names=label_encoder.classes_, digits=4)}")
    
    # Feature importance
    print(f"\n{'='*80}")
    print("Top 20 Most Important Features")
    print(f"{'='*80}\n")
    
    importance = bst.feature_importance(importance_type='gain')
    importance_indices = np.argsort(importance)[-20:][::-1]
    
    for idx, feat_idx in enumerate(importance_indices, 1):
        event_num = feat_idx // 10
        feat_num = feat_idx % 10
        feat_names = ['total_dur', 'mean_int', 'std_int', 'min_int', 'max_int',
                      'samp_rate', 'n_samp', 'q25', 'q50', 'q75']
        print(f"  {idx:2d}. f{feat_idx}: {importance[feat_idx]:8.2f}  (event_{event_num:02d} {feat_names[feat_num]})")
    
    # Confusion matrix analysis
    conf_mat = confusion_matrix(y_test, y_pred)
    
    # Find most confused pairs
    print(f"\n{'='*80}")
    print("Top 5 Most Confused Function Pairs")
    print(f"{'='*80}\n")
    
    confused_pairs = []
    for i in range(len(conf_mat)):
        for j in range(len(conf_mat)):
            if i != j and conf_mat[i, j] > 0:
                confused_pairs.append((i, j, conf_mat[i, j]))
    
    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    
    for idx, (true_idx, pred_idx, count) in enumerate(confused_pairs[:5], 1):
        true_label = label_encoder.classes_[true_idx]
        pred_label = label_encoder.classes_[pred_idx]
        print(f"  {idx}. {true_label} → {pred_label}: {count} misclassifications")
    
    # Save model
    if args.save_model:
        os.makedirs('models', exist_ok=True)
        model_path = 'models/lightgbm_model.txt'
        bst.save_model(model_path)
        
        # Save additional info
        import pickle
        metadata = {
            'label_encoder': label_encoder,
            'mean': mean,
            'std': std,
            'params': params,
            'best_iteration': bst.best_iteration,
            'test_accuracy': test_acc,
            'balanced_accuracy': balanced_acc
        }
        with open('models/lightgbm_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"\n✓ Model saved to: {model_path}")
        print(f"✓ Metadata saved to: models/lightgbm_metadata.pkl")
    
    # Summary
    print(f"\n{'='*80}")
    print("Training Summary")
    print(f"{'='*80}")
    print(f"  Model: LightGBM")
    print(f"  Device: {'GPU' if args.gpu else 'CPU'}")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]} (38 events × 10 statistical features)")
    print(f"  Classes: {len(label_encoder.classes_)}")
    print(f"  Trees: {bst.best_iteration}")
    print(f"  Test Accuracy: {test_acc*100:.2f}%")
    print(f"  Balanced Accuracy: {balanced_acc*100:.2f}%")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

