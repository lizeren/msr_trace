#!/usr/bin/env python3
"""
PMC Logistic Regression Classifier

Trains a logistic regression model on PMC temporal features with proper normalization.
"""

import json
import numpy as np
from pathlib import Path
import glob
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import argparse


class PMCFeatureExtractor:
    """Extract ML features from PMC temporal data."""
    
    def __init__(self, features_json: str):
        self.features_json = features_json

        # Single scaler for all features (will normalize each column independently)
        # Each of the 370 columns gets its own mean and std
        self.scaler = StandardScaler()
        self.num_features_per_event = 10  # total_duration, mean_interval, etc.

        # LabelEncoder converts string class labels (e.g., "workloadA", "workloadB") into integer IDs:
        # This is required because scikit-learn models expect integer class labels.
        self.label_encoder = LabelEncoder()
        self.num_events = None  # Will be set when building feature matrix
        
    def load_features(self):
        """Load PMC features from JSON file(s)."""
        # If the path contains * or ?, it assumes you want to load many files:
        if '*' in self.features_json or '?' in self.features_json:
            # Glob pattern - load multiple files
            return self.load_multiple_files(self.features_json)
        else:
            # Single file
            with open(self.features_json, 'r') as f:
                return json.load(f)
    
    def load_multiple_files(self, pattern):
        """
        Load multiple JSON files matching a pattern.
        Supports patterns like:
          - pmc_features_*.json (numbered: 1, 2, 3, ...)
          - pmc_features_http_*.json (descriptive + numbered)
          - features/pmc_features_rsa_*.json (with path)
        
        Returns a list of (filename, pmc_data) tuples.
        """
        # Return a list of paths matching a pathname pattern.
        files = sorted(glob.glob(pattern))
        
        if not files:
            raise FileNotFoundError(f"No files found matching pattern: {pattern}")
        
        all_data = []
        print(f"  Found {len(files)} matching files:")
        print(f"  Pattern: {pattern}")
        
        for file_path in files:
            # print(f"    - {Path(file_path).name}")
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    all_data.append((file_path, data))
            except Exception as e:
                print(f"      Warning: Failed to load {file_path}: {e}")
        
        if len(all_data) == 0:
            raise ValueError(f"No files successfully loaded from pattern: {pattern}")
        
        print(f"  Successfully loaded: {len(all_data)}/{len(files)} files\n")
        
        return all_data
    
    def extract_temporal_stats(self, timestamps_ns, num_samples):
        """
        Extract statistical features from temporal sequence.
        
        Args:
            timestamps_ns: List of timestamps (normalized to t=0)
            num_samples: Number of samples
            
        Returns:
            List of derived features
        """
        if len(timestamps_ns) == 0:
            return [0] * self.num_features_per_event
        
        timestamps = np.array(timestamps_ns)
        
        # Time-based features
        total_duration = timestamps[-1] if len(timestamps) > 0 else 0
        mean_interval = np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 0
        std_interval = np.std(np.diff(timestamps)) if len(timestamps) > 1 else 0
        min_interval = np.min(np.diff(timestamps)) if len(timestamps) > 1 else 0
        max_interval = np.max(np.diff(timestamps)) if len(timestamps) > 1 else 0
        
        # Rate features
        sample_rate = num_samples / (total_duration + 1)  # samples per nanosecond
        
        # Quartile features
        q25 = np.percentile(timestamps, 25) if len(timestamps) > 0 else 0
        q50 = np.percentile(timestamps, 50) if len(timestamps) > 0 else 0
        q75 = np.percentile(timestamps, 75) if len(timestamps) > 0 else 0
        
        # Full feature list (10 features)
        all_features = [
            total_duration,
            mean_interval,
            std_interval,
            min_interval,
            max_interval,
            sample_rate,
            num_samples,
            q25,
            q50,
            q75
        ]
        
        # Return only the first num_features_per_event features
        return all_features[:self.num_features_per_event]
    
    def build_feature_matrix(self, pmc_data):
        """
        Build feature matrix from PMC data.
        Uses ALL events across all workloads (union), filling in zeros
        for workloads that don't have certain events.
        
        Args:
            pmc_data: Either a dict (single file) or list of (filename, dict) tuples (multiple files)
            pmc_data = (pmc_feature1_1.json , {workload_label: {event_key: {event_name: event_name, timestamps_ns: [timestamp1, timestamp2, ...], num_samples: num_samples}}})
        
        Returns:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            feature_names: List of feature names
            workload_names: List of workload names
            all_event_keys_sorted: Sorted list of all event keys (for inference)
        """
        # Detect if we have multiple files or single file
        is_multiple_files = isinstance(pmc_data, list)
        
        # First pass: find ALL events across all workloads (union)
        all_event_keys = set()
        
        if is_multiple_files:
            # Multiple files: iterate through each file's workloads
            for filename, data in pmc_data:
                for workload_label, events in data.items():
                    # events = {event_key: {event_name: event_name, timestamps_ns: [timestamp1, timestamp2, ...], num_samples: num_samples}}
                    all_event_keys = all_event_keys.union(set(events.keys()))
        else:
            # Single file: iterate through workloads
            for workload_label, events in pmc_data.items():
                all_event_keys = all_event_keys.union(set(events.keys()))
        
        if len(all_event_keys) == 0:
            raise ValueError("No events found in any workload")
        
        # Sort event keys for consistent ordering
        all_event_keys_sorted = sorted(all_event_keys, key=lambda x: int(x.split('_')[1]))
        
        print(f"  Total unique events across all workloads: {len(all_event_keys_sorted)}")
        print(f"  Event keys: {all_event_keys_sorted[:5]}{'...' if len(all_event_keys_sorted) > 5 else ''}")
        
        # Build feature names from all events
        feature_names = []
        event_name_map = {}  # Map event_key to event_name for consistent naming
        
        for event_key in all_event_keys_sorted:
            # Find event name from any workload that has this event
            event_name = event_key  # Default to event_key
            
            if is_multiple_files:
                # Search through all files
                for filename, data in pmc_data:
                    for workload_label, events in data.items():
                        if event_key in events:
                            event_name = events[event_key].get('event_name', event_key)
                            break
                    if event_name != event_key:
                        break
            else:
                # Search through single file
                for workload_label, events in pmc_data.items():
                    if event_key in events:
                        event_name = events[event_key].get('event_name', event_key) # Get the event name, otherwise use the event key
                        break
            
            event_name_map[event_key] = event_name
            
            # Full feature name list (10 features)
            all_feature_names = [
                f"{event_name}_total_duration",
                f"{event_name}_mean_interval",
                f"{event_name}_std_interval",
                f"{event_name}_min_interval",
                f"{event_name}_max_interval",
                f"{event_name}_sample_rate",
                f"{event_name}_num_samples",
                f"{event_name}_q25",
                f"{event_name}_q50",
                f"{event_name}_q75"
            ]
            
            # Add only the first num_features_per_event feature names
            feature_names.extend(all_feature_names[:self.num_features_per_event])
        
        # Second pass: extract features for each workload using all events
        X = []
        y = []
        workload_names = []
        
        if is_multiple_files:
            print(f"\n  Processing {len(pmc_data)} files...")
            # Multiple files: each workload in each file becomes a separate sample
            for file_idx, (filename, data) in enumerate(pmc_data, 1):
                # print(f"\n  File {file_idx}/{len(pmc_data)}: {Path(filename).name}")
                for workload_label, events in data.items():
                    workload_features = []
                    num_missing = 0
                    
                    # Extract features from all events (fill zeros if event doesn't exist)
                    for event_key in all_event_keys_sorted:
                        if event_key in events:
                            # Event exists, extract temporal statistics
                            event = events[event_key]
                            temporal_features = self.extract_temporal_stats(
                                event.get('timestamps_ns', []),
                                event.get('num_samples', 0)
                            )
                        else:
                            # Event doesn't exist for this workload, fill with zeros
                            temporal_features = [0] * self.num_features_per_event
                            num_missing += 1
                        
                        workload_features.extend(temporal_features) 
                        #one long feature vector (a single list) for a workload that contains:
                        #all 10 temporal features for event_1
                        #all 10 temporal features for event_2
                        #...
                        #all 10 temporal features for event_37
                    
                    X.append(workload_features)
                    y.append(workload_label)
                    workload_names.append(f"{workload_label}_run{file_idx}")
                    
                    # print(f"    {workload_label}: {len(events)}/{len(all_event_keys_sorted)} events " + f"({num_missing} filled with zeros)")
                    """
                    X = [
                        [workload_0_sample0_event_0_feat_0, workload_0_sample0_event_0_feat_1, ..., workload_0_sample0_event_0_feat_9, workload_0_sample0_event_1_feat_0, ..., workload_0_sample0_event_1_feat_9, ...],
                        [workload_1_sample0_event_0_feat_0, workload_1_sample0_event_0_feat_1, ..., workload_1_sample0_event_0_feat_9, workload_1_sample0_event_1_feat_0, ..., workload_1_sample0_event_1_feat_9, ...],
                        [workload_2_sample0_event_0_feat_0, workload_2_sample0_event_0_feat_1, ..., workload_2_sample0_event_0_feat_9, workload_2_sample0_event_1_feat_0, ..., workload_2_sample0_event_1_feat_9, ...],
                        ...
                        [workload_N_sample0_event_0_feat_0, workload_N_sample0_event_0_feat_1, ..., workload_N_sample0_event_0_feat_9, workload_N_sample0_event_1_feat_0, ..., workload_N_sample0_event_1_feat_9, ...],
                        [workload_0_sample1_event_0_feat_0, workload_0_sample1_event_0_feat_1, ..., workload_0_sample1_event_0_feat_9, workload_0_sample1_event_1_feat_0, ..., workload_0_sample1_event_1_feat_9, ...],
                        [workload_1_sample1_event_0_feat_0, workload_1_sample1_event_0_feat_1, ..., workload_1_sample1_event_0_feat_9, workload_1_sample1_event_1_feat_0, ..., workload_1_sample1_event_1_feat_9, ...],
                        [workload_2_sample1_event_0_feat_0, workload_2_sample1_event_0_feat_1, ..., workload_2_sample1_event_0_feat_9, workload_2_sample1_event_1_feat_0, ..., workload_2_sample1_event_1_feat_9, ...],
                        ...
                        [workload_N_sample1_event_0_feat_0, workload_N_sample1_event_0_feat_1, ..., workload_N_sample1_event_0_feat_9, workload_N_sample1_event_1_feat_0, ..., workload_N_sample1_event_1_feat_9, ...],
                        ...
                        [workload_0_sampleM_event_0_feat_0, workload_0_sampleM_event_0_feat_1, ..., workload_0_sampleM_event_0_feat_9, workload_0_sampleM_event_1_feat_0, ..., workload_0_sampleM_event_1_feat_9, ...],
                    ]
                    """
        else:
            print(f"\n  Event coverage per workload:")
            # Single file: each workload is one sample
            for workload_label, events in pmc_data.items():
                workload_features = []
                num_missing = 0
                
                # Extract features from all events (fill zeros if event doesn't exist)
                for event_key in all_event_keys_sorted:
                    if event_key in events:
                        # Event exists, extract temporal statistics
                        event = events[event_key]
                        temporal_features = self.extract_temporal_stats(
                            event.get('timestamps_ns', []),
                            event.get('num_samples', 0)
                        )
                    else:
                        # Event doesn't exist for this workload, fill with zeros
                        temporal_features = [0] * self.num_features_per_event
                        num_missing += 1
                    
                    workload_features.extend(temporal_features)
                
                X.append(workload_features)
                y.append(workload_label)
                workload_names.append(workload_label)
                
                print(f"    {workload_label}: {len(events)}/{len(all_event_keys_sorted)} events " +
                      f"({num_missing} filled with zeros)")
        
        # Store number of events for normalization
        self.num_events = len(all_event_keys_sorted)
        
        return np.array(X), np.array(y), feature_names, workload_names, all_event_keys_sorted
    
    def normalize_features(self, X_train, X_test=None):
        """
        Normalize each column independently across samples.
        Each (event, feature_type) combination gets its own mean and std.
        
        This is correct because:
        - Each event has a different sampling period
        - event_0_duration and event_1_duration are on different scales
        - Each of the 370 columns should be normalized independently
        
        Feature structure: [event_0_feat_0, ..., event_0_feat_9, event_1_feat_0, ..., event_N_feat_9]
        
        Normalization:
        - Column 0 (event_0_duration): normalized across all samples
        - Column 1 (event_0_mean_interval): normalized across all samples
        - ...
        - Column 369 (event_36_q75): normalized across all samples
        
        Args:
            X_train: Training feature matrix (n_samples, n_features)
            X_test: Optional test feature matrix
            
        Returns:
            X_train_scaled, X_test_scaled (or just X_train_scaled if X_test is None)
        """
        if self.num_events is None:
            raise ValueError("num_events not set. Call build_feature_matrix first.")
        
        print(f"\n  Per-column normalization:")
        print(f"    Number of events: {self.num_events}")
        print(f"    Features per event: {self.num_features_per_event}")
        print(f"    Total features: {X_train.shape[1]}")
        print(f"    Each of {X_train.shape[1]} columns normalized independently")
        print(f"    (Each column gets its own mean and std computed across {X_train.shape[0]} samples)")
        
        # StandardScaler normalizes each column independently by default
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled


class PMCClassifier:
    """Logistic Regression classifier for PMC features."""
    
    def __init__(self, max_iter=1000, random_state=42):
        self.model = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            multi_class='ovr',  # One-vs-Rest for multi-class
            solver='lbfgs'
        )
        self.feature_extractor = None
        
    def train(self, X_train, y_train):
        """Train the classifier."""
        print(f"\nTraining Logistic Regression...")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Classes: {len(np.unique(y_train))}")
        
        self.model.fit(X_train, y_train)
        
        # Training accuracy
        train_acc = self.model.score(X_train, y_train)
        print(f"  Training accuracy: {train_acc:.4f}")
        
    def evaluate(self, X_test, y_test, label_names=None):
        """Evaluate the classifier."""
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{'='*60}")
        print(f"Test Set Evaluation")
        print(f"{'='*60}")
        print(f"Accuracy: {accuracy:.4f}")
        # print(f"\nConfusion Matrix:")
        # print(confusion_matrix(y_test, y_pred))
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=label_names))
        
        return accuracy, y_pred
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation."""
        print(f"\n{'='*60}")
        print(f"Cross-Validation (k={cv})")
        print(f"{'='*60}")
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        print(f"Cross-validation scores: {scores}")
        print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores
    
    def save_model(self, output_dir='models'):
        """Save trained model and preprocessing objects."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Include all_event_keys and scaler for inference
        model_data = {
            'model': self.model,
            'scaler': self.feature_extractor.scaler,  # Single scaler (normalizes each column independently)
            'num_events': self.feature_extractor.num_events,
            'num_features_per_event': self.feature_extractor.num_features_per_event,
            'label_encoder': self.feature_extractor.label_encoder,
            'feature_names': getattr(self, 'feature_names', None),
            'all_event_keys': getattr(self, 'all_event_keys', None)  # CRITICAL for inference
        }
        
        model_path = os.path.join(output_dir, 'pmc_classifier.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n{'='*60}")
        print(f"Model saved to: {model_path}")
        print(f"  - Trained model (LogisticRegression)")
        print(f"  - Feature scaler (StandardScaler, per-column normalization)")
        print(f"  - Label encoder ({len(self.feature_extractor.label_encoder.classes_)} classes)")
        print(f"  - Feature names ({len(model_data['feature_names']) if model_data['feature_names'] else 0} features)")
        print(f"  - Event keys ({len(model_data['all_event_keys']) if model_data['all_event_keys'] else 0} events)")
        print(f"{'='*60}")
        
        return model_path


def main():
    parser = argparse.ArgumentParser(
        description='Train logistic regression classifier on PMC features'
    )
    parser.add_argument(
        '--features',
        default='pmc_features.json',
        help='Path to PMC features JSON file or glob pattern. Examples:\n'
             '  Single: "pmc_features.json"\n'
             '  Multiple numbered: "pmc_features_*.json"\n'
             '  Descriptive naming: "pmc_features_http_*.json" or "features/pmc_features_rsa_*.json"'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set size (default: 0.2)'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    parser.add_argument(
        '--no-split',
        action='store_true',
        help='Skip train/test split, only do cross-validation'
    )
    parser.add_argument(
        '--save-model',
        action='store_true',
        help='Save trained model to disk for inference'
    )
    parser.add_argument(
        '--model-dir',
        default='models',
        help='Directory to save model (default: models)'
    )
    parser.add_argument(
        '--cache',
        action='store_true',
        help='Load from cached features (libpmc_dl/features_16/features_cache.pkl)'
    )
    
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print(f"PMC Logistic Regression Classifier")
    print(f"{'='*60}")
    
    # Initialize feature extractor
    extractor = PMCFeatureExtractor(args.features)
    
    # Load features (from cache or JSON)
    if args.cache:
        print(f"Loading from cache...")
        cache_file = '../libpmc_dl/features_16/features_cache.pkl'
        
        import os
        if not os.path.exists(cache_file):
            print(f"❌ Cache file not found: {cache_file}")
            print(f"   Run: cd ../libpmc_dl && python3 preprocess_features.py")
            return 1
        
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Cache contains [N, 38, num_features]
        X_raw = cache_data['X']  # Shape: [N, 38, num_features]
        y = cache_data['y']
        num_features = cache_data['num_features']
        
        # Flatten to [N, 38*num_features] for logistic regression
        X = X_raw.reshape(len(X_raw), -1)
        
        print(f"✓ Loaded {len(X)} samples from cache")
        print(f"  Shape: {X_raw.shape} -> {X.shape}")
        print(f"  Features per event: {num_features}")
        print(f"  Total features: {X.shape[1]}")
        
        # Create feature names from cache
        feature_names = []
        for event_idx in range(38):
            for feat_name in cache_data['feature_names']:
                feature_names.append(f"event_{event_idx}_{feat_name}")
        
        workload_names = y.tolist()
        all_event_keys_sorted = [f"event_{i}" for i in range(38)]
        
        # Update extractor settings
        extractor.num_features_per_event = num_features
        extractor.num_events = 38
        
    else:
        print(f"Features file: {args.features}")
        print(f"\nLoading features...")
        pmc_data = extractor.load_features()
        if isinstance(pmc_data, list):
            # Multiple files
            num_files = len(pmc_data)
            # Count unique workloads across all files
            unique_workloads = set()
            for _, data in pmc_data:
                unique_workloads.update(data.keys())
            print(f"  Loaded {num_files} files with {len(unique_workloads)} unique workloads")
        else:
            # Single file
            print(f"  Found {len(pmc_data)} workloads in single file")
        
        print(f"\nExtracting feature matrix...")
        X, y, feature_names, workload_names, all_event_keys_sorted = extractor.build_feature_matrix(pmc_data)
        print(f"\n{'='*60}")
        print(f"  Feature matrix shape: {X.shape}")
        print(f"  Total features: {len(feature_names)}")
        print(f"  Total samples: {len(X)}")
        print(f"  Unique classes: {len(np.unique(y))}")
    
    # Show samples per class (only if not from cache, to avoid clutter)
    if not args.cache:
        print(f"\n  Samples per class:")
        unique_labels, counts = np.unique(y, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"    {label}: {count} samples")
    else:
        print(f"\n  Unique classes: {len(np.unique(y))}")
    print(f"{'='*60}")
    
    # Check if we have enough samples
    if len(X) < 2:
        print(f"\nError: Need at least 2 workload samples for classification")
        print(f"Currently have {len(X)} sample(s)")
        return 1
    
    # Encode labels
    y_encoded = extractor.label_encoder.fit_transform(y) # Convert string labels to integer IDs
    label_names = extractor.label_encoder.classes_ # fit_transform does sorting. so label_names is the sorted list of workload labels
    
    print(f"\n{'='*60}")
    print(f"Feature Normalization (StandardScaler)")
    print(f"{'='*60}")
    
    # Initialize classifier
    classifier = PMCClassifier()
    classifier.feature_extractor = extractor
    
    if args.no_split or len(X) < 5:
        # Dataset too small for proper train/test split
        X_scaled = extractor.normalize_features(X)
        
        # Check if we have enough samples for cross-validation
        min_class_size = min([np.sum(y_encoded == cls) for cls in np.unique(y_encoded)])
        
        if min_class_size < 2:
            print(f"Dataset too small for cross-validation (only {min_class_size} sample per class)")
            print(f"Training on full dataset for demonstration purposes only...")
            
            # Just fit the model to show it works (no evaluation)
            classifier.train(X_scaled, y_encoded)
            
            print(f"\n{'='*60}")
            print(f"Note: With only {len(X)} samples, model evaluation is not meaningful.")
            print(f"Collect more workload samples for proper evaluation.")
            print(f"{'='*60}")
        else:
            print(f"Using cross-validation only (dataset too small for split)")
            cv_folds = min(args.cv_folds, min_class_size)
            classifier.cross_validate(X_scaled, y_encoded, cv=cv_folds)
    else:
        # Train/test split
        # X has all the features for all the workloads (one long feature vector for each workload)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=args.test_size, 
            random_state=42,
            stratify=y_encoded if len(np.unique(y_encoded)) > 1 else None
        )
        
        print(f"Train/test split: {len(X_train)}/{len(X_test)}")
        
        # Normalize features

        # X_train contains e.g. 10 runs for 14 workloads. And we take 80 percent of them 
        # X_train[0:9] contains 10 runs for the first workload.
        # X_train[0][0:9] contains 10 features
        print("X_train.shape: ", X_train.shape)
        print("len(X_train): ", len(X_train))
        print("y_train shape: ", len(y_train))
        print("X_train[0] shape: ", len(X_train[0]))
        
        X_train_scaled, X_test_scaled = extractor.normalize_features(X_train, X_test)
        
        print(f"Features normalized to zero mean, unit variance")
        # axis=0 is per column
        print(f"  Mean (before): {X_train.mean(axis=0)[:3]} ...")
        print(f"  Mean (after):  {X_train_scaled.mean(axis=0)[:3]} ...")
        print(f"  Std (before):  {X_train.std(axis=0)[:3]} ...")
        print(f"  Std (after):   {X_train_scaled.std(axis=0)[:3]} ...")
        # Train classifier
        classifier.train(X_train_scaled, y_train)
        
        # Evaluate on test set
        classifier.evaluate(X_test_scaled, y_test, label_names=label_names)
        
        # Cross-validation on full dataset (if we have enough samples)
        X_full_scaled = extractor.normalize_features(X)
        min_class_size = min([np.sum(y_encoded == cls) for cls in np.unique(y_encoded)])
        if min_class_size >= 2:
            cv_folds = min(args.cv_folds, min_class_size)
            classifier.cross_validate(X_full_scaled, y_encoded, cv=cv_folds)
        else:
            print(f"\n{'='*60}")
            print(f"Skipping cross-validation (need at least 2 samples per class)")
            print(f"{'='*60}")
    
    print(f"\n{'='*60}")
    print(f"Top 10 Most Important Features (by coefficient magnitude)")
    print(f"{'='*60}")
    
    # Show feature importance (coefficient magnitudes)
    if hasattr(classifier.model, 'coef_'):
        coef_magnitudes = np.abs(classifier.model.coef_).mean(axis=0)
        top_indices = np.argsort(coef_magnitudes)[-10:][::-1]
        
        for i, idx in enumerate(top_indices, 1):
            if idx < len(feature_names):
                print(f"{i:2d}. {feature_names[idx]:50s} {coef_magnitudes[idx]:.6f}")
    
    # Save model if requested
    if args.save_model:
        # Store feature names and event keys in classifier for inference
        classifier.feature_names = feature_names
        classifier.all_event_keys = all_event_keys_sorted  # Save the event keys used in training
        classifier.save_model(args.model_dir)
    
    print(f"\nDone!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

