#!/usr/bin/env python3
"""
PMC Classifier Inference

Load a trained classifier and run inference on unseen PMC features,
showing probability distributions (softmax scores) for each workload.
"""

import json
import numpy as np
import pickle
import argparse
from pathlib import Path


class PMCInference:
    """Inference engine for trained PMC classifier."""
    
    def __init__(self, model_path):
        """Load trained model and preprocessing objects."""
        print(f"Loading model from: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data.get('scaler', None)
        self.num_events = model_data.get('num_events', None)
        self.num_features_per_event = model_data.get('num_features_per_event', 10)
        
        if self.scaler is None:
            raise ValueError("Model does not contain scaler. Please retrain your model.")
        
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data.get('feature_names', None)
        self.all_event_keys = model_data.get('all_event_keys', None)  # Event keys from training
        
        print(f"  Model loaded successfully!")
        print(f"  Classes: {self.label_encoder.classes_}")
        print(f"  Features: {len(self.feature_names) if self.feature_names else 'Unknown'}")
        print(f"  Training event keys: {len(self.all_event_keys) if self.all_event_keys else 'Unknown'}")
        print(f"  Normalization: Per-column (each of {len(self.feature_names) if self.feature_names else 'N'} columns normalized independently)")
    
    def extract_temporal_stats(self, timestamps_ns, num_samples):
        """
        Extract statistical features from temporal sequence.
        Same as in training script.
        """
        if len(timestamps_ns) == 0:
            return [0] * 10
        
        timestamps = np.array(timestamps_ns)
        
        # Time-based features
        total_duration = timestamps[-1] if len(timestamps) > 0 else 0
        mean_interval = np.mean(np.diff(timestamps)) if len(timestamps) > 1 else 0
        std_interval = np.std(np.diff(timestamps)) if len(timestamps) > 1 else 0
        min_interval = np.min(np.diff(timestamps)) if len(timestamps) > 1 else 0
        max_interval = np.max(np.diff(timestamps)) if len(timestamps) > 1 else 0
        
        # Rate features
        sample_rate = num_samples / (total_duration + 1)
        
        # Quartile features
        q25 = np.percentile(timestamps, 25) if len(timestamps) > 0 else 0
        q50 = np.percentile(timestamps, 50) if len(timestamps) > 0 else 0
        q75 = np.percentile(timestamps, 75) if len(timestamps) > 0 else 0
        
        return [
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
    
    def extract_features_from_workload(self, workload_data, all_event_keys):
        """
        Extract feature vector from a single workload.
        
        Args:
            workload_data: Dictionary of events for one workload
            all_event_keys: List of all event keys (sorted)
        
        Returns:
            Feature vector (numpy array)
        """
        features = []
        
        for event_key in all_event_keys:
            if event_key in workload_data:
                event = workload_data[event_key]
                temporal_features = self.extract_temporal_stats(
                    event.get('timestamps_ns', []),
                    event.get('num_samples', 0)
                )
            else:
                # Event missing, fill with zeros
                temporal_features = [0] * 10
            
            features.extend(temporal_features)
        
        return np.array(features)
    
    def normalize_features(self, X):
        """
        Normalize features using the scaler from training.
        Each column is normalized independently (same as training).
        
        Args:
            X: Feature matrix (n_samples, n_features)
        
        Returns:
            X_scaled: Normalized feature matrix
        """
        # StandardScaler.transform() normalizes each column independently
        return self.scaler.transform(X)
    
    def predict(self, features_json):
        """
        Run inference on unseen data.
        
        Args:
            features_json: Path to JSON file with PMC features
        
        Returns:
            predictions, probabilities, workload_names
        """
        print(f"\n{'='*60}")
        print(f"Running Inference")
        print(f"{'='*60}")
        print(f"Input file: {features_json}")
        
        # Load features
        with open(features_json, 'r') as f:
            pmc_data = json.load(f)
        
        print(f"  Found {len(pmc_data)} workloads")
        
        # CRITICAL: Use event keys from TRAINING, not from unseen data
        if self.all_event_keys is None:
            raise ValueError("Model does not contain training event keys. "
                           "Please retrain with updated train_classifier.py and use --save-model")
        
        all_event_keys_sorted = self.all_event_keys
        
        # Count events in unseen data
        unseen_event_keys = set()
        for workload_label, events in pmc_data.items():
            unseen_event_keys = unseen_event_keys.union(set(events.keys()))
        
        print(f"  Training events: {len(all_event_keys_sorted)}")
        print(f"  Unseen data events: {len(unseen_event_keys)}")
        
        # Check for missing events
        missing_events = set(all_event_keys_sorted) - unseen_event_keys
        if missing_events:
            print(f"  ⚠️  Missing {len(missing_events)} events in unseen data (will zero-fill)")
        
        # Extract features for each workload
        X = []
        workload_names = []
        
        for workload_label, events in pmc_data.items():
            # Use TRAINING event keys, zero-fill missing events
            features = self.extract_features_from_workload(events, all_event_keys_sorted)
            X.append(features)
            workload_names.append(workload_label)
        
        X = np.array(X)
        print(f"  Feature matrix shape: {X.shape}")
        print(f"  Expected by model: ({len(pmc_data)}, {len(all_event_keys_sorted) * 10})")
        
        # Normalize features using per-feature-type scalers
        X_scaled = self.normalize_features(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Decode labels
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        return predicted_labels, probabilities, workload_names
    
    def print_results(self, predicted_labels, probabilities, workload_names):
        """Pretty print inference results with probabilities."""
        print(f"\n{'='*60}")
        print(f"Inference Results")
        print(f"{'='*60}\n")
        
        for i, (workload, pred_label, probs) in enumerate(zip(workload_names, predicted_labels, probabilities), 1):
            print(f"{i}. Workload: {workload}")
            print(f"   Predicted: {pred_label}")
            print(f"   Confidence: {probs.max():.4f}")
            print(f"   Probability distribution:")
            
            # Sort by probability (descending)
            sorted_indices = np.argsort(probs)[::-1]
            
            for idx in sorted_indices:
                class_name = self.label_encoder.classes_[idx]
                probability = probs[idx]
                bar_length = int(probability * 40)
                bar = '█' * bar_length + '░' * (40 - bar_length)
                print(f"     {class_name:20s} {probability:.4f} |{bar}|")
            
            print()
        
        print(f"{'='*60}\n")
    
    def save_results(self, predicted_labels, probabilities, workload_names, output_file):
        """Save inference results to JSON file."""
        results = []
        
        for workload, pred_label, probs in zip(workload_names, predicted_labels, probabilities):
            result = {
                'workload': workload,
                'predicted_class': pred_label,
                'confidence': float(probs.max()),
                'probabilities': {
                    self.label_encoder.classes_[i]: float(probs[i])
                    for i in range(len(probs))
                }
            }
            results.append(result)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Run inference on unseen PMC features'
    )
    parser.add_argument(
        '--model',
        default='models/pmc_classifier.pkl',
        help='Path to trained model (default: models/pmc_classifier.pkl)'
    )
    parser.add_argument(
        '--features',
        required=True,
        help='Path to PMC features JSON file for inference'
    )
    parser.add_argument(
        '--output',
        help='Optional: Save results to JSON file'
    )
    
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print(f"PMC Classifier Inference")
    print(f"{'='*60}\n")
    
    # Load model and run inference
    inference = PMCInference(args.model)
    predicted_labels, probabilities, workload_names = inference.predict(args.features)
    
    # Print results
    inference.print_results(predicted_labels, probabilities, workload_names)
    
    # Save results if requested
    if args.output:
        inference.save_results(predicted_labels, probabilities, workload_names, args.output)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

