#!/usr/bin/env python3
"""
1D CNN Classifier for PMC Temporal Features

Trains a 1D CNN on raw PMC event rate sequences.
Each sample is a 2D tensor: [38 events, L timesteps]
"""

import json
import numpy as np
import glob
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import time


class PMCRawDataset(Dataset):
    """
    PyTorch Dataset for PMC raw temporal sequences.
    Each sample: [38 events, L timesteps] of event rates.
    """
    
    def __init__(self, samples, labels, event_stats=None, fit_stats=False):
        """
        Args:
            samples: List of dicts, each with 38 events of rate sequences
            labels: List of string labels
            event_stats: Dict with normalization statistics (mean, std per event)
            fit_stats: If True, compute stats from this data
        """
        self.samples = samples
        self.label_encoder = LabelEncoder()
        self.labels_encoded = self.label_encoder.fit_transform(labels)
        self.num_classes = len(self.label_encoder.classes_)
        self.event_stats = event_stats
        
        # Compute normalization statistics if needed
        if fit_stats:
            self.event_stats = self._compute_normalization_stats()
    
    def _compute_normalization_stats(self):
        """
        Compute per-event (per-channel) mean and std across all samples.
        """
        print(f"\nComputing per-event normalization statistics...")
        
        num_events = len(self.samples[0])
        seq_len = len(next(iter(self.samples[0].values())))
        
        # Collect all values per event
        event_values = {event_idx: [] for event_idx in range(num_events)}
        
        for sample in self.samples:
            for event_idx, rate_seq in sample.items():
                event_values[event_idx].extend(rate_seq)
        
        # Compute mean and std per event
        stats = {}
        for event_idx in range(num_events):
            values = np.array(event_values[event_idx])
            # Only compute stats on non-zero values to avoid padding bias
            non_zero_values = values[values != 0]
            
            if len(non_zero_values) > 0:
                mean = np.mean(non_zero_values)
                std = np.std(non_zero_values)
            else:
                mean = 0.0
                std = 1.0
            
            stats[event_idx] = {'mean': mean, 'std': std}
            print(f"  Event {event_idx}: mean={mean:.4f}, std={std:.4f}")
        
        return stats
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            x: Tensor of shape [38, L] (normalized event rates)
            y: Integer class label
        """
        sample = self.samples[idx]
        label = self.labels_encoded[idx]
        
        # Convert to tensor: [38, L]
        num_events = len(sample)
        seq_len = len(next(iter(sample.values())))
        
        x = np.zeros((num_events, seq_len), dtype=np.float32)
        
        for event_idx, rate_seq in sample.items():
            x[event_idx, :] = rate_seq
            
            # Normalize using training statistics
            if self.event_stats is not None:
                mean = self.event_stats[event_idx]['mean']
                std = self.event_stats[event_idx]['std']
                # Normalize: (x - mean) / (std + eps)
                x[event_idx, :] = (x[event_idx, :] - mean) / (std + 1e-8)
        
        return torch.FloatTensor(x), torch.LongTensor([label])[0]


class PMCDataLoader:
    """Load and preprocess PMC features for CNN."""
    
    def __init__(self, features_pattern: str, seq_len: int = 128, epsilon: float = 1e-9):
        """
        Args:
            features_pattern: Glob pattern for JSON files (e.g., "features/pmc_features_*.json")
            seq_len: Fixed sequence length L
            epsilon: Small value to avoid division by zero
        """
        self.features_pattern = features_pattern
        self.seq_len = seq_len
        self.epsilon = epsilon
        self.samples = []
        self.labels = []
        
    def load_all_features(self):
        """Load all JSON files and extract samples."""
        files = sorted(glob.glob(self.features_pattern))
        
        if not files:
            raise FileNotFoundError(f"No files found matching pattern: {self.features_pattern}")
        
        print(f"\n{'='*60}")
        print(f"Loading PMC Features")
        print(f"{'='*60}")
        print(f"Pattern: {self.features_pattern}")
        print(f"Found {len(files)} files")
        print(f"Sequence length: {self.seq_len}")
        
        for file_path in files:
            self._load_file(file_path)
        
        print(f"\n{'='*60}")
        print(f"Loaded {len(self.samples)} samples total")
        print(f"Unique classes: {len(set(self.labels))}")
        print(f"{'='*60}")
        
        return self.samples, self.labels
    
    def _load_file(self, file_path: str):
        """Load one JSON file and extract all runs as samples."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Each key is a workload (function name), each is one sample
        for workload_label, events in data.items():
            sample = self._process_sample(events, workload_label)
            if sample is not None:
                self.samples.append(sample)
                self.labels.append(workload_label)
    
    def _process_sample(self, events: Dict, workload_label: str) -> Dict[int, List[float]]:
        """
        Process one sample (one run) into [38, L] format.
        
        Returns:
            Dict mapping event_idx -> rate_sequence (length L)
        """
        processed_events = {}
        
        # Sort events by index to ensure consistent ordering
        event_keys = sorted(events.keys(), key=lambda x: int(x.split('_')[1]))
        
        for event_idx, event_key in enumerate(event_keys):
            event_data = events[event_key]
            
            timestamps = event_data.get('timestamps_ns', [])
            sampling_period = event_data.get('sampling_period', 100)
            
            # Convert timestamps to event rates
            rate_seq = self._timestamps_to_rates(timestamps, sampling_period)
            
            # Pad or truncate to fixed length
            rate_seq = self._pad_or_truncate(rate_seq, self.seq_len)
            
            processed_events[event_idx] = rate_seq
        
        # Ensure we have exactly 38 events
        if len(processed_events) != 38:
            # Fill missing events with zeros
            for i in range(38):
                if i not in processed_events:
                    processed_events[i] = [0.0] * self.seq_len
        
        return processed_events
    
    def _timestamps_to_rates(self, timestamps: List[int], sampling_period: int) -> List[float]:
        """
        Convert timestamps to event rates.
        
        Steps:
        1. Compute time differences: Δt[j] = t[j] - t[j-1]
        2. Compute rate: r[j] = sampling_period / Δt[j]
        3. Apply log1p for stability: log(1 + r[j])
        
        Returns:
            List of event rates
        """
        if len(timestamps) == 0:
            return []
        
        rates = []
        
        for j in range(len(timestamps)):
            if j == 0:
                # First timestamp: use the timestamp itself as delta
                delta_t = max(timestamps[0], self.epsilon)
            else:
                # Time difference from previous timestamp
                delta_t = timestamps[j] - timestamps[j-1]
                # Clip to avoid division by zero
                delta_t = max(delta_t, self.epsilon)
            
            # Event rate: events per unit time (nanosecond)
            rate = sampling_period / delta_t
            
            # Apply log1p for numerical stability
            rate_log = np.log1p(rate)
            
            rates.append(rate_log)
        
        return rates
    
    def _pad_or_truncate(self, sequence: List[float], target_len: int) -> List[float]:
        """Pad with zeros or truncate to target length."""
        if len(sequence) >= target_len:
            return sequence[:target_len]
        else:
            # Pad with zeros
            return sequence + [0.0] * (target_len - len(sequence))


class PMC_1DCNN(nn.Module):
    """
    1D CNN for PMC event rate sequences.
    
    Input: [batch_size, 38, L]
    Output: [batch_size, num_classes]
    """
    
    def __init__(self, num_classes: int = 31, seq_len: int = 128, dropout: float = 0.3):
        super(PMC_1DCNN, self).__init__()
        
        # Input: [batch, 38, seq_len]
        
        # Conv block 1
        self.conv1 = nn.Conv1d(38, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        # Conv block 2
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        # Conv block 3
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        
        # Conv block 4
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(512)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # x: [batch, 38, seq_len]
        
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Conv block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        # Global pooling: [batch, 512, seq_len/8] -> [batch, 512, 1]
        x = self.global_pool(x)
        
        # Flatten: [batch, 512]
        x = x.squeeze(-1)
        
        # Fully connected
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience=7, min_delta=0, verbose=True):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as an improvement
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = -np.inf
        
    def __call__(self, val_acc):
        score = val_acc
        
        if self.best_score is None:
            self.best_score = score
            self.val_acc_max = val_acc
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'  EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_acc_max = val_acc
            self.counter = 0


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def test(model, dataloader, device, label_encoder):
    """Test the model and return detailed metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    print(f"\n{'='*60}")
    print(f"Test Set Evaluation")
    print(f"{'='*60}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    print(f"\nClassification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=label_encoder.classes_,
        digits=4
    ))
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(
        description='Train 1D CNN on PMC raw temporal features'
    )
    parser.add_argument(
        '--features',
        default='features/pmc_features_*.json',
        help='Glob pattern for PMC feature JSON files'
    )
    parser.add_argument(
        '--seq-len',
        type=int,
        default=128,
        help='Fixed sequence length L (default: 128)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size (default: 64)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of epochs (default: 50)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate (default: 1e-3)'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-4,
        help='Weight decay (default: 1e-4)'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.3,
        help='Dropout rate (default: 0.3)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience (default: 10)'
    )
    parser.add_argument(
        '--save-model',
        action='store_true',
        help='Save the trained model'
    )
    parser.add_argument(
        '--model-path',
        default='models/pmc_cnn.pt',
        help='Path to save model (default: models/pmc_cnn.pt)'
    )
    
    args = parser.parse_args()
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"PMC 1D CNN Classifier")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"{'='*60}")
    
    # Load data
    data_loader = PMCDataLoader(args.features, seq_len=args.seq_len)
    samples, labels = data_loader.load_all_features()
    
    print(f"\nDataset Statistics:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"  Total samples: {len(samples)}")
    print(f"  Number of classes: {len(unique_labels)}")
    print(f"  Samples per class: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")
    
    # Stratified split: 70% train, 15% val, 15% test
    train_samples, temp_samples, train_labels, temp_labels = train_test_split(
        samples, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    val_samples, test_samples, val_labels, test_labels = train_test_split(
        temp_samples, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    print(f"\n{'='*60}")
    print(f"Data Split")
    print(f"{'='*60}")
    print(f"Train: {len(train_samples)} samples ({len(train_samples)/len(samples)*100:.1f}%)")
    print(f"Val:   {len(val_samples)} samples ({len(val_samples)/len(samples)*100:.1f}%)")
    print(f"Test:  {len(test_samples)} samples ({len(test_samples)/len(samples)*100:.1f}%)")
    
    # Create datasets
    train_dataset = PMCRawDataset(train_samples, train_labels, fit_stats=True)
    val_dataset = PMCRawDataset(val_samples, val_labels, event_stats=train_dataset.event_stats)
    test_dataset = PMCRawDataset(test_samples, test_labels, event_stats=train_dataset.event_stats)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model
    num_classes = train_dataset.num_classes
    model = PMC_1DCNN(num_classes=num_classes, seq_len=args.seq_len, dropout=args.dropout)
    model = model.to(device)
    
    print(f"\n{'='*60}")
    print(f"Model Architecture")
    print(f"{'='*60}")
    print(f"Input shape: [batch_size, 38, {args.seq_len}]")
    print(f"Output: {num_classes} classes")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Training")
    print(f"{'='*60}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Dropout: {args.dropout}")
    print(f"Early stopping patience: {args.patience}")
    print(f"{'='*60}\n")
    
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch [{epoch+1:3d}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if args.save_model:
                os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'label_encoder': train_dataset.label_encoder,
                    'event_stats': train_dataset.event_stats,
                    'seq_len': args.seq_len,
                }, args.model_path)
        
        # Early stopping
        early_stopping(val_acc)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training Complete")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Load best model for testing
    if args.save_model and os.path.exists(args.model_path):
        print(f"\nLoading best model from {args.model_path}")
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test
    test_acc = test(model, test_loader, device, train_dataset.label_encoder)
    
    if args.save_model:
        print(f"\n{'='*60}")
        print(f"Model saved to: {args.model_path}")
        print(f"{'='*60}")
    
    print(f"\nDone!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

