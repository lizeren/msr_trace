#!/usr/bin/env python3
"""
Dual-Stream CNN: Uses BOTH statistical features AND timeseries data.

Architecture:
- Stream 1: CNN processes timeseries [38 events, 128 timesteps]
- Stream 2: CNN processes statistical features [38 events, 10 features]
- Fusion: Combine both streams before classification

This should capture both temporal patterns and statistical summaries.
"""

import json
import numpy as np
import glob
import argparse
import os
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score


class DualStreamCNN(nn.Module):
    """Dual-stream CNN: processes both timeseries and statistical features."""
    
    def __init__(self, num_classes=31, seq_len=128, dropout=0.4):
        super().__init__()
        
        # Stream 1: Timeseries CNN [38, seq_len]
        self.time_conv1 = nn.Conv1d(38, 64, kernel_size=7, padding=3)
        self.time_bn1 = nn.BatchNorm1d(64)
        self.time_pool1 = nn.MaxPool1d(2)
        
        self.time_conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.time_bn2 = nn.BatchNorm1d(128)
        self.time_pool2 = nn.MaxPool1d(2)
        
        self.time_conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.time_bn3 = nn.BatchNorm1d(256)
        
        self.time_global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Stream 2: Statistical features CNN [38, 10]
        self.stats_conv1 = nn.Conv1d(38, 64, kernel_size=3, padding=1)
        self.stats_bn1 = nn.BatchNorm1d(64)
        
        self.stats_conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.stats_bn2 = nn.BatchNorm1d(128)
        
        self.stats_conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.stats_bn3 = nn.BatchNorm1d(256)
        
        self.stats_global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fusion and classification
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),  # 256 from time + 256 from stats
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, timeseries, stats):
        # timeseries: [batch, 38, seq_len]
        # stats: [batch, 38, 10]
        
        # Stream 1: Process timeseries
        x_time = F.relu(self.time_bn1(self.time_conv1(timeseries)))
        x_time = self.time_pool1(x_time)
        x_time = F.relu(self.time_bn2(self.time_conv2(x_time)))
        x_time = self.time_pool2(x_time)
        x_time = F.relu(self.time_bn3(self.time_conv3(x_time)))
        x_time = self.time_global_pool(x_time).squeeze(-1)  # [batch, 256]
        
        # Stream 2: Process statistical features
        x_stats = F.relu(self.stats_bn1(self.stats_conv1(stats)))
        x_stats = F.relu(self.stats_bn2(self.stats_conv2(x_stats)))
        x_stats = F.relu(self.stats_bn3(self.stats_conv3(x_stats)))
        x_stats = self.stats_global_pool(x_stats).squeeze(-1)  # [batch, 256]
        
        # Fusion
        x_fused = torch.cat([x_time, x_stats], dim=1)  # [batch, 512]
        x_fused = self.fusion(x_fused)  # [batch, 256]
        
        # Classification
        out = self.classifier(x_fused)
        return out


class DualStreamDataset(Dataset):
    """Dataset with both timeseries and statistical features."""
    
    def __init__(self, timeseries_samples, stat_samples, labels):
        self.timeseries_samples = timeseries_samples  # [N, 38, 128]
        self.stat_samples = stat_samples  # [N, 38, 10]
        self.label_encoder = LabelEncoder()
        self.labels_encoded = self.label_encoder.fit_transform(labels)
        self.num_classes = len(self.label_encoder.classes_)
    
    def __len__(self):
        return len(self.timeseries_samples)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.timeseries_samples[idx]),
            torch.FloatTensor(self.stat_samples[idx]),
            torch.LongTensor([self.labels_encoded[idx]])[0]
        )


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


def timestamps_to_sequence(timestamps: List[int], sampling_period: int, seq_len: int = 128) -> np.ndarray:
    """
    Convert timestamps to a fixed-length time series sequence.
    Uses binning approach to create density representation.
    """
    if len(timestamps) < 2:
        return np.zeros(seq_len, dtype=np.float32)
    
    # Create time bins
    min_time = timestamps[0]
    max_time = timestamps[-1]
    duration = max_time - min_time
    
    if duration == 0:
        return np.zeros(seq_len, dtype=np.float32)
    
    # Bin timestamps into fixed-length sequence
    bins = np.linspace(min_time, max_time, seq_len + 1)
    counts, _ = np.histogram(timestamps, bins=bins)
    
    # Normalize to [0, 1] range
    if counts.max() > 0:
        counts = counts.astype(np.float32) / counts.max()
    
    return counts.astype(np.float32)


def load_dual_data(features_pattern: str, seq_len: int = 128) -> Tuple[List, List, List]:
    """Load data and create both timeseries sequences and statistical features."""
    files = sorted(glob.glob(features_pattern))
    timeseries_samples = []
    stat_samples = []
    labels = []
    
    print(f"Loading {len(files)} files with dual-stream data...")
    
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
            timeseries_features = []
            stat_features = []
            
            # Sort events by event number
            event_keys = sorted(events.keys(), key=lambda x: int(x.split('_')[1]))
            
            for event_key in event_keys:
                event_data = events[event_key]
                timestamps = event_data.get('timestamps_ns', [])
                sampling_period = event_data.get('sampling_period', 100)
                
                # Create timeseries representation
                timeseries_seq = timestamps_to_sequence(timestamps, sampling_period, seq_len)
                timeseries_features.append(timeseries_seq)
                
                # Create statistical features
                stats = compute_statistical_features(timestamps, sampling_period)
                stat_features.append(stats)
            
            # Pad to 38 events
            while len(timeseries_features) < 38:
                timeseries_features.append(np.zeros(seq_len, dtype=np.float32))
                stat_features.append(np.zeros(10, dtype=np.float32))
            
            timeseries_samples.append(np.array(timeseries_features[:38]))  # [38, seq_len]
            stat_samples.append(np.array(stat_features[:38]))  # [38, 10]
            labels.append(workload_label)
    
    print(f"Loaded {len(timeseries_samples)} samples")
    return timeseries_samples, stat_samples, labels


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    correct, total = 0, 0
    total_loss = 0.0
    
    for timeseries, stats, labels in dataloader:
        timeseries, stats, labels = timeseries.to(device), stats.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(timeseries, stats)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return 100.0 * correct / total, total_loss / len(dataloader)


def validate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for timeseries, stats, labels in dataloader:
            timeseries, stats, labels = timeseries.to(device), stats.to(device), labels.to(device)
            outputs = model(timeseries, stats)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100.0 * correct / total


def test(model, dataloader, device, label_encoder):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for timeseries, stats, labels in dataloader:
            timeseries, stats, labels = timeseries.to(device), stats.to(device), labels.to(device)
            outputs = model(timeseries, stats)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    
    print(f"\nTest Accuracy: {acc*100:.2f}%")
    print(f"Balanced Accuracy: {balanced_acc*100:.2f}%")
    print(f"\n{classification_report(all_labels, all_preds, target_names=label_encoder.classes_, digits=4)}")
    
    return acc, balanced_acc


def main():
    parser = argparse.ArgumentParser(description='Dual-Stream CNN with timeseries + stats')
    parser.add_argument('--features', default='features/pmc_features_*.json')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seq-len', type=int, default=128, help='Timeseries sequence length')
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--class-weights', action='store_true', help='Use class weighting for imbalanced data')
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--cache', action='store_true',
                        help='Use pre-computed cached features from features_16/')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Timeseries sequence length: {args.seq_len}")
    
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
        
        # Cache contains [N, 38, 10] statistical features
        # We need to create timeseries data from original JSON files
        print("⚠️  Note: Cache only contains statistical features.")
        print("   Loading JSON for timeseries data...")
        timeseries_samples, stat_samples_from_json, labels = load_dual_data(args.features, seq_len=args.seq_len)
        
        # Use cached stats but loaded timeseries
        stat_samples = cache_data['X']  # [N, 38, 10]
        print(f"✓ Loaded {len(stat_samples)} samples")
        print(f"  Stats from cache: {stat_samples.shape}")
        print(f"  Timeseries computed: {len(timeseries_samples)} samples")
    else:
        timeseries_samples, stat_samples, labels = load_dual_data(args.features, seq_len=args.seq_len)
    
    # Split
    indices = list(range(len(labels)))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, random_state=42, stratify=labels
    )
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    # Extract splits
    train_timeseries = [timeseries_samples[i] for i in train_idx]
    train_stats = [stat_samples[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    
    val_timeseries = [timeseries_samples[i] for i in val_idx]
    val_stats = [stat_samples[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    
    test_timeseries = [timeseries_samples[i] for i in test_idx]
    test_stats = [stat_samples[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    
    print(f"Train: {len(train_timeseries)} | Val: {len(val_timeseries)} | Test: {len(test_timeseries)}")
    
    # Normalize timeseries
    train_time_arr = np.array(train_timeseries).reshape(-1, args.seq_len)
    time_mean = np.mean(train_time_arr, axis=0)
    time_std = np.std(train_time_arr, axis=0) + 1e-8
    
    train_timeseries = [(s - time_mean) / time_std for s in train_timeseries]
    val_timeseries = [(s - time_mean) / time_std for s in val_timeseries]
    test_timeseries = [(s - time_mean) / time_std for s in test_timeseries]
    
    # Normalize stats
    train_stats_arr = np.array(train_stats).reshape(-1, 10)
    stats_mean = np.mean(train_stats_arr, axis=0)
    stats_std = np.std(train_stats_arr, axis=0) + 1e-8
    
    train_stats = [(s - stats_mean) / stats_std for s in train_stats]
    val_stats = [(s - stats_mean) / stats_std for s in val_stats]
    test_stats = [(s - stats_mean) / stats_std for s in test_stats]
    
    # Create datasets
    train_dataset = DualStreamDataset(train_timeseries, train_stats, train_labels)
    val_dataset = DualStreamDataset(val_timeseries, val_stats, val_labels)
    test_dataset = DualStreamDataset(test_timeseries, test_stats, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Model
    model = DualStreamCNN(
        num_classes=train_dataset.num_classes,
        seq_len=args.seq_len,
        dropout=args.dropout
    ).to(device)
    
    print(f"\nDual-Stream CNN")
    print(f"Stream 1: Timeseries [38 events, {args.seq_len} timesteps]")
    print(f"Stream 2: Statistical [38 events, 10 features]")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function with optional class weighting
    if args.class_weights:
        class_counts = np.bincount(train_dataset.labels_encoded)
        class_weights = 1.0 / class_counts
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        print("Using class weights for imbalanced data")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Train
    print(f"\nTraining...")
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        train_acc, train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = validate(model, val_loader, device)
        scheduler.step(val_acc)
        
        print(f"Epoch [{epoch+1:3d}/{args.epochs}] Train: {train_acc:.2f}% (Loss: {train_loss:.4f}) | Val: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            if args.save_model:
                os.makedirs('models', exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'label_encoder': train_dataset.label_encoder,
                    'time_mean': time_mean,
                    'time_std': time_std,
                    'stats_mean': stats_mean,
                    'stats_std': stats_std,
                    'seq_len': args.seq_len,
                }, 'models/dual_stream_cnn.pt')
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    print(f"\nBest validation: {best_val_acc:.2f}%")
    
    # Test
    if args.save_model and os.path.exists('models/dual_stream_cnn.pt'):
        checkpoint = torch.load('models/dual_stream_cnn.pt', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    test(model, test_loader, device, train_dataset.label_encoder)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

