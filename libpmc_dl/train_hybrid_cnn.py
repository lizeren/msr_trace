#!/usr/bin/env python3
"""
Hybrid CNN: Uses statistical features per event (EXACTLY like XGBoost)
Input: [38 events, 10 statistical features] instead of [38 events, 128 timesteps]
"""

import json
import numpy as np
import glob
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score


class HybridCNN(nn.Module):
    """CNN on statistical features [38, N] instead of temporal sequences."""
    
    def __init__(self, num_classes=31, num_features=16, dropout=0.4):
        super().__init__()
        self.num_features = num_features
        
        # Input: [batch, 38, num_features]
        # Treat 38 as "channels" and num_features as "sequence length"
        
        # Conv layers
        self.conv1 = nn.Conv1d(38, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # FC layers
        self.fc1 = nn.Linear(256, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # x: [batch, 38, num_features]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.global_pool(x).squeeze(-1)  # [batch, 256]
        
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class StatisticalDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.label_encoder = LabelEncoder()
        self.labels_encoded = self.label_encoder.fit_transform(labels)
        self.num_classes = len(self.label_encoder.classes_)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.samples[idx]), torch.LongTensor([self.labels_encoded[idx]])[0]


def compute_statistical_features(timestamps, sampling_period):
    """Compute 10 features matching libpmc_ml/classifier.md"""
    if len(timestamps) < 2:
        return np.zeros(10, dtype=np.float32)
    
    intervals = np.diff(timestamps)
    if len(intervals) == 0 or np.all(intervals == 0):
        return np.zeros(10, dtype=np.float32)
    
    features = [
        timestamps[-1] - timestamps[0],
        np.mean(intervals),
        np.std(intervals),
        np.min(intervals),
        np.max(intervals),
        len(timestamps) / (timestamps[-1] - timestamps[0] + 1),
        len(timestamps),
        np.percentile(intervals, 25),
        np.percentile(intervals, 50),
        np.percentile(intervals, 75),
    ]
    
    return np.array(features, dtype=np.float32)


def load_data_with_stats(features_pattern):
    files = sorted(glob.glob(features_pattern))
    samples, labels = [], []
    
    print(f"Loading {len(files)} files...")
    
    for file_idx, file_path in enumerate(files):
        if file_idx % 200 == 0:
            print(f"  Progress: {file_idx}/{len(files)}")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  Warning: Skipping corrupted file: {file_path}")
            continue
        
        for workload_label, events in data.items():
            event_features = []
            event_keys = sorted(events.keys(), key=lambda x: int(x.split('_')[1]))
            
            for event_key in event_keys:
                event_data = events[event_key]
                timestamps = event_data.get('timestamps_ns', [])
                sampling_period = event_data.get('sampling_period', 100)
                stats = compute_statistical_features(timestamps, sampling_period)
                event_features.append(stats)
            
            while len(event_features) < 38:
                event_features.append(np.zeros(10, dtype=np.float32))
            
            samples.append(np.array(event_features[:38]))
            labels.append(workload_label)
    
    print(f"Loaded {len(samples)} samples")
    return samples, labels


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    correct, total = 0, 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return 100.0 * correct / total


def validate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100.0 * correct / total


def test(model, dataloader, device, label_encoder):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {acc*100:.2f}%")
    print(f"\n{classification_report(all_labels, all_preds, target_names=label_encoder.classes_, digits=4)}")
    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', default='../libpmc_dl/features/pmc_features_*.json')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--cache', action='store_true',
                        help='Use pre-computed cached features from features_10/')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    if args.cache:
        print("\n📦 Loading cached features...")
        cache_file = 'features_10/features_cache.pkl'
        if not os.path.exists(cache_file):
            print(f"❌ Cache file not found: {cache_file}")
            print(f"   Run: python3 preprocess_features.py")
            return 1
        
        import pickle
        with open(cache_file, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Cache contains [N, 38, 10] - exactly what we need!
        samples = [cache_data['X'][i] for i in range(len(cache_data['X']))]
        labels = cache_data['y'].tolist()
        print(f"✓ Loaded {len(samples)} samples from cache (shape: [{len(samples)}, 38, 10])")
    else:
        samples, labels = load_data_with_stats(args.features)
    
    # Split
    train_samples, temp_samples, train_labels, temp_labels = train_test_split(
        samples, labels, test_size=0.3, random_state=42, stratify=labels
    )
    val_samples, test_samples, val_labels, test_labels = train_test_split(
        temp_samples, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")
    
    # Normalize (dynamically detect number of features from data)
    num_features = train_samples[0].shape[1] if len(train_samples[0].shape) > 1 else 16
    print(f"Detected {num_features} features per event")
    
    train_arr = np.array(train_samples).reshape(-1, num_features)
    mean = np.mean(train_arr, axis=0)
    std = np.std(train_arr, axis=0) + 1e-8
    
    train_samples = [(s - mean) / std for s in train_samples]
    val_samples = [(s - mean) / std for s in val_samples]
    test_samples = [(s - mean) / std for s in test_samples]
    
    # Datasets
    train_dataset = StatisticalDataset(train_samples, train_labels)
    val_dataset = StatisticalDataset(val_samples, val_labels)
    test_dataset = StatisticalDataset(test_samples, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Model
    model = HybridCNN(num_classes=train_dataset.num_classes, num_features=num_features, dropout=args.dropout).to(device)
    print(f"\nHybrid CNN (statistical features)")
    print(f"Input: [38 events, {num_features} features]")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Train
    print(f"\nTraining...")
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = validate(model, val_loader, device)
        scheduler.step(val_acc)
        
        print(f"Epoch [{epoch+1:3d}/{args.epochs}] Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            if args.save_model:
                os.makedirs('models', exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'label_encoder': train_dataset.label_encoder,
                }, 'models/hybrid_cnn.pt')
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    print(f"\nBest validation: {best_val_acc:.2f}%")
    
    # Test
    if args.save_model and os.path.exists('models/hybrid_cnn.pt'):
        checkpoint = torch.load('models/hybrid_cnn.pt', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    test(model, test_loader, device, train_dataset.label_encoder)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

