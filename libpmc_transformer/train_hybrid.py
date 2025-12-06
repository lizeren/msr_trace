#!/usr/bin/env python3
"""
Hybrid Transformer: Uses statistical features per event (EXACTLY like XGBoost)
but lets Transformer learn cross-event patterns.

Input: [38 events, 10 statistical features] instead of [38 events, 128 timesteps]

Features match libpmc_ml/classifier.md:
1. total_duration, 2. mean_interval, 3. std_interval, 4. min_interval, 5. max_interval
6. sample_rate, 7. num_samples, 8. q25, 9. q50 (median), 10. q75
"""

import json
import numpy as np
import glob
import argparse
import os
from typing import Dict, List
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score


class HybridTransformer(nn.Module):
    """Transformer on statistical features."""
    
    def __init__(self, num_classes=31, num_events=38, num_features=10, 
                 d_model=128, nhead=8, num_layers=4, dropout=0.3):
        super().__init__()
        
        # Project [38, 10] to [38, d_model]
        self.input_proj = nn.Linear(num_features, d_model)
        
        # Event embedding (learnable positional encoding for events)
        self.event_embedding = nn.Parameter(torch.randn(1, num_events, d_model))
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # x: [batch, 38, 10]
        x = self.input_proj(x)  # [batch, 38, d_model]
        x = x + self.event_embedding  # Add event embeddings
        x = self.transformer(x)  # [batch, 38, d_model]
        x = x.mean(dim=1)  # Global pool: [batch, d_model]
        x = self.dropout(x)
        x = self.fc(x)
        return x


class StatisticalDataset(Dataset):
    """Dataset with 10 statistical features per event."""
    
    def __init__(self, samples, labels):
        self.samples = samples  # [N, 38, 10]
        self.label_encoder = LabelEncoder()
        self.labels_encoded = self.label_encoder.fit_transform(labels)
        self.num_classes = len(self.label_encoder.classes_)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.samples[idx]), torch.LongTensor([self.labels_encoded[idx]])[0]


def compute_statistical_features(timestamps: List[int], sampling_period: int) -> np.ndarray:
    """
    Compute 10 statistical features on timestamp intervals (same as XGBoost ML uses).
    
    Features match libpmc_ml/classifier.md exactly:
    1. total_duration, 2. mean_interval, 3. std_interval, 4. min_interval, 5. max_interval
    6. sample_rate, 7. num_samples, 8. q25, 9. q50 (median), 10. q75
    """
    if len(timestamps) < 2:
        return np.zeros(10, dtype=np.float32)
    
    # Compute intervals between consecutive timestamps
    intervals = np.diff(timestamps)
    
    # Handle empty or all-zero intervals
    if len(intervals) == 0 or np.all(intervals == 0):
        return np.zeros(10, dtype=np.float32)
    
    # 10 features matching ML classifier
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


def load_data_with_stats(features_pattern: str, seq_len: int = 128):
    """Load data and compute statistical features on timestamp intervals."""
    files = sorted(glob.glob(features_pattern))
    samples = []
    labels = []
    
    print(f"Loading {len(files)} files and computing timestamp interval statistics...")
    
    for file_idx, file_path in enumerate(files):
        if file_idx % 200 == 0:
            print(f"  Progress: {file_idx}/{len(files)} files...")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        for workload_label, events in data.items():
            # Process into statistical features
            event_features = []
            event_keys = sorted(events.keys(), key=lambda x: int(x.split('_')[1]))
            
            for event_key in event_keys:
                event_data = events[event_key]
                timestamps = event_data.get('timestamps_ns', [])
                sampling_period = event_data.get('sampling_period', 100)
                
                # Compute stats directly on timestamps (same as ML classifier)
                stats = compute_statistical_features(timestamps, sampling_period)
                event_features.append(stats)
            
            # Ensure 38 events
            while len(event_features) < 38:
                event_features.append(np.zeros(10, dtype=np.float32))
            
            samples.append(np.array(event_features[:38]))  # [38, 10]
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
    parser = argparse.ArgumentParser(description='Hybrid Transformer with statistical features')
    parser.add_argument('--features', default='../libpmc_dl/features/pmc_features_*.json')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--d-model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--save-model', action='store_true')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data with statistical features
    samples, labels = load_data_with_stats(args.features)
    
    # Split
    train_samples, temp_samples, train_labels, temp_labels = train_test_split(
        samples, labels, test_size=0.3, random_state=42, stratify=labels
    )
    val_samples, test_samples, val_labels, test_labels = train_test_split(
        temp_samples, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    print(f"Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")
    
    # Normalize features
    train_arr = np.array(train_samples).reshape(-1, 10)
    mean = np.mean(train_arr, axis=0)
    std = np.std(train_arr, axis=0) + 1e-8
    
    train_samples = [(s - mean) / std for s in train_samples]
    val_samples = [(s - mean) / std for s in val_samples]
    test_samples = [(s - mean) / std for s in test_samples]
    
    # Create datasets
    train_dataset = StatisticalDataset(train_samples, train_labels)
    val_dataset = StatisticalDataset(val_samples, val_labels)
    test_dataset = StatisticalDataset(test_samples, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Model
    model = HybridTransformer(
        num_classes=train_dataset.num_classes,
        num_events=38,
        num_features=10,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    print(f"\nHybrid Transformer (statistical features)")
    print(f"Input: [38 events, 10 features] -> learns cross-event patterns")
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
                }, 'models/hybrid_transformer.pt')
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    print(f"\nBest validation: {best_val_acc:.2f}%")
    
    # Test
    if args.save_model and os.path.exists('models/hybrid_transformer.pt'):
        checkpoint = torch.load('models/hybrid_transformer.pt', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    test(model, test_loader, device, train_dataset.label_encoder)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

