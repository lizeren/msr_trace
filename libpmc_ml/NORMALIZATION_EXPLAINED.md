# Per-Column Normalization: The Correct Approach

## Why Per-Column Normalization is Correct

Each of the **370 columns** in your feature matrix represents a unique `(event, feature_type)` combination, and they should be normalized **independently** across all samples (workloads).

### The Key Insight

**Different events have different sampling periods**, which means they operate at fundamentally different scales:

```
Event 0: sampling_period = 100 cycles
  → Samples frequently, shorter durations, smaller intervals

Event 1: sampling_period = 1000 cycles  
  → Samples rarely, longer durations, larger intervals

Event 2: sampling_period = 500 cycles
  → Somewhere in between
```

If we normalize all `event_0_duration`, `event_1_duration`, ..., `event_36_duration` **together**, we're mixing measurements from different scales - that's wrong!

---

## Feature Matrix Structure

With 37 events and 10 features per event = 370 total columns:

```
Sample (row):  [e0_f0, e0_f1, ..., e0_f9, e1_f0, e1_f1, ..., e1_f9, ..., e36_f0, ..., e36_f9]
                 └─────event 0─────┘ └─────event 1─────┘         └─────event 36────┘
                 
Column indices:  [0,    1,    ..., 9,    10,   11,   ..., 19,   ..., 360,  ..., 369]
```

### What Each Column Represents

- **Column 0**: `event_0_total_duration` (sampled every 100 cycles)
- **Column 10**: `event_1_total_duration` (sampled every 1000 cycles)
- **Column 20**: `event_2_total_duration` (sampled every 500 cycles)
- ...

These are **fundamentally different measurements** - they shouldn't share normalization parameters!

---

## The Correct Approach: Per-Column Normalization

```python
# StandardScaler by default normalizes each column independently
X_train_scaled = scaler.fit_transform(X_train)

# For each column j (j=0 to 369):
#   mean_j = average of X_train[:, j]  (across all samples)
#   std_j = std dev of X_train[:, j]   (across all samples)
#   X_train_scaled[:, j] = (X_train[:, j] - mean_j) / std_j
```

### Example with 3 samples (workloads):

```
Before normalization:
                 col_0    col_10   col_20   ...  col_369
                 (e0_dur) (e1_dur) (e2_dur)     (e36_q75)
encrypt_run1:    100000   500000   250000   ...  80000
encrypt_run2:    110000   520000   260000   ...  82000
decrypt_run1:     50000   400000   200000   ...  65000
                    ↓        ↓        ↓            ↓
          Each column normalized independently
                    ↓        ↓        ↓            ↓
After normalization:
                 col_0    col_10   col_20   ...  col_369
encrypt_run1:     1.15     0.87     0.75    ...   1.02
encrypt_run2:     1.44     1.13     1.25    ...   1.35
decrypt_run1:    -0.58    -0.96    -1.00    ...  -0.89
```

Each column gets its own mean and std computed across the 3 samples.

---

## What Gets Normalized Together vs Separately

### Normalized TOGETHER (across samples):
✅ All instances of `event_5_mean_interval` across different workload samples  
✅ This makes sense - comparing the same event across different workloads

### Normalized SEPARATELY (different columns):
✅ `event_0_mean_interval` vs `event_1_mean_interval` (different events)  
✅ `event_0_duration` vs `event_0_mean_interval` (different feature types)  
✅ This makes sense - they measure different things!

---

## Verification

After normalization, each column should have:
```python
# Check normalization
for col in range(370):
    column_data = X_train_scaled[:, col]
    print(f"Column {col}: mean={column_data.mean():.6f}, std={column_data.std():.6f}")
    # Should output: mean ≈ 0.0, std ≈ 1.0 for each column
```

Expected output:
```
Column 0: mean=0.000000, std=1.000000
Column 1: mean=0.000000, std=1.000000
...
Column 369: mean=0.000000, std=1.000000
```

---

