# PMC Machine Learning Classifier

Logistic regression classifier for PMC temporal features with proper feature normalization.

## Overview

This classifier distinguishes between different workloads based on their PMC (Performance Monitoring Counter) temporal features. It handles the fact that feature scales vary greatly across different events (e.g., "time to 10 events" vs "time to 10,000 events") by applying StandardScaler normalization.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
# Train on default pmc_features.json
python3 train_classifier.py --features pmc_features.json

# Use custom test/train split
python3 train_classifier.py --features pmc_features.json --test-size 0.3

# Cross-validation only (useful for small datasets)
python3 train_classifier.py --features pmc_features.json --no-split
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--features` | pmc_features.json | Path to PMC features JSON file |
| `--test-size` | 0.2 | Test set size (fraction) |
| `--cv-folds` | 5 | Number of cross-validation folds |
| `--no-split` | False | Skip train/test split, only do CV |

## Feature Engineering

The classifier extracts 10 statistical features per event:

1. **total_duration** - Total time span of all samples
2. **mean_interval** - Average time between consecutive samples
3. **std_interval** - Standard deviation of intervals
4. **min_interval** - Minimum time between samples
5. **max_interval** - Maximum time between samples
6. **sample_rate** - Samples per nanosecond
7. **num_samples** - Total number of samples
8. **q25** - 25th percentile timestamp
9. **q50** - 50th percentile timestamp (median)
10. **q75** - 75th percentile timestamp

## Normalization

**StandardScaler** is applied to all features:
- Centers each feature to **zero mean**
- Scales each feature to **unit variance**

This ensures that features with different scales (e.g., nanoseconds vs. sample counts) are comparable and don't bias the classifier.

**Why normalization matters:**
- Event A might have `mean_interval = 100,000` ns
- Event B might have `num_samples = 5`
- Without normalization, the classifier would be dominated by large-scale features
- StandardScaler makes all features contribute equally

## Model Details

**Logistic Regression** with:
- `solver='lbfgs'` - L-BFGS optimization
- `multi_class='ovr'` - One-vs-Rest for multi-class classification
- `max_iter=1000` - Maximum iterations for convergence

## Output Example

```
============================================================
PMC Logistic Regression Classifier
============================================================
Features file: pmc_features.json

Loading features...
  Found 2 workloads

Extracting feature matrix...
  Feature matrix shape: (2, 300)
  Workloads: ['workload1', 'workload2']
  Total features: 300

============================================================
Feature Normalization (StandardScaler)
============================================================
Train/test split: 1/1
Features normalized to zero mean, unit variance
  Mean (before): [1234.56 789.01 456.78] ...
  Mean (after):  [0.0 0.0 0.0] ...
  Std (before):  [567.89 123.45 678.90] ...
  Std (after):   [1.0 1.0 1.0] ...

Training Logistic Regression...
  Training samples: 1
  Features: 300
  Classes: 2
  Training accuracy: 1.0000

============================================================
Test Set Evaluation
============================================================
Accuracy: 1.0000

Confusion Matrix:
[[1 0]
 [0 1]]

Classification Report:
              precision    recall  f1-score   support
   workload1       1.00      1.00      1.00         1
   workload2       1.00      1.00      1.00         1

============================================================
Top 10 Most Important Features (by coefficient magnitude)
============================================================
 1. MEM_LOAD_RETIRED.L1_MISS_mean_interval              0.123456
 2. BR_MISP_RETIRED.ALL_BRANCHES_total_duration         0.098765
 ...
```

## Notes

- **Small dataset warning**: If you have fewer than 5 samples, the script automatically uses cross-validation only
- **Feature importance**: The script shows the top 10 most important features based on coefficient magnitudes
- **Multi-class support**: Works with any number of workload classes (not just binary classification)

## Extending to Other Classifiers

The feature extraction and normalization pipeline can be easily adapted for other classifiers:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Replace LogisticRegression with:
model = RandomForestClassifier(n_estimators=100, random_state=42)
# or
model = SVC(kernel='rbf', C=1.0, random_state=42)
# or
model = MLPClassifier(hidden_layers=(100, 50), max_iter=1000, random_state=42)
```

## Input Format

Expected JSON structure:

```json
{
  "workload1": {
    "event_0": {
      "event_name": "BR_MISP_RETIRED.ALL_BRANCHES",
      "mode": "sampling",
      "sampling_period": 100,
      "num_runs": 5,
      "timestamps_ns": [0, 252746, 328521, 422791, 503381],
      "num_samples": 5,
      "stats": {...}
    },
    "event_1": {...}
  },
  "workload2": {...}
}
```

