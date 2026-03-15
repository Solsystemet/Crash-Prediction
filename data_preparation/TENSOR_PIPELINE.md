# Tensor Data Pipeline

This module converts traffic crash CSV data into PyTorch tensors ready for training neural networks.

## Quick Start

```python
from data_preparation.prepare_tensor_data import prepare_tensor_data
from data_preparation.tensor_config import SEVERITY_PREDICTION_CONFIG

# Prepare data using a preset config
result = prepare_tensor_data(SEVERITY_PREDICTION_CONFIG)

# Access the datasets
train_dataset = result.train_dataset
val_dataset = result.val_dataset
test_dataset = result.test_dataset

print(f"Training samples: {len(train_dataset)}")
print(f"Features per sample: {train_dataset.num_features}")
```

## Using with PyTorch DataLoader

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    result.train_dataset,
    batch_size=32,
    shuffle=True,
)

for features, labels in train_loader:
    # features: FloatTensor of shape (batch_size, num_features)
    # labels: LongTensor of shape (batch_size,) for classification
    pass
```

## Configuration

### Using Preset Configs

Three preset configurations are available:

```python
from data_preparation.tensor_config import (
    SEVERITY_PREDICTION_CONFIG,  # Predict MOST_SEVERE_INJURY
    FATAL_CRASH_CONFIG,          # Predict INJURIES_FATAL (regression)
    MINIMAL_TEST_CONFIG,         # 3 features only, for quick testing
)
```

### Creating Custom Configs

```python
from data_preparation.tensor_config import TensorConfig

my_config = TensorConfig(
    # What to predict
    target_column="MOST_SEVERE_INJURY",

    # Which columns to use as input features
    feature_columns=[
        "WEATHER_CONDITION",
        "LIGHTING_CONDITION",
        "POSTED_SPEED_LIMIT",
        "CRASH_HOUR",
    ],

    # Which features are categorical (need label encoding)
    categorical_columns=[
        "WEATHER_CONDITION",
        "LIGHTING_CONDITION",
    ],

    # Which features are numerical (need scaling)
    numerical_columns=[
        "POSTED_SPEED_LIMIT",
        "CRASH_HOUR",
    ],

    # Task type determines label tensor dtype
    task_type="classification",  # or "regression"

    # Train/val/test split ratios (must sum to 1.0)
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,

    # Random seed for reproducible splits
    random_seed=42,

    # How to handle missing values
    fill_categorical_na="UNKNOWN",
    fill_numerical_na="median",  # or "mean"
)
```

### Configuration Options

| Parameter             | Type                               | Default            | Description                               |
| --------------------- | ---------------------------------- | ------------------ | ----------------------------------------- |
| `target_column`       | `str`                              | _required_         | Column to predict                         |
| `feature_columns`     | `list[str] \| None`                | `None`             | Input features (None = all except target) |
| `categorical_columns` | `list[str]`                        | `[]`               | Columns for label encoding                |
| `numerical_columns`   | `list[str]`                        | `[]`               | Columns for standard scaling              |
| `task_type`           | `"classification" \| "regression"` | `"classification"` | ML task type                              |
| `train_ratio`         | `float`                            | `0.7`              | Training set proportion                   |
| `val_ratio`           | `float`                            | `0.15`             | Validation set proportion                 |
| `test_ratio`          | `float`                            | `0.15`             | Test set proportion                       |
| `random_seed`         | `int`                              | `42`               | RNG seed for splits                       |
| `fill_categorical_na` | `str`                              | `"UNKNOWN"`        | Fill value for missing categoricals       |
| `fill_numerical_na`   | `"median" \| "mean"`               | `"median"`         | Strategy for missing numericals           |

## Saving and Loading Checkpoints

Save the entire pipeline (datasets + encoders + config) for later use:

```python
# Save
result.save("checkpoints/severity_pipeline.pt")

# Load
from data_preparation.prepare_tensor_data import TensorPipelineResult

loaded = TensorPipelineResult.load("checkpoints/severity_pipeline.pt")
train_dataset = loaded.train_dataset
encoder_registry = loaded.encoder_registry
```

## Encoder Registry

The `encoder_registry` stores all fitted encoders, useful for inference:

```python
registry = result.encoder_registry

# Get number of output classes (for classification)
num_classes = registry.get_num_classes()

# Get number of input features
num_features = registry.get_num_features()

# Decode predictions back to original labels
predictions = model(features)  # tensor of class indices
predicted_labels = registry.target_encoder.inverse_transform(
    predictions.numpy()
)
```

## Available Columns

From `FilteredTrafficCrashesSchema`:

**Categorical:**

- `TRAFFIC_CONTROL_DEVICE` - Type of traffic control (signal, sign, etc.)
- `DEVICE_CONDITION` - Condition of traffic control device
- `WEATHER_CONDITION` - Weather at time of crash
- `LIGHTING_CONDITION` - Lighting at time of crash
- `ROADWAY_SURFACE_COND` - Road surface condition
- `ROAD_DEFECT` - Any road defects present
- `MOST_SEVERE_INJURY` - Most severe injury in crash (common target)

**Numerical:**

- `POSTED_SPEED_LIMIT` - Speed limit at location
- `LANE_CNT` - Number of lanes
- `CRASH_HOUR` - Hour of day (0-23)
- `CRASH_DAY_OF_WEEK` - Day of week (1-7)
- `CRASH_MONTH` - Month (1-12)
- `LATITUDE`, `LONGITUDE` - Location coordinates
- `INJURIES_FATAL`, `INJURIES_INCAPACITATING`, etc. - Injury counts

## Example: Classification Task

```python
from data_preparation.prepare_tensor_data import prepare_tensor_data
from data_preparation.tensor_config import TensorConfig
from torch.utils.data import DataLoader
import torch.nn as nn

# Configure
config = TensorConfig(
    target_column="MOST_SEVERE_INJURY",
    feature_columns=[
        "WEATHER_CONDITION", "LIGHTING_CONDITION",
        "POSTED_SPEED_LIMIT", "CRASH_HOUR", "CRASH_MONTH",
    ],
    categorical_columns=["WEATHER_CONDITION", "LIGHTING_CONDITION"],
    numerical_columns=["POSTED_SPEED_LIMIT", "CRASH_HOUR", "CRASH_MONTH"],
    task_type="classification",
)

# Prepare data
result = prepare_tensor_data(config)

# Create model
num_features = result.train_dataset.num_features
num_classes = result.encoder_registry.get_num_classes()

model = nn.Sequential(
    nn.Linear(num_features, 64),
    nn.ReLU(),
    nn.Linear(64, num_classes),
)

# Training loop
train_loader = DataLoader(result.train_dataset, batch_size=32, shuffle=True)
criterion = nn.CrossEntropyLoss()

for features, labels in train_loader:
    outputs = model(features)
    loss = criterion(outputs, labels)
    # ... backprop, etc.
```

## Example: Regression Task

```python
config = TensorConfig(
    target_column="INJURIES_FATAL",
    feature_columns=[
        "WEATHER_CONDITION", "LIGHTING_CONDITION",
        "POSTED_SPEED_LIMIT", "CRASH_HOUR",
    ],
    categorical_columns=["WEATHER_CONDITION", "LIGHTING_CONDITION"],
    numerical_columns=["POSTED_SPEED_LIMIT", "CRASH_HOUR"],
    task_type="regression",
)

result = prepare_tensor_data(config)

# For regression, labels are FloatTensor
# Use MSELoss or similar
criterion = nn.MSELoss()
```

## Pipeline Flow

```
┌─────────────────┐
│  Load CSV       │  get_traffic_crashes()
└────────┬────────┘
         ▼
┌─────────────────┐
│ Filter Columns  │  Keep only feature_columns + target_column
└────────┬────────┘
         ▼
┌─────────────────┐
│  Fill Missing   │  Categorical: "UNKNOWN", Numerical: median/mean
└────────┬────────┘
         ▼
┌─────────────────┐
│ Encode & Scale  │  LabelEncoder (categorical), StandardScaler (numerical)
└────────┬────────┘
         ▼
┌─────────────────┐
│  Split Data     │  Shuffle → train/val/test by ratios
└────────┬────────┘
         ▼
┌─────────────────┐
│ Convert Tensors │  FloatTensor (features), Long/FloatTensor (labels)
└────────┬────────┘
         ▼
┌─────────────────┐
│ CrashTensorData │  PyTorch Dataset ready for DataLoader
│     Dataset     │
└─────────────────┘
```
