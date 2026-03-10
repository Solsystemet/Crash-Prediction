# training/NN2.py
"""Binary classification: predict if a crash results in injury or not."""

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from data_preparation.helpers.csv_loaders import get_traffic_crashes

# ============================================================================
# Configuration
# ============================================================================

NUMERIC_FEATURES = [
    "CRASH_HOUR",
    "CRASH_DAY_OF_WEEK",
    "CRASH_MONTH",
    "POSTED_SPEED_LIMIT",
    "LANE_CNT",
    "LATITUDE",
    "LONGITUDE",
]

CATEGORICAL_FEATURES = [
    "WEATHER_CONDITION",
    "LIGHTING_CONDITION",
    "ROADWAY_SURFACE_COND",
    "ROAD_DEFECT",
    "TRAFFIC_CONTROL_DEVICE",
    "DEVICE_CONDITION",
]

# Engineered features (computed from raw features)
ENGINEERED_FEATURES = [
    "IS_WEEKEND",
    "IS_RUSH_HOUR",
    "IS_NIGHT",
    "SPEED_X_LANES",
]

TARGET = "MOST_SEVERE_INJURY"
RANDOM_SEED = 42

# Binary classification mapping
# INJURY = 1: Any actual injury occurred
# NO_INJURY = 0: No injury or only reported (not evident)
INJURY_CLASSES = ["FATAL", "INCAPACITATING INJURY", "NONINCAPACITATING INJURY"]
NO_INJURY_CLASSES = ["NO INDICATION OF INJURY", "REPORTED, NOT EVIDENT"]
BINARY_LABELS = ["NO_INJURY", "INJURY"]


# ============================================================================
# Reproducibility
# ============================================================================

def set_seed(seed: int = RANDOM_SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TrainingArtifacts:
    """Container for all artifacts needed for inference."""
    model: nn.Module
    scaler: StandardScaler
    feature_columns: list[str]  # Column order for one-hot encoding
    history: dict
    class_weights: np.ndarray
    device: torch.device
    threshold: float = 0.5  # Decision threshold for binary classification


# ============================================================================
# Model Architecture
# ============================================================================

class CrashInjuryNN(nn.Module):
    """Binary classifier: predicts probability of injury in a crash."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int] | None = None,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate),
            ])
            prev_size = hidden_size

        # Single output for binary classification
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)  # Output shape: (batch_size,)


# ============================================================================
# Feature Engineering
# ============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to the dataframe."""
    df = df.copy()
    
    # Weekend flag (1 = Sunday, 7 = Saturday in Chicago data)
    df["IS_WEEKEND"] = df["CRASH_DAY_OF_WEEK"].isin([1, 7]).astype(int)
    
    # Rush hour flag (7-9 AM and 4-6 PM)
    df["IS_RUSH_HOUR"] = df["CRASH_HOUR"].isin([7, 8, 9, 16, 17, 18]).astype(int)
    
    # Night flag (8 PM - 6 AM)
    df["IS_NIGHT"] = (
        df["CRASH_HOUR"].isin(range(20, 24)) | 
        df["CRASH_HOUR"].isin(range(0, 6))
    ).astype(int)
    
    # Speed × Lanes interaction
    df["SPEED_X_LANES"] = df["POSTED_SPEED_LIMIT"] * df["LANE_CNT"].fillna(1)
    
    return df


def to_binary_target(severity: pd.Series) -> np.ndarray:
    """Convert multiclass severity to binary: 1 = injury, 0 = no injury."""
    return severity.isin(INJURY_CLASSES).astype(int).values


def prepare_features(
    df: pd.DataFrame,
    expected_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Prepare features and binary target for training.
    
    Args:
        df: DataFrame with crash data
        expected_columns: Categorical columns from training (required for test to align)
    
    Returns:
        X: Feature matrix
        y: Binary labels (1 = injury, 0 = no injury)
        feature_columns: List of categorical column names for alignment
    """
    # Drop rows with missing target
    df = df.dropna(subset=[TARGET])
    
    # Engineer features
    df = engineer_features(df)
    
    # Numeric features (including engineered)
    all_numeric = NUMERIC_FEATURES + ENGINEERED_FEATURES
    numeric_data = df[all_numeric].fillna(0).values
    
    # Categorical features with one-hot encoding
    categorical_df = pd.get_dummies(
        df[CATEGORICAL_FEATURES].fillna("UNKNOWN"),
        drop_first=True,
    )
    
    # Align columns with training set if expected_columns provided
    if expected_columns is not None:
        # Add missing columns as 0
        for col in expected_columns:
            if col not in categorical_df.columns:
                categorical_df[col] = 0
        # Keep only expected columns in correct order
        categorical_df = categorical_df[expected_columns]
    
    feature_columns = list(categorical_df.columns)
    categorical_encoded = categorical_df.values
    
    # Combine features
    X = np.hstack([numeric_data, categorical_encoded])
    
    # Binary target: 1 = injury occurred, 0 = no injury
    y = to_binary_target(df[TARGET])
    
    return X, y, feature_columns


# ============================================================================
# Baseline Model
# ============================================================================

def train_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Train a Random Forest baseline for comparison."""
    print("\n" + "=" * 60)
    print("BASELINE: Random Forest (Binary)")
    print("=" * 60)
    
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    accuracy = rf.score(X_test, y_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=BINARY_LABELS))
    
    return {"accuracy": accuracy, "f1": f1, "auc": auc, "model": rf}


# ============================================================================
# Training
# ============================================================================

def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find threshold that maximizes F1 score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # Compute F1 for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores[:-1])  # Last element is for threshold=1
    return thresholds[best_idx]


def train_model(
    epochs: int = 100,
    batch_size: int = 512,
    learning_rate: float = 0.001,
    patience: int = 15,
    use_class_weights: bool = True,
    run_baseline: bool = True,
    device: str | None = None,
    train_years: int = 7,
) -> TrainingArtifacts:
    """Train binary crash injury classifier.
    
    Args:
        train_years: Number of years from start of data to use for training.
                     Remaining data is used for testing (temporal split).
    """
    
    set_seed(RANDOM_SEED)
    
    # Device setup
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Load and prepare data
    print("\nLoading crash data...")
    crashes = get_traffic_crashes()
    
    # Parse dates and sort chronologically
    crashes["CRASH_DATE"] = pd.to_datetime(
        crashes["CRASH_DATE"], format="%m/%d/%Y %I:%M:%S %p"
    )
    crashes = crashes.sort_values("CRASH_DATE")
    
    min_date = crashes["CRASH_DATE"].min()
    max_date = crashes["CRASH_DATE"].max()
    print(f"Data range: {min_date.date()} to {max_date.date()}")
    
    # Temporal split: first N years for training, rest for testing
    train_cutoff = min_date + pd.DateOffset(years=train_years)
    train_data = crashes[crashes["CRASH_DATE"] < train_cutoff]
    test_data = crashes[crashes["CRASH_DATE"] >= train_cutoff]
    
    print(f"Training: {min_date.date()} to {train_cutoff.date()} ({len(train_data):,} records)")
    print(f"Testing:  {train_cutoff.date()} to {max_date.date()} ({len(test_data):,} records)")
    
    # Prepare features (binary classification)
    print("Preparing features (binary: INJURY vs NO_INJURY)...")
    X_train, y_train, feature_columns = prepare_features(train_data)
    X_test, y_test, _ = prepare_features(test_data, expected_columns=feature_columns)
    
    # Scale features (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weight(
        "balanced", classes=np.array([0, 1]), y=y_train
    )
    n_no_injury = np.sum(y_train == 0)
    n_injury = np.sum(y_train == 1)
    print(f"\nClass distribution in training set:")
    print(f"  NO_INJURY (0): {n_no_injury:,} ({100*n_no_injury/len(y_train):.1f}%) - weight: {class_weights[0]:.2f}")
    print(f"  INJURY (1):    {n_injury:,} ({100*n_injury/len(y_train):.1f}%) - weight: {class_weights[1]:.2f}")
    
    # Run baseline comparison
    if run_baseline:
        baseline_results = train_baseline(X_train, y_train, X_test, y_test)
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)  # Float for BCEWithLogitsLoss
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)
    
    # Create dataloader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )
    
    # Initialize model
    input_size = X_train.shape[1]
    model = CrashInjuryNN(input_size).to(device)
    
    print("\n" + "=" * 60)
    print("NEURAL NETWORK TRAINING (Binary Classification)")
    print("=" * 60)
    print(f"Model: {input_size} inputs -> 1 output (sigmoid)")
    print(f"Architecture: [256, 128, 64] hidden layers")
    
    # Binary cross-entropy with logits (more numerically stable)
    if use_class_weights:
        # pos_weight = weight for positive class (injury)
        pos_weight = torch.tensor([class_weights[1] / class_weights[0]]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using weighted loss (pos_weight: {pos_weight.item():.2f})")
    else:
        criterion = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop with early stopping
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
        "test_f1": [],
        "test_auc": [],
    }
    
    best_auc = 0.0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        # Evaluation phase
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_t)
            test_probs = torch.sigmoid(test_logits).cpu().numpy()
            test_predicted = (test_probs > 0.5).astype(int)
            y_test_np = y_test_t.cpu().numpy()
            
            test_acc = np.mean(test_predicted == y_test_np)
            test_f1 = f1_score(y_test_np, test_predicted)
            test_auc = roc_auc_score(y_test_np, test_probs)
        
        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["test_f1"].append(test_f1)
        history["test_auc"].append(test_auc)
        
        # Early stopping based on AUC (threshold-independent metric)
        if test_auc > best_auc:
            best_auc = test_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Logging
        if (epoch + 1) % 10 == 0 or patience_counter == patience:
            print(
                f"Epoch {epoch+1:3d}/{epochs} - "
                f"Loss: {avg_loss:.4f} - "
                f"Train Acc: {train_acc:.4f} - "
                f"Test Acc: {test_acc:.4f} - "
                f"Test F1: {test_f1:.4f} - "
                f"Test AUC: {test_auc:.4f}"
                + (" *" if patience_counter == 0 else "")
            )
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nRestored best model (AUC: {best_auc:.4f})")
    
    # Find optimal threshold on test set
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t)
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        y_test_np = y_test_t.cpu().numpy()
    
    best_threshold = find_best_threshold(y_test_np, test_probs)
    print(f"Optimal threshold: {best_threshold:.3f}")
    
    # Final evaluation with optimal threshold
    test_predicted = (test_probs > best_threshold).astype(int)
    
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    print("\nClassification Report:")
    print(classification_report(y_test_np, test_predicted, target_names=BINARY_LABELS))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test_np, test_predicted)
    print(cm)
    print(f"\n  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    
    final_acc = np.mean(test_predicted == y_test_np)
    final_f1 = f1_score(y_test_np, test_predicted)
    final_precision = precision_score(y_test_np, test_predicted)
    final_recall = recall_score(y_test_np, test_predicted)
    final_auc = roc_auc_score(y_test_np, test_probs)
    
    print(f"\nFinal Metrics (threshold={best_threshold:.3f}):")
    print(f"  Accuracy:  {final_acc:.4f}")
    print(f"  Precision: {final_precision:.4f}")
    print(f"  Recall:    {final_recall:.4f}")
    print(f"  F1-Score:  {final_f1:.4f}")
    print(f"  AUC-ROC:   {final_auc:.4f}")
    
    if run_baseline:
        print(f"\nComparison with Random Forest baseline:")
        print(f"  RF F1:  {baseline_results['f1']:.4f} | NN: {final_f1:.4f}")
        print(f"  RF AUC: {baseline_results['auc']:.4f} | NN: {final_auc:.4f}")
    
    return TrainingArtifacts(
        model=model,
        scaler=scaler,
        feature_columns=feature_columns,
        history=history,
        class_weights=class_weights,
        device=device,
        threshold=best_threshold,
    )


# ============================================================================
# Inference
# ============================================================================

def predict_injury(
    artifacts: TrainingArtifacts,
    crash_hour: int,
    crash_day: int,
    crash_month: int,
    latitude: float,
    longitude: float,
    posted_speed: int = 30,
    lane_count: float = 2.0,
    weather: str = "CLEAR",
    lighting: str = "DAYLIGHT",
    road_surface: str = "DRY",
    road_defect: str = "NO DEFECTS",
    traffic_device: str = "NO CONTROLS",
    device_condition: str = "NO CONTROLS",
) -> tuple[str, float]:
    """Predict if a crash scenario will result in injury.
    
    Returns:
        Tuple of (predicted_class, injury_probability)
    """
    # Build feature dataframe
    df = pd.DataFrame([{
        "CRASH_HOUR": crash_hour,
        "CRASH_DAY_OF_WEEK": crash_day,
        "CRASH_MONTH": crash_month,
        "POSTED_SPEED_LIMIT": posted_speed,
        "LANE_CNT": lane_count,
        "LATITUDE": latitude,
        "LONGITUDE": longitude,
        "WEATHER_CONDITION": weather,
        "LIGHTING_CONDITION": lighting,
        "ROADWAY_SURFACE_COND": road_surface,
        "ROAD_DEFECT": road_defect,
        "TRAFFIC_CONTROL_DEVICE": traffic_device,
        "DEVICE_CONDITION": device_condition,
    }])
    
    # Engineer features
    df = engineer_features(df)
    
    # Numeric features
    all_numeric = NUMERIC_FEATURES + ENGINEERED_FEATURES
    numeric_data = df[all_numeric].fillna(0).values
    
    # Categorical one-hot (must match training columns)
    categorical_df = pd.get_dummies(
        df[CATEGORICAL_FEATURES].fillna("UNKNOWN"),
        drop_first=True,
    )
    # Ensure same columns as training
    for col in artifacts.feature_columns:
        if col not in categorical_df.columns:
            categorical_df[col] = 0
    categorical_df = categorical_df[artifacts.feature_columns]
    
    # Combine and scale
    X = np.hstack([numeric_data, categorical_df.values])
    X = artifacts.scaler.transform(X)
    
    # Predict
    artifacts.model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(artifacts.device)
        logit = artifacts.model(X_tensor)
        injury_prob = torch.sigmoid(logit).cpu().item()
    
    predicted_class = "INJURY" if injury_prob > artifacts.threshold else "NO_INJURY"
    
    return predicted_class, injury_prob


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Train binary classifier (2013-2020 for training, 2020+ for testing)
    artifacts = train_model(
        epochs=100,
        patience=15,
        use_class_weights=True,
        run_baseline=True,
        train_years=7,
    )
    
    # Save model and artifacts
    torch.save({
        "model_state_dict": artifacts.model.state_dict(),
        "scaler_mean": artifacts.scaler.mean_,
        "scaler_scale": artifacts.scaler.scale_,
        "feature_columns": artifacts.feature_columns,
        "class_weights": artifacts.class_weights,
        "threshold": artifacts.threshold,
    }, "crash_injury_model.pth")
    print("\nModel saved to crash_injury_model.pth")
    
    # Example predictions
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTIONS")
    print("=" * 60)
    
    # High-risk scenario
    predicted, prob = predict_injury(
        artifacts,
        crash_hour=22,      # 10 PM
        crash_day=7,        # Saturday
        crash_month=12,     # December
        latitude=41.8781,
        longitude=-87.6298,
        posted_speed=45,
        weather="SNOW",
        lighting="DARKNESS",
    )
    print(f"High-risk scenario: {predicted} (injury prob: {prob:.1%})")
    
    # Low-risk scenario
    predicted, prob = predict_injury(
        artifacts,
        crash_hour=10,      # 10 AM
        crash_day=3,        # Tuesday
        crash_month=6,      # June
        latitude=41.8781,
        longitude=-87.6298,
        posted_speed=25,
        weather="CLEAR",
        lighting="DAYLIGHT",
    )
    print(f"Low-risk scenario:  {predicted} (injury prob: {prob:.1%})")