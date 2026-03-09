# training/severity_predictor.py
"""Neural network to predict crash severity from crash data features."""

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
    label_encoder: LabelEncoder
    scaler: StandardScaler
    feature_columns: list[str]  # Column order for one-hot encoding
    history: dict
    class_weights: np.ndarray
    device: torch.device


# ============================================================================
# Model Architecture
# ============================================================================

class CrashSeverityNN(nn.Module):
    """Feedforward neural network for crash severity prediction."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_sizes: list[int] | None = None,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]

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

        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


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


def prepare_features(
    df: pd.DataFrame,
    fit_encoder: bool = True,
    label_encoder: LabelEncoder | None = None,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder, list[str]]:
    """Prepare features and target for training."""
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
    categorical_encoded = categorical_df.values
    feature_columns = list(categorical_df.columns)
    
    # Combine features
    X = np.hstack([numeric_data, categorical_encoded])
    
    # Encode target
    if fit_encoder:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df[TARGET])
    else:
        if label_encoder is None:
            raise ValueError("label_encoder required when fit_encoder=False")
        y = label_encoder.transform(df[TARGET])
    
    return X, y, label_encoder, feature_columns


# ============================================================================
# Baseline Model
# ============================================================================

def train_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
) -> dict:
    """Train a Random Forest baseline for comparison."""
    print("\n" + "=" * 60)
    print("BASELINE: Random Forest")
    print("=" * 60)
    
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    accuracy = rf.score(X_test, y_test)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Macro: {f1_macro:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    return {"accuracy": accuracy, "f1_macro": f1_macro, "model": rf}


# ============================================================================
# Training
# ============================================================================

def train_model(
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    patience: int = 10,
    use_class_weights: bool = True,
    run_baseline: bool = True,
    device: str | None = None,
) -> TrainingArtifacts:
    """Train the crash severity prediction model with best practices."""
    
    set_seed(RANDOM_SEED)
    
    # Device setup
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Load and prepare data
    print("\nLoading crash data...")
    crashes = get_traffic_crashes()
    
    print("Preparing features...")
    X, y, label_encoder, feature_columns = prepare_features(crashes)
    
    # Scale features BEFORE splitting to avoid data leakage
    # Note: In production, fit scaler only on training data
    scaler = StandardScaler()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    # Fit scaler on training data only
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Compute class weights
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    print(f"\nClass distribution in training set:")
    for cls, weight in zip(label_encoder.classes_, class_weights):
        count = np.sum(y_train == label_encoder.transform([cls])[0])
        print(f"  {cls}: {count:,} samples (weight: {weight:.2f})")
    
    # Run baseline comparison
    if run_baseline:
        baseline_results = train_baseline(
            X_train, y_train, X_test, y_test, label_encoder
        )
    
    # Convert to tensors (keep on CPU for DataLoader, move to device in training loop)
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)
    
    # Create dataloader (data stays on CPU, batches moved to GPU in training loop)
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),  # Faster CPU->GPU transfer
        num_workers=0,  # Set >0 for parallel data loading if needed
    )
    
    # Initialize model
    input_size = X_train.shape[1]
    num_classes = len(label_encoder.classes_)
    model = CrashSeverityNN(input_size, num_classes).to(device)
    
    print("\n" + "=" * 60)
    print("NEURAL NETWORK TRAINING")
    print("=" * 60)
    print(f"Model: {input_size} inputs -> {num_classes} classes")
    print(f"Architecture: {[128, 64, 32]} hidden layers")
    
    # Loss with class weights
    if use_class_weights:
        weight_tensor = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        print("Using class-weighted loss")
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop with early stopping
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
        "test_f1_macro": [],
    }
    
    best_f1 = 0.0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            # Move batch to device (GPU if available)
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        # Evaluation phase
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t)
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_predicted_np = test_predicted.cpu().numpy()
            y_test_np = y_test_t.cpu().numpy()
            
            test_acc = (test_predicted == y_test_t).sum().item() / len(y_test_t)
            test_f1 = f1_score(y_test_np, test_predicted_np, average="macro")
        
        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["test_f1_macro"].append(test_f1)
        
        # Early stopping check (based on F1-macro, not accuracy)
        if test_f1 > best_f1:
            best_f1 = test_f1
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
                f"Test F1: {test_f1:.4f}"
                + (" *" if patience_counter == 0 else "")
            )
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nRestored best model (F1-macro: {best_f1:.4f})")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        _, test_predicted = torch.max(test_outputs.data, 1)
        test_predicted_np = test_predicted.cpu().numpy()
        y_test_np = y_test_t.cpu().numpy()
    
    print("\nClassification Report:")
    print(classification_report(y_test_np, test_predicted_np, target_names=label_encoder.classes_))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_np, test_predicted_np))
    
    final_f1 = f1_score(y_test_np, test_predicted_np, average="macro")
    final_acc = (test_predicted_np == y_test_np).sum() / len(y_test_np)
    
    print(f"\nFinal Metrics:")
    print(f"  Accuracy:  {final_acc:.4f}")
    print(f"  F1-Macro:  {final_f1:.4f}")
    
    if run_baseline:
        print(f"\nComparison with Random Forest baseline:")
        print(f"  RF Accuracy: {baseline_results['accuracy']:.4f} | NN: {final_acc:.4f}")
        print(f"  RF F1-Macro: {baseline_results['f1_macro']:.4f} | NN: {final_f1:.4f}")
    
    return TrainingArtifacts(
        model=model,
        label_encoder=label_encoder,
        scaler=scaler,
        feature_columns=feature_columns,
        history=history,
        class_weights=class_weights,
        device=device,
    )


# ============================================================================
# Inference
# ============================================================================

def predict_severity(
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
) -> tuple[str, dict[str, float]]:
    """Predict severity for a single crash scenario.
    
    Returns:
        Tuple of (predicted_class, probability_dict)
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
    
    # Predict (move tensor to same device as model)
    artifacts.model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(artifacts.device)
        outputs = artifacts.model(X_tensor)
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        predicted_idx = np.argmax(probabilities)
    
    predicted_class = artifacts.label_encoder.inverse_transform([predicted_idx])[0]
    prob_dict = {
        cls: float(prob) 
        for cls, prob in zip(artifacts.label_encoder.classes_, probabilities)
    }
    
    return predicted_class, prob_dict


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Train model
    artifacts = train_model(
        epochs=100,
        patience=10,
        use_class_weights=True,
        run_baseline=True,
    )
    
    # Save model and artifacts
    torch.save({
        "model_state_dict": artifacts.model.state_dict(),
        "label_encoder_classes": artifacts.label_encoder.classes_,
        "scaler_mean": artifacts.scaler.mean_,
        "scaler_scale": artifacts.scaler.scale_,
        "feature_columns": artifacts.feature_columns,
        "class_weights": artifacts.class_weights,
    }, "crash_severity_model.pth")
    print("\nModel saved to crash_severity_model.pth")
    
    # Example prediction
    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTION")
    print("=" * 60)
    predicted, probs = predict_severity(
        artifacts,
        crash_hour=22,      # 10 PM
        crash_day=7,        # Saturday
        crash_month=12,     # December
        latitude=41.8781,
        longitude=-87.6298,
        posted_speed=35,
        weather="SNOW",
        lighting="DARKNESS",
    )
    print(f"Predicted severity: {predicted}")
    print("Probabilities:")
    for cls, prob in sorted(probs.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {prob:.2%}")