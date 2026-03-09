# training/severity_predictor.py
"""Neural network to predict crash severity from crash data features."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_preparation.helpers.csv_loaders import get_traffic_crashes


# Features to use for prediction
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

TARGET = "MOST_SEVERE_INJURY"


class CrashSeverityNN(nn.Module):
    """Simple feedforward neural network for crash severity prediction."""

    def __init__(self, input_size: int, num_classes: int, hidden_sizes: list[int] = None):
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
                nn.Dropout(0.3),
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Prepare features and target for training."""
    # Drop rows with missing target
    df = df.dropna(subset=[TARGET])
    
    # Handle numeric features
    numeric_data = df[NUMERIC_FEATURES].fillna(0).values
    
    # Handle categorical features with one-hot encoding
    categorical_encoded = pd.get_dummies(
        df[CATEGORICAL_FEATURES].fillna("UNKNOWN"),
        drop_first=True
    ).values
    
    # Combine features
    X = np.hstack([numeric_data, categorical_encoded])
    
    # Encode target
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET])
    
    return X, y, label_encoder


def train_model(
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 0.001,
) -> tuple[CrashSeverityNN, LabelEncoder, dict]:
    """Train the crash severity prediction model."""
    
    print("Loading crash data...")
    crashes = get_traffic_crashes()
    
    print("Preparing features...")
    X, y, label_encoder = prepare_features(crashes)
    
    # Scale numeric features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)
    
    # Create dataloaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    input_size = X_train.shape[1]
    num_classes = len(label_encoder.classes_)
    model = CrashSeverityNN(input_size, num_classes)
    
    print(f"Model: {input_size} inputs -> {num_classes} classes")
    print(f"Classes: {label_encoder.classes_}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    history = {"train_loss": [], "train_acc": [], "test_acc": []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t)
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_acc = (test_predicted == y_test_t).sum().item() / len(y_test_t)
        
        train_acc = correct / total
        avg_loss = total_loss / len(train_loader)
        
        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - "
                  f"Train Acc: {train_acc:.4f} - Test Acc: {test_acc:.4f}")
    
    return model, label_encoder, history


def predict_severity(
    model: CrashSeverityNN,
    label_encoder: LabelEncoder,
    crash_hour: int,
    crash_day: int,
    crash_month: int,
    latitude: float,
    longitude: float,
    weather: str = "CLEAR",
    lighting: str = "DAYLIGHT",
) -> str:
    """Predict severity for a single crash."""
    # This would need the same feature preparation as training
    # Simplified example - in practice, use the same preprocessing pipeline
    model.eval()
    # ... feature preparation would go here
    raise NotImplementedError("Use full preprocessing pipeline for predictions")


if __name__ == "__main__":
    model, encoder, history = train_model(epochs=50)
    print(f"\nFinal Test Accuracy: {history['test_acc'][-1]:.4f}")
    
    # Save model
    torch.save(model.state_dict(), "crash_severity_model.pth")
    print("Model saved to crash_severity_model.pth")