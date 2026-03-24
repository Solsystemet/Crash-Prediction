#!/usr/bin/env python
"""Load trained model and predict crash counts.

Usage:
    uv run training/predict_crashes.py --datetime "2026-03-23 22:00" --temp 5 --humidity 80 --rain 2.5 --wind 15
    uv run training/predict_crashes.py --interactive
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data_preparation.time_features import TimeFeatureConfig, add_time_features
from data_preparation.encoders import EncoderRegistry


class CrashCountMLP(nn.Module):
    """MLP for predicting hourly crash counts (regression)."""

    def __init__(
        self,
        num_features: int,
        hidden_sizes: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_size = num_features
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class CrashPredictor:
    """Wrapper for loading and using trained crash prediction model."""

    def __init__(self, model_path: Path | str) -> None:
        """Load model from checkpoint file.
        
        Args:
            model_path: Path to the .pt checkpoint file.
        """
        self.model_path = Path(model_path)
        self.checkpoint = torch.load(
            self.model_path, map_location="cpu", weights_only=False
        )
        
        # Reconstruct model
        self.model = CrashCountMLP(
            num_features=self.checkpoint["num_features"],
            hidden_sizes=self.checkpoint["hidden_sizes"],
            dropout=self.checkpoint["dropout"],
        )
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()
        
        # Reconstruct time config
        self.time_config = TimeFeatureConfig(**self.checkpoint["time_config"])
        
        # Load encoder registry for scaling
        self.encoder_registry = EncoderRegistry.from_dict(
            self.checkpoint["encoder_registry"]
        )
        
        # Load additional feature flags
        self.include_holidays = self.checkpoint.get("include_holidays", False)
        self.include_conditions = self.checkpoint.get("include_conditions", False)
        self.condition_lookup = self.checkpoint.get("condition_lookup", None)
        
        # Feature columns in correct order
        self.feature_columns = self.checkpoint["feature_columns"]
        
        # Training statistics
        self.train_mean = self.checkpoint["train_mean"]
        self.test_metrics = self.checkpoint.get("test_metrics", {})

    def predict(
        self,
        dt: datetime | str,
        temperature: float,
        humidity: float,
        rain_intensity: float = 0.0,
        wind_speed: float = 10.0,
        # Optional lag values (defaults to current values)
        temp_lag1h: float | None = None,
        temp_lag2h: float | None = None,
        temp_lag3h: float | None = None,
        humidity_lag1h: float | None = None,
        humidity_lag2h: float | None = None,
        humidity_lag3h: float | None = None,
        rain_lag1h: float | None = None,
        rain_lag2h: float | None = None,
        rain_lag3h: float | None = None,
        wind_lag1h: float | None = None,
        wind_lag2h: float | None = None,
        wind_lag3h: float | None = None,
    ) -> float:
        """Predict crash count for given conditions.
        
        Args:
            dt: Datetime for prediction (string or datetime object).
            temperature: Current temperature in °C.
            humidity: Current humidity percentage.
            rain_intensity: Rain intensity (default 0).
            wind_speed: Wind speed (default 10).
            *_lag*h: Weather values from 1/2/3 hours ago. If not provided,
                uses current values as approximation.
        
        Returns:
            Predicted number of crashes for that hour.
        """
        if isinstance(dt, str):
            dt = datetime.strptime(dt, "%Y-%m-%d %H:%M")
        
        # Create DataFrame with all required columns
        df = pd.DataFrame({
            "hour_timestamp": [pd.Timestamp(dt)],
            "Air Temperature": [temperature],
            "Humidity": [humidity],
            "Rain Intensity": [rain_intensity],
            "Wind Speed": [wind_speed],
            # Lag features - use current value if not provided
            "Air Temperature_lag_1h": [temp_lag1h if temp_lag1h is not None else temperature],
            "Air Temperature_lag_2h": [temp_lag2h if temp_lag2h is not None else temperature],
            "Air Temperature_lag_3h": [temp_lag3h if temp_lag3h is not None else temperature],
            "Humidity_lag_1h": [humidity_lag1h if humidity_lag1h is not None else humidity],
            "Humidity_lag_2h": [humidity_lag2h if humidity_lag2h is not None else humidity],
            "Humidity_lag_3h": [humidity_lag3h if humidity_lag3h is not None else humidity],
            "Rain Intensity_lag_1h": [rain_lag1h if rain_lag1h is not None else rain_intensity],
            "Rain Intensity_lag_2h": [rain_lag2h if rain_lag2h is not None else rain_intensity],
            "Rain Intensity_lag_3h": [rain_lag3h if rain_lag3h is not None else rain_intensity],
            "Wind Speed_lag_1h": [wind_lag1h if wind_lag1h is not None else wind_speed],
            "Wind Speed_lag_2h": [wind_lag2h if wind_lag2h is not None else wind_speed],
            "Wind Speed_lag_3h": [wind_lag3h if wind_lag3h is not None else wind_speed],
        })
        
        # Add time features based on config
        df = add_time_features(df, timestamp_col="hour_timestamp", config=self.time_config)
        
        # Add holiday feature if enabled
        if self.include_holidays:
            from data_preparation.aggregate_hourly import US_HOLIDAYS
            month_day = (dt.month, dt.day)
            df["is_holiday"] = 1 if month_day in US_HOLIDAYS else 0
        
        # Add condition features if enabled
        if self.include_conditions and self.condition_lookup is not None:
            hour = dt.hour
            if hour in self.condition_lookup:
                df["pct_darkness"] = self.condition_lookup[hour]["pct_darkness"]
                df["pct_wet_road"] = self.condition_lookup[hour]["pct_wet_road"]
                df["pct_bad_weather"] = self.condition_lookup[hour]["pct_bad_weather"]
            else:
                # Fallback to average if hour not found
                df["pct_darkness"] = 0.5
                df["pct_wet_road"] = 0.1
                df["pct_bad_weather"] = 0.1
        
        # Select features in correct order
        features_df = df[self.feature_columns].copy()
        
        # Apply scaling using saved numerical scaler
        if self.encoder_registry.numerical_scaler is not None:
            scaler = self.encoder_registry.numerical_scaler
            # The scaler expects columns in the same order it was fit on
            scaler_cols = scaler.column_names
            # Scale only the columns that were in the original scaler
            cols_to_scale = [c for c in scaler_cols if c in features_df.columns]
            if cols_to_scale:
                scaled_values = scaler.transform(features_df[cols_to_scale])
                for i, col in enumerate(cols_to_scale):
                    features_df[col] = scaled_values[:, i]
        
        # Convert to tensor and predict
        features_tensor = torch.tensor(features_df.values, dtype=torch.float32)
        
        with torch.no_grad():
            prediction = self.model(features_tensor).item()
        
        # Crash count can't be negative
        return max(0.0, prediction)

    def predict_risk_level(
        self,
        dt: datetime | str,
        temperature: float,
        humidity: float,
        rain_intensity: float = 0.0,
        wind_speed: float = 10.0,
    ) -> tuple[float, str]:
        """Predict crash count and risk level.
        
        Returns:
            Tuple of (predicted_count, risk_level).
            Risk levels: "LOW", "NORMAL", "HIGH", "VERY HIGH"
        """
        prediction = self.predict(dt, temperature, humidity, rain_intensity, wind_speed)
        
        if prediction < self.train_mean * 0.5:
            risk = "LOW"
        elif prediction < self.train_mean * 1.2:
            risk = "NORMAL"
        elif prediction < self.train_mean * 2.0:
            risk = "HIGH"
        else:
            risk = "VERY HIGH"
        
        return prediction, risk


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict crash counts using trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prediction
  uv run training/predict_crashes.py --datetime "2026-03-23 22:00" --temp 5 --humidity 80

  # With rain and wind
  uv run training/predict_crashes.py --datetime "2026-03-23 22:00" --temp 5 --humidity 80 --rain 2.5 --wind 20

  # Interactive mode
  uv run training/predict_crashes.py --interactive
        """,
    )
    
    parser.add_argument(
        "--model", 
        type=Path, 
        default=Path("trained_models/crash_count_mlp.pt"),
        help="Path to trained model file",
    )
    parser.add_argument("--datetime", type=str, help="Datetime (YYYY-MM-DD HH:MM)")
    parser.add_argument("--temp", type=float, help="Temperature (°C)")
    parser.add_argument("--humidity", type=float, help="Humidity (%%)")
    parser.add_argument("--rain", type=float, default=0, help="Rain intensity (default: 0)")
    parser.add_argument("--wind", type=float, default=10, help="Wind speed (default: 10)")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    # Load model
    if not args.model.exists():
        print(f"Error: Model not found at {args.model}")
        print("Train a model first with: uv run training/train_with_time_features.py --preset full")
        return
    
    print(f"Loading model from {args.model}...")
    predictor = CrashPredictor(args.model)
    
    print(f"Model loaded successfully!")
    print(f"  Features: {len(predictor.feature_columns)}")
    print(f"  Training mean: {predictor.train_mean:.2f} crashes/hour")
    if predictor.test_metrics:
        print(f"  Test MAE: {predictor.test_metrics.get('mae', 'N/A'):.2f}")
    
    if args.interactive:
        print("\n" + "=" * 60)
        print("Interactive Prediction Mode")
        print("Type 'quit' or 'q' to exit, 'help' for guidance")
        print("=" * 60)
        
        while True:
            try:
                print()
                dt_str = input("Datetime (YYYY-MM-DD HH:MM): ").strip()
                
                if dt_str.lower() in ("quit", "q", "exit"):
                    print("Goodbye!")
                    break
                
                if dt_str.lower() == "help":
                    print("\nEnter values for prediction:")
                    print("  - Datetime: e.g., 2026-03-23 22:00")
                    print("  - Temperature: in Celsius")
                    print("  - Humidity: percentage (0-100)")
                    print("  - Rain intensity: 0 for no rain, higher for heavier rain")
                    print("  - Wind speed: in km/h or m/s (consistent with training data)")
                    continue
                
                temp = float(input("Temperature (°C): "))
                humidity = float(input("Humidity (%): "))
                rain_str = input("Rain intensity [0]: ").strip()
                rain = float(rain_str) if rain_str else 0.0
                wind_str = input("Wind speed [10]: ").strip()
                wind = float(wind_str) if wind_str else 10.0
                
                prediction, risk = predictor.predict_risk_level(
                    dt_str, temp, humidity, rain, wind
                )
                
                print()
                print("-" * 40)
                print(f"Predicted crashes: {prediction:.1f}")
                print(f"Risk level: {risk}")
                print(f"(Training average: {predictor.train_mean:.1f})")
                print("-" * 40)
                
                if risk == "HIGH":
                    print("⚠️  Consider deploying additional resources")
                elif risk == "VERY HIGH":
                    print("🚨 HIGH ALERT - Significant crash risk expected")
                elif risk == "LOW":
                    print("✓ Below-average crash risk")
                    
            except ValueError as e:
                print(f"Invalid input: {e}")
                print("Please enter numeric values for temperature, humidity, etc.")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
    
    else:
        # Single prediction mode
        if not args.datetime or args.temp is None or args.humidity is None:
            parser.error("--datetime, --temp, and --humidity required (or use --interactive)")
        
        prediction, risk = predictor.predict_risk_level(
            args.datetime, args.temp, args.humidity, args.rain, args.wind
        )
        
        print()
        print("=" * 60)
        print("PREDICTION")
        print("=" * 60)
        print(f"  Datetime: {args.datetime}")
        print(f"  Weather: {args.temp}°C, {args.humidity}% humidity, rain={args.rain}, wind={args.wind}")
        print()
        print(f"  Predicted crashes: {prediction:.1f}")
        print(f"  Risk level: {risk}")
        print(f"  (Training average: {predictor.train_mean:.1f})")
        print("=" * 60)


if __name__ == "__main__":
    main()
