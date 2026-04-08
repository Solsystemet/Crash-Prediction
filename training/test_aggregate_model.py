"""Test script for the aggregate injury prediction model.

This script allows you to test the trained model with specific input values.

Run with:
    python -m training.test_aggregate_model

Or with custom values:
    python -m training.test_aggregate_model --hour 14 --day 4 --month 7 --temp 25.0 --humidity 60
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from data_preparation.aggregate_data import AggregateDataResult
from training.train_aggregate import load_aggregate_model
from training.device_utils import print_device_info
from training.losses import is_count_loss


MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "trained"
DEFAULT_MODEL_PATH = MODELS_DIR / "aggregate_injury_mlp.pt"
DEFAULT_DATA_PATH = MODELS_DIR / "aggregate_injury_mlp_data.pt"
LOGTRANSFORM_MODEL_PATH = MODELS_DIR / "aggregate_injury_mlp_logtransform.pt"
LOGTRANSFORM_DATA_PATH = MODELS_DIR / "aggregate_injury_mlp_logtransform_data.pt"
POISSON_MODEL_PATH = MODELS_DIR / "aggregate_injury_mlp_poisson.pt"
POISSON_DATA_PATH = MODELS_DIR / "aggregate_injury_mlp_poisson_data.pt"
NEGBIN_MODEL_PATH = MODELS_DIR / "aggregate_injury_mlp_negbin.pt"
NEGBIN_DATA_PATH = MODELS_DIR / "aggregate_injury_mlp_negbin_data.pt"


def predict_injuries(
    model_path: Path,
    data_path: Path,
    hour_of_day: int,
    day_of_week: int,
    month: int,
    year: int,
    air_temperature: float,
    humidity: float,
    rain_intensity: float,
    total_rain: float,
    precipitation_type: float,
    device: str = "cpu",
) -> dict[str, float]:
    """Predict injury counts for given conditions.

    Args:
        model_path: Path to the trained model.
        data_path: Path to the data preprocessing info.
        hour_of_day: Hour of day (0-23).
        day_of_week: Day of week (0=Monday, 6=Sunday).
        month: Month (1-12).
        year: Year.
        air_temperature: Air temperature in Celsius.
        humidity: Humidity percentage (0-100).
        rain_intensity: Rain intensity.
        total_rain: Total rain amount.
        precipitation_type: Precipitation type code (0, 5, 40, 60, 70).
        device: Device to run inference on.

    Returns:
        Dictionary mapping injury type to predicted count.
    """
    # Load model and data info
    model = load_aggregate_model(model_path, device=device)
    data_result = AggregateDataResult.load(data_path)

    # Build feature vector in the correct order
    feature_values = {
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "month": month,
        "year": year,
        "Air Temperature": air_temperature,
        "Humidity": humidity,
        "Rain Intensity": rain_intensity,
        "Total Rain": total_rain,
        "Precipitation Type": precipitation_type,
    }

    # Create feature tensor in the correct order
    features = []
    for col in data_result.feature_columns:
        if col in feature_values:
            features.append(feature_values[col])
        else:
            print(f"Warning: Unknown feature column '{col}', using 0.0")
            features.append(0.0)

    # Convert to tensor and normalize
    feature_tensor = torch.tensor(features, dtype=torch.float32)
    feature_tensor = (
        feature_tensor - data_result.feature_means
    ) / data_result.feature_stds
    feature_tensor = feature_tensor.unsqueeze(0).to(device)  # Add batch dimension

    # Run inference
    model.eval()
    with torch.no_grad():
        predictions = model(feature_tensor)

    # Convert predictions to numpy
    predictions = predictions.squeeze(0).cpu().numpy()

    # Transform predictions back to original scale
    if is_count_loss(data_result.loss_function):
        # Model outputs log-rates, convert to rates
        predictions = np.exp(predictions)
    elif data_result.log_transform_targets:
        # Inverse of log(1 + x) is exp(x) - 1
        predictions = np.expm1(predictions)

    # Convert to dictionary
    results = {}
    for i, col in enumerate(data_result.target_columns):
        # Clamp to non-negative (injury counts can't be negative)
        results[col] = max(0.0, float(predictions[i]))

    return results


def main() -> None:
    """Main function to test the model with specific values."""
    parser = argparse.ArgumentParser(
        description="Test aggregate injury prediction model with specific values"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to trained model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to data preprocessing info (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--hour",
        type=int,
        default=14,
        help="Hour of day (0-23, default: 14)",
    )
    parser.add_argument(
        "--day",
        type=int,
        default=4,
        help="Day of week (0=Monday, 6=Sunday, default: 4=Friday)",
    )
    parser.add_argument(
        "--month",
        type=int,
        default=7,
        help="Month (1-12, default: 7=July)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2024,
        help="Year (default: 2024)",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=24.0,
        help="Air temperature in Celsius (default: 24.0)",
    )
    parser.add_argument(
        "--humidity",
        type=float,
        default=50.0,
        help="Humidity percentage (default: 50.0)",
    )
    parser.add_argument(
        "--rain-intensity",
        type=float,
        default=0.0,
        help="Rain intensity (default: 0.0)",
    )
    parser.add_argument(
        "--total-rain",
        type=float,
        default=0.0,
        help="Total rain amount (default: 0.0)",
    )
    parser.add_argument(
        "--precip-type",
        type=float,
        default=0.0,
        help="Precipitation type code (0=none, 5=?, 40=?, 60=rain, 70=snow, default: 0)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode to test multiple scenarios",
    )
    parser.add_argument(
        "--scenarios",
        action="store_true",
        help="Run scenario comparison across different conditions",
    )
    parser.add_argument(
        "--log-transform",
        action="store_true",
        help="Use the log-transformed model (aggregate_injury_mlp_logtransform.pt)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["mse", "poisson", "negbin"],
        default=None,
        help="Use model trained with specific loss (auto-selects model path)",
    )
    args = parser.parse_args()

    # Update paths based on model type
    if args.loss == "poisson":
        if args.model == DEFAULT_MODEL_PATH:
            args.model = POISSON_MODEL_PATH
        if args.data == DEFAULT_DATA_PATH:
            args.data = POISSON_DATA_PATH
    elif args.loss == "negbin":
        if args.model == DEFAULT_MODEL_PATH:
            args.model = NEGBIN_MODEL_PATH
        if args.data == DEFAULT_DATA_PATH:
            args.data = NEGBIN_DATA_PATH
    elif args.log_transform:
        if args.model == DEFAULT_MODEL_PATH:
            args.model = LOGTRANSFORM_MODEL_PATH
        if args.data == DEFAULT_DATA_PATH:
            args.data = LOGTRANSFORM_DATA_PATH

    # Check if model files exist
    if not args.model.exists():
        print(f"Error: Model file not found: {args.model}")
        print("Please train the model first with: python -m training.main_aggregate")
        return

    if not args.data.exists():
        print(f"Error: Data file not found: {args.data}")
        print("Please train the model first with: python -m training.main_aggregate")
        return

    # Print device info
    print("=" * 60)
    print("Aggregate Injury Prediction Model Test")
    print("=" * 60)
    device = print_device_info()
    print()

    if args.interactive:
        run_interactive(args.model, args.data, device)
    elif args.scenarios:
        run_scenario_comparison(args.model, args.data, device)
    else:
        run_single_prediction(args, device)


def run_single_prediction(args: argparse.Namespace, device: str) -> None:
    """Run a single prediction with command line arguments."""
    print("Input conditions:")
    print(f"  Hour of day:        {args.hour}")
    print(
        f"  Day of week:        {args.day} ({['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][args.day]})"
    )
    print(f"  Month:              {args.month}")
    print(f"  Year:               {args.year}")
    print(f"  Air Temperature:    {args.temp}°C")
    print(f"  Humidity:           {args.humidity}%")
    print(f"  Rain Intensity:     {args.rain_intensity}")
    print(f"  Total Rain:         {args.total_rain}")
    print(f"  Precipitation Type: {args.precip_type}")
    print()

    # Run prediction
    predictions = predict_injuries(
        model_path=args.model,
        data_path=args.data,
        hour_of_day=args.hour,
        day_of_week=args.day,
        month=args.month,
        year=args.year,
        air_temperature=args.temp,
        humidity=args.humidity,
        rain_intensity=args.rain_intensity,
        total_rain=args.total_rain,
        precipitation_type=args.precip_type,
        device=device,
    )

    print("Predicted hourly injury counts:")
    print("-" * 40)
    total = 0.0
    for injury_type, count in predictions.items():
        # Format the injury type name nicely
        name = injury_type.replace("INJURIES_", "").replace("_", " ").title()
        print(f"  {name:30s} {count:8.2f}")
        total += count
    print("-" * 40)
    print(f"  {'Total':30s} {total:8.2f}")


def run_interactive(model_path: Path, data_path: Path, device: str) -> None:
    """Run interactive mode to test multiple scenarios."""
    print("Interactive mode - enter values to test predictions")
    print("Press Ctrl+C to exit")
    print()

    # Pre-load model and data for faster repeated predictions
    model = load_aggregate_model(model_path, device=device)
    data_result = AggregateDataResult.load(data_path)

    while True:
        try:
            print("-" * 60)
            hour = int(input("Hour of day (0-23) [14]: ") or "14")
            day = int(input("Day of week (0=Mon, 6=Sun) [4]: ") or "4")
            month = int(input("Month (1-12) [7]: ") or "7")
            year = int(input("Year [2024]: ") or "2024")
            temp = float(input("Temperature (°C) [24.0]: ") or "24.0")
            humidity = float(input("Humidity (%) [50.0]: ") or "50.0")
            rain = float(input("Rain intensity [0.0]: ") or "0.0")
            total_rain = float(input("Total rain [0.0]: ") or "0.0")
            precip = float(input("Precipitation type [0.0]: ") or "0.0")

            # Build feature tensor
            feature_values = {
                "hour_of_day": hour,
                "day_of_week": day,
                "month": month,
                "year": year,
                "Air Temperature": temp,
                "Humidity": humidity,
                "Rain Intensity": rain,
                "Total Rain": total_rain,
                "Precipitation Type": precip,
            }

            features = [
                feature_values.get(col, 0.0) for col in data_result.feature_columns
            ]
            feature_tensor = torch.tensor(features, dtype=torch.float32)
            feature_tensor = (
                feature_tensor - data_result.feature_means
            ) / data_result.feature_stds
            feature_tensor = feature_tensor.unsqueeze(0).to(device)

            model.eval()
            with torch.no_grad():
                predictions = model(feature_tensor)

            predictions = predictions.squeeze(0).cpu().numpy()

            # Transform predictions back to original scale
            if is_count_loss(data_result.loss_function):
                # Model outputs log-rates, convert to rates
                predictions = np.exp(predictions)
            elif data_result.log_transform_targets:
                # Inverse of log(1 + x) is exp(x) - 1
                predictions = np.expm1(predictions)

            print()
            print("Predicted hourly injury counts:")
            total = 0.0
            for i, col in enumerate(data_result.target_columns):
                name = col.replace("INJURIES_", "").replace("_", " ").title()
                count = max(0.0, float(predictions[i]))
                print(f"  {name:30s} {count:8.2f}")
                total += count
            print(f"  {'Total':30s} {total:8.2f}")
            print()

        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
            break
        except ValueError as e:
            print(f"Invalid input: {e}")
            continue


def run_scenario_comparison(model_path: Path, data_path: Path, device: str) -> None:
    """Compare predictions across different scenarios."""
    print("\nScenario Comparison")
    print("=" * 80)

    scenarios = [
        {
            "name": "Weekday afternoon, clear weather",
            "hour": 14,
            "day": 2,
            "month": 6,
            "year": 2024,
            "temp": 24.0,
            "humidity": 50.0,
            "rain": 0.0,
            "total_rain": 0.0,
            "precip": 0.0,
        },
        {
            "name": "Friday night, rainy",
            "hour": 22,
            "day": 4,
            "month": 6,
            "year": 2024,
            "temp": 18.0,
            "humidity": 85.0,
            "rain": 2.0,
            "total_rain": 1.5,
            "precip": 60.0,
        },
        {
            "name": "Saturday night (party time)",
            "hour": 23,
            "day": 5,
            "month": 6,
            "year": 2024,
            "temp": 21.0,
            "humidity": 60.0,
            "rain": 0.0,
            "total_rain": 0.0,
            "precip": 0.0,
        },
        {
            "name": "Winter morning rush hour",
            "hour": 8,
            "day": 1,
            "month": 1,
            "year": 2024,
            "temp": -4.0,
            "humidity": 70.0,
            "rain": 0.5,
            "total_rain": 0.2,
            "precip": 70.0,
        },
        {
            "name": "Sunday morning (low traffic)",
            "hour": 6,
            "day": 6,
            "month": 6,
            "year": 2024,
            "temp": 18.0,
            "humidity": 55.0,
            "rain": 0.0,
            "total_rain": 0.0,
            "precip": 0.0,
        },
    ]

    for scenario in scenarios:
        predictions = predict_injuries(
            model_path=model_path,
            data_path=data_path,
            hour_of_day=scenario["hour"],
            day_of_week=scenario["day"],
            month=scenario["month"],
            year=scenario["year"],
            air_temperature=scenario["temp"],
            humidity=scenario["humidity"],
            rain_intensity=scenario["rain"],
            total_rain=scenario["total_rain"],
            precipitation_type=scenario["precip"],
            device=device,
        )

        total = sum(predictions.values())
        print(f"\n{scenario['name']}:")
        print(
            f"  Conditions: {scenario['hour']}:00, Day {scenario['day']}, Month {scenario['month']}"
        )
        print(
            f"  Weather: {scenario['temp']}°C, {scenario['humidity']}% humidity, rain={scenario['rain']}"
        )
        print(f"  Predicted total injuries: {total:.2f}")


if __name__ == "__main__":
    main()
