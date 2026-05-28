# Crash Prediction

A machine learning system for predicting traffic crash severity using Chicago Traffic Crashes data. Includes model training pipelines, a FastAPI backend, and a React frontend.

## Table of Contents

- [Data](#data)
- [Installation](#installation)
- [Backend API](#backend-api)
- [Frontend](#frontend)
- [Training Models](#training-models)
- [Model Comparison](#model-comparison)
- [Project Structure](#project-structure)

## Data

Download the following datasets and place them in the `data/` directory with the specified filenames:

| Dataset                    | Download Link                                                                                                                                               | Filename                                       |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| Traffic Crashes            | [Traffic Crashes - Crashes](https://data.cityofchicago.org/Transportation/Traffic-Crashes-Crashes/85ca-t3if/about_data)                                     | `traffic_crashes.csv`                          |
| Beach Weather Stations     | [Beach Weather Stations - Automated Sensors](https://data.cityofchicago.org/Parks-Recreation/Beach-Weather-Stations-Automated-Sensors/k7hf-8y75/about_data) | `beach_weather_stations_automated_sensors.csv` |
| Traffic Crashes - People   | [Traffic Crashes - People](https://data.cityofchicago.org/Transportation/Traffic-Crashes-People/u6pd-qa9d/about_data)                                       | `traffic-crashes-people.csv`                   |
| Traffic Crashes - Vehicles | [Traffic Crashes - Vehicles](https://data.cityofchicago.org/Transportation/Traffic-Crashes-Vehicles/68nd-jvt3/about_data)                                   | `traffic_crashes_vehicles.csv`                 |

## Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Backend API

The FastAPI backend serves predictions and model information.

### Running the API

```bash
# Development mode with auto-reload
uvicorn api.main:app --reload --port 8000

# Or run directly
python -m api.main
```

### API Endpoints

| Endpoint            | Method | Description                |
| ------------------- | ------ | -------------------------- |
| `/health`           | GET    | Health check               |
| `/models`           | GET    | List available models      |
| `/predict`          | POST   | Make severity prediction   |
| `/feature-options`  | GET    | Get feature value options  |
| `/accuracy`         | GET    | Get model accuracy metrics |
| `/roc-data`         | GET    | Get ROC curve data         |
| `/model-comparison` | GET    | Compare all models         |

### Example Prediction Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "lgbm",
    "features": {
      "POSTED_SPEED_LIMIT": 30,
      "TRAFFIC_CONTROL_DEVICE": "TRAFFIC SIGNAL",
      "WEATHER_CONDITION": "CLEAR",
      "LIGHTING_CONDITION": "DAYLIGHT",
      "ROADWAY_SURFACE_COND": "DRY",
      "CRASH_HOUR": 14,
      "CRASH_DAY_OF_WEEK": 3
    }
  }'
```

## Frontend

The React frontend is located in `../crash-prediction-frontend/`.

### Running the Frontend

```bash
cd ../crash-prediction-frontend

# Install dependencies
bun install  # or npm install

# Run development server
bun dev  # or npm run dev
```

The frontend will be available at `http://localhost:5173`.

## Training Models

### Quick Start: Train All Models

```bash
# Train all models with temporal split (recommended)
python -m training.train_all_models --temporal-split --feature-filter none

# Train with sampling for faster testing
python -m training.train_all_models --temporal-split --sample 50000
```

### Individual Model Training

#### Random Forest Baseline

```bash
python -m training.main_simple_rf --temporal-split --feature-filter none
```

#### Gradient Boosting Models (LightGBM, XGBoost, CatBoost)

```bash
# Train all boosting models with hyperparameter tuning
python -m training.tune_boosting --model all --temporal-split --feature-filter none

# Train specific model
python -m training.tune_boosting --model lgbm --temporal-split --n-trials 30
python -m training.tune_boosting --model xgb --temporal-split --n-trials 30
python -m training.tune_boosting --model catboost --temporal-split --n-trials 30
```

#### TabNet (Deep Learning)

```bash
python -m training.tabnet.train --temporal-split
```

### Training Options

| Flag                    | Description                                 |
| ----------------------- | ------------------------------------------- |
| `--temporal-split`      | Use chronological 80/20 split (recommended) |
| `--feature-filter none` | Use all features                            |
| `--sample N`            | Train on N samples for faster testing       |
| `--n-trials N`          | Number of Optuna hyperparameter trials      |
| `--skip-boosting`       | Skip slow boosting model tuning             |

## Model Comparison

### Compare All Trained Models

```bash
# Compare models on temporal test set
python -m training.compare_all_models --temporal-split --feature-filter none

# With sampling for faster comparison
python -m training.compare_all_models --temporal-split --sample 50000
```

### Output

The comparison generates:

- **Console output**: Accuracy, F1 scores, per-class recall
- **ROC curves**: `models/plots/roc_curves_all_models.png`
- **CSV results**: `models/plots/model_comparison_*.csv`

### Expected Results (Temporal Split)

| Model         | Accuracy | F1 Macro | SEVERE Recall |
| ------------- | -------- | -------- | ------------- |
| XGBoost       | 78.1%    | 0.579    | 64.8%         |
| LightGBM      | 77.8%    | 0.579    | 66.1%         |
| Random Forest | 82.1%    | 0.496    | 11.9%         |
| CatBoost      | 67.9%    | 0.437    | 62.6%         |

## Project Structure

```
Crash-Prediction/
├── api/                    # FastAPI backend
│   ├── main.py            # API entry point
│   ├── models.py          # Pydantic schemas
│   └── prediction.py      # Prediction logic
├── data/                   # Dataset files (not in git)
├── data_preparation/       # Data preprocessing
│   ├── resampling.py      # Train/test splits
│   └── get_prepared_data.py
├── models/
│   ├── trained/           # Saved model files
│   └── plots/             # Generated visualizations
├── training/              # Model training scripts
│   ├── main_simple_rf.py  # Random Forest
│   ├── tune_boosting.py   # LightGBM/XGBoost/CatBoost
│   ├── compare_all_models.py
│   └── train_all_models.py
└── tests/                 # Unit tests
```

## License

MIT
