"""Unit tests for the WeatherStationsSchema."""

import pandas as pd
import pandera.errors
import pytest

from models.data_schemas.weather_stations import WeatherStationsSchema

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_valid_row(**overrides: object) -> dict[str, object]:
    """Return a dict representing one valid weather station row."""
    row: dict[str, object] = {
        "Station Name": "Foster Weather Station",
        "Measurement Timestamp": "01/01/2025 12:00:00 PM",
        "Air Temperature": 22.5,
        "Wet Bulb Temperature": 18.3,
        "Humidity": 55.0,
        "Rain Intensity": 0.0,
        "Interval Rain": 0.0,
        "Total Rain": 0.0,
        "Precipitation Type": 0.0,
        "Wind Direction": 180.0,
        "Wind Speed": 3.5,
        "Maximum Wind Speed": 5.0,
        "Barometric Pressure": 1013.25,
        "Solar Radiation": 500.0,
        "Heading": 0.0,
        "Battery Life": 12.5,
        "Measurement Timestamp Label": "01/01/2025 12:00 PM",
        "Measurement ID": "FosterWeatherStation202501011200",
    }
    row.update(overrides)
    return row


def _df_from_row(**overrides: object) -> pd.DataFrame:
    return pd.DataFrame([_make_valid_row(**overrides)])


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


class TestWeatherStationsSchemaValid:
    """Tests that valid data passes schema validation."""

    def test_minimal_valid_row(self) -> None:
        df = _df_from_row()
        validated = WeatherStationsSchema.validate(df)
        assert len(validated) == 1

    def test_nullable_fields_accept_none(self) -> None:
        df = _df_from_row(
            **{
                "Air Temperature": None,
                "Wet Bulb Temperature": None,
                "Humidity": None,
                "Rain Intensity": None,
                "Interval Rain": None,
                "Total Rain": None,
                "Precipitation Type": None,
                "Wind Direction": None,
                "Wind Speed": None,
                "Maximum Wind Speed": None,
                "Barometric Pressure": None,
                "Solar Radiation": None,
                "Heading": None,
                "Battery Life": None,
            }
        )
        validated = WeatherStationsSchema.validate(df)
        assert len(validated) == 1

    def test_multiple_rows(self) -> None:
        rows = [
            _make_valid_row(**{"Measurement ID": "id_1"}),
            _make_valid_row(**{"Measurement ID": "id_2"}),
        ]
        df = pd.DataFrame(rows)
        validated = WeatherStationsSchema.validate(df)
        assert len(validated) == 2

    def test_all_precipitation_types(self) -> None:
        for ptype in (0, 40, 60, 70):
            df = _df_from_row(
                **{
                    "Precipitation Type": float(ptype),
                    "Measurement ID": f"id_precip_{ptype}",
                }
            )
            validated = WeatherStationsSchema.validate(df)
            assert len(validated) == 1

    def test_wind_direction_boundaries(self) -> None:
        for direction in (0.0, 180.0, 360.0):
            df = _df_from_row(**{"Wind Direction": direction})
            validated = WeatherStationsSchema.validate(df)
            assert len(validated) == 1

    def test_humidity_boundaries(self) -> None:
        for h in (0.0, 50.0, 100.0):
            df = _df_from_row(**{"Humidity": h})
            validated = WeatherStationsSchema.validate(df)
            assert len(validated) == 1

    def test_negative_air_temperature(self) -> None:
        """Sub-zero temperatures are valid."""
        df = _df_from_row(**{"Air Temperature": -15.0})
        validated = WeatherStationsSchema.validate(df)
        assert len(validated) == 1

    def test_negative_solar_radiation(self) -> None:
        """Sensor anomalies can produce small negative values."""
        df = _df_from_row(**{"Solar Radiation": -2.0})
        validated = WeatherStationsSchema.validate(df)
        assert len(validated) == 1

    def test_csv_sample(self) -> None:
        """Validate a sample from the real CSV file."""
        df = pd.read_csv("data/beach_weather_stations_automated_sensors.csv", nrows=500)
        validated = WeatherStationsSchema.validate(df)
        assert len(validated) == 500


# ---------------------------------------------------------------------------
# Negative tests
# ---------------------------------------------------------------------------


class TestWeatherStationsSchemaInvalid:
    """Tests that invalid data is rejected."""

    def test_duplicate_measurement_id(self) -> None:
        rows = [
            _make_valid_row(**{"Measurement ID": "dup"}),
            _make_valid_row(**{"Measurement ID": "dup"}),
        ]
        df = pd.DataFrame(rows)
        with pytest.raises(pandera.errors.SchemaError):
            _ = WeatherStationsSchema.validate(df)

    def test_humidity_over_100(self) -> None:
        df = _df_from_row(**{"Humidity": 101.0})
        with pytest.raises(pandera.errors.SchemaError):
            _ = WeatherStationsSchema.validate(df)

    def test_humidity_negative(self) -> None:
        df = _df_from_row(**{"Humidity": -1.0})
        with pytest.raises(pandera.errors.SchemaError):
            _ = WeatherStationsSchema.validate(df)

    def test_rain_intensity_negative(self) -> None:
        df = _df_from_row(**{"Rain Intensity": -0.5})
        with pytest.raises(pandera.errors.SchemaError):
            _ = WeatherStationsSchema.validate(df)

    def test_wind_direction_over_360(self) -> None:
        df = _df_from_row(**{"Wind Direction": 361.0})
        with pytest.raises(pandera.errors.SchemaError):
            _ = WeatherStationsSchema.validate(df)

    def test_wind_speed_negative(self) -> None:
        df = _df_from_row(**{"Wind Speed": -1.0})
        with pytest.raises(pandera.errors.SchemaError):
            _ = WeatherStationsSchema.validate(df)

    def test_invalid_precipitation_type(self) -> None:
        df = _df_from_row(**{"Precipitation Type": 99.0})
        with pytest.raises(pandera.errors.SchemaError):
            _ = WeatherStationsSchema.validate(df)

    def test_negative_total_rain(self) -> None:
        df = _df_from_row(**{"Total Rain": -1.0})
        with pytest.raises(pandera.errors.SchemaError):
            _ = WeatherStationsSchema.validate(df)

    def test_heading_over_360(self) -> None:
        df = _df_from_row(**{"Heading": 400.0})
        with pytest.raises(pandera.errors.SchemaError):
            _ = WeatherStationsSchema.validate(df)
