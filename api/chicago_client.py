"""Chicago SODA API client for fetching real crash data.

This module provides functions to fetch recent crash data from the
City of Chicago's open data portal using the SODA API.

API Documentation: https://dev.socrata.com/docs/endpoints.html
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Chicago Open Data Portal base URL
BASE_URL = "https://data.cityofchicago.org/resource"

# Dataset IDs
CRASHES_DATASET = "85ca-t3if"
VEHICLES_DATASET = "68nd-jvt3"
PEOPLE_DATASET = "u6pd-qa9d"
WEATHER_DATASET = "k7hf-8y75"

# Default request timeout (seconds)
REQUEST_TIMEOUT = 30

# Maximum records per request (SODA API default limit is 1000)
DEFAULT_LIMIT = 1000
MAX_LIMIT = 50000


class ChicagoAPIError(Exception):
    """Exception raised for Chicago API errors."""

    pass


def _make_request(
    dataset_id: str,
    params: dict[str, Any] | None = None,
    timeout: int = REQUEST_TIMEOUT,
) -> list[dict[str, Any]]:
    """Make a request to the SODA API.

    Args:
        dataset_id: The dataset identifier (e.g., '85ca-t3if')
        params: Query parameters for the request
        timeout: Request timeout in seconds

    Returns:
        List of records from the API

    Raises:
        ChicagoAPIError: If the request fails
    """
    url = f"{BASE_URL}/{dataset_id}.json"

    try:
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        raise ChicagoAPIError(f"Request timed out after {timeout} seconds")
    except requests.exceptions.RequestException as e:
        raise ChicagoAPIError(f"API request failed: {e}")
    except ValueError as e:
        raise ChicagoAPIError(f"Failed to parse JSON response: {e}")


def _format_date(dt: datetime) -> str:
    """Format datetime for SODA API query (ISO 8601 format)."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def get_recent_crashes(
    days: int = 7,
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Fetch recent crash records from the Chicago API.

    Args:
        days: Number of days back to fetch crashes
        limit: Maximum number of records to return
        offset: Number of records to skip (for pagination)

    Returns:
        List of crash records
    """
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Build SoQL query
    # We need crashes that have injury data (most_severe_injury is not null)
    where_clause = (
        f"crash_date >= '{_format_date(start_date)}' AND most_severe_injury IS NOT NULL"
    )

    params = {
        "$where": where_clause,
        "$limit": min(limit, MAX_LIMIT),
        "$offset": offset,
        "$order": "crash_date DESC",
    }

    logger.info(
        f"Fetching crashes from last {days} days (limit={limit}, offset={offset})"
    )

    crashes = _make_request(CRASHES_DATASET, params)
    logger.info(f"Retrieved {len(crashes)} crash records")

    return crashes


def get_all_recent_crashes(
    days: int = 7, max_records: int = 5000
) -> list[dict[str, Any]]:
    """Fetch all recent crashes with pagination.

    Args:
        days: Number of days back to fetch crashes
        max_records: Maximum total records to fetch

    Returns:
        List of all crash records within the time period
    """
    all_crashes = []
    offset = 0

    while len(all_crashes) < max_records:
        batch = get_recent_crashes(
            days=days,
            limit=min(DEFAULT_LIMIT, max_records - len(all_crashes)),
            offset=offset,
        )

        if not batch:
            break

        all_crashes.extend(batch)
        offset += len(batch)

        if len(batch) < DEFAULT_LIMIT:
            break

    return all_crashes


def get_vehicles_for_crashes(crash_ids: list[str]) -> list[dict[str, Any]]:
    """Fetch vehicle records for given crash IDs.

    Args:
        crash_ids: List of crash_record_id values

    Returns:
        List of vehicle records
    """
    if not crash_ids:
        return []

    # SODA API has limits on query length, so we batch the requests
    batch_size = 100
    all_vehicles = []

    for i in range(0, len(crash_ids), batch_size):
        batch_ids = crash_ids[i : i + batch_size]

        # Build IN clause for SoQL
        ids_str = "', '".join(batch_ids)
        where_clause = f"crash_record_id IN ('{ids_str}')"

        params = {
            "$where": where_clause,
            "$limit": MAX_LIMIT,
        }

        try:
            vehicles = _make_request(VEHICLES_DATASET, params)
            all_vehicles.extend(vehicles)
        except ChicagoAPIError as e:
            logger.warning(f"Failed to fetch vehicles for batch: {e}")

    logger.info(
        f"Retrieved {len(all_vehicles)} vehicle records for {len(crash_ids)} crashes"
    )
    return all_vehicles


def get_people_for_crashes(crash_ids: list[str]) -> list[dict[str, Any]]:
    """Fetch people records for given crash IDs.

    Args:
        crash_ids: List of crash_record_id values

    Returns:
        List of people records
    """
    if not crash_ids:
        return []

    batch_size = 100
    all_people = []

    for i in range(0, len(crash_ids), batch_size):
        batch_ids = crash_ids[i : i + batch_size]

        ids_str = "', '".join(batch_ids)
        where_clause = f"crash_record_id IN ('{ids_str}')"

        params = {
            "$where": where_clause,
            "$limit": MAX_LIMIT,
        }

        try:
            people = _make_request(PEOPLE_DATASET, params)
            all_people.extend(people)
        except ChicagoAPIError as e:
            logger.warning(f"Failed to fetch people for batch: {e}")

    logger.info(
        f"Retrieved {len(all_people)} people records for {len(crash_ids)} crashes"
    )
    return all_people


def get_weather_data(
    start_date: datetime,
    end_date: datetime,
    limit: int = DEFAULT_LIMIT,
) -> list[dict[str, Any]]:
    """Fetch weather station data for a date range.

    Args:
        start_date: Start of date range
        end_date: End of date range
        limit: Maximum number of records

    Returns:
        List of weather records
    """
    where_clause = (
        f"measurement_timestamp >= '{_format_date(start_date)}' "
        f"AND measurement_timestamp <= '{_format_date(end_date)}'"
    )

    params = {
        "$where": where_clause,
        "$limit": min(limit, MAX_LIMIT),
        "$order": "measurement_timestamp DESC",
    }

    weather = _make_request(WEATHER_DATASET, params)
    logger.info(f"Retrieved {len(weather)} weather records")

    return weather


def fetch_crash_data_for_accuracy(
    days: int = 7, max_crashes: int = 500
) -> dict[str, Any]:
    """Fetch all data needed for accuracy calculation.

    This is a convenience function that fetches crashes, vehicles, and people
    data in one call.

    Args:
        days: Number of days back to fetch
        max_crashes: Maximum number of crashes to process

    Returns:
        Dictionary with 'crashes', 'vehicles', 'people', and 'weather' keys
    """
    logger.info(
        f"Fetching crash data for accuracy calculation (days={days}, max={max_crashes})"
    )

    # Fetch crashes
    crashes = get_all_recent_crashes(days=days, max_records=max_crashes)

    if not crashes:
        return {"crashes": [], "vehicles": [], "people": [], "weather": []}

    # Extract crash IDs
    crash_ids = [c["crash_record_id"] for c in crashes if "crash_record_id" in c]

    # Fetch related data
    vehicles = get_vehicles_for_crashes(crash_ids)
    people = get_people_for_crashes(crash_ids)

    # Fetch weather for the time period
    if crashes:
        # Find date range from crashes
        dates = []
        for c in crashes:
            if "crash_date" in c:
                try:
                    dt = datetime.fromisoformat(c["crash_date"].replace("Z", "+00:00"))
                    dates.append(dt)
                except (ValueError, TypeError):
                    pass

        if dates:
            start_date = min(dates) - timedelta(hours=1)
            end_date = max(dates) + timedelta(hours=1)
            weather = get_weather_data(start_date, end_date)
        else:
            weather = []
    else:
        weather = []

    return {
        "crashes": crashes,
        "vehicles": vehicles,
        "people": people,
        "weather": weather,
    }
