from data_preparation.helpers.csv_loaders import (
    get_crash_people,
    get_crash_vehicles,
    get_traffic_crashes,
)


def prepare_crash_data():
    crashes = get_traffic_crashes()
    crash_people = get_crash_people()
    crash_vehicles = get_crash_vehicles()
    print(crashes.head())
    print(crash_people.head())
    print(crash_vehicles.head())
