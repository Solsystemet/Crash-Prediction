from data_preparation.helpers.csv_loaders import get_traffic_crashes


def prepare_crash_data():
    crashes = get_traffic_crashes()

    print(crashes.head())
