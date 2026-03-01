from data_preparation.helpers.csv_loaders import get_traffic_tracker


def prepare_traffic_tracker():
    traffic_tracker = get_traffic_tracker()

    print(traffic_tracker.head())
