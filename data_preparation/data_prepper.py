from data_preparation.prepare_crash_data import prepare_crash_data
from data_preparation.prepare_traffic_tracker import prepare_traffic_tracker
from data_preparation.prepare_weather_data import prepare_weather_data


def prepare_data():
    prepare_crash_data()
    prepare_traffic_tracker()
    prepare_weather_data()
