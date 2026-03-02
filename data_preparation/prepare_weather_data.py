from data_preparation.helpers.csv_loaders import get_weather_stations


def prepare_weather_data():
    weather_stations = get_weather_stations()

    print(weather_stations.head())
