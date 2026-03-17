from pandera.typing.pandas import DataFrame

from models.data_schemas.full.traffic_crashes import TrafficCrashesSchema


def remove_outliers(data: DataFrame[TrafficCrashesSchema]):
    lat_min, lat_max = 41.6, 42.1
    lon_min, lon_max = -87.9, -87.5

    df_clean = (
        data[
            (data[TrafficCrashesSchema.LATITUDE] > lat_min)
            & (data[TrafficCrashesSchema.LATITUDE] < lat_max)
            & (data[TrafficCrashesSchema.LONGITUDE] > lon_min)
            & (data[TrafficCrashesSchema.LONGITUDE] < lon_max)
        ]
        .dropna(subset=[TrafficCrashesSchema.LATITUDE, TrafficCrashesSchema.LONGITUDE])
        .copy()
    )

    return df_clean
