from data_preparation.helpers.csv_loaders import get_traffic_crashes
from k_means import k_means, group_by_k_means


def splice():
    trafficData = get_traffic_crashes()
    centroids = k_means(trafficData, 10, 100)
    grouped_data = group_by_k_means(trafficData, centroids)
    return grouped_data
