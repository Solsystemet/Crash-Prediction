from pathlib import Path

import pandas as pd

from data_preparation.data_prepper import prepare_data

PATH_TO_PROCESSED_TRAINING_DATA = "../data/proccessed_trainraining_data"
PATH_TO_PROCESSED_TESTING_DATA = "../data/proccessed_testing_data"


def get_processed_training_data():
    processed_training_data = Path(PATH_TO_PROCESSED_TRAINING_DATA)
    return _get_processed_data(processed_training_data)


def get_processed_testing_data():
    processed_testing_data = Path(PATH_TO_PROCESSED_TESTING_DATA)
    return _get_processed_data(processed_testing_data)


def _get_processed_data(path: Path):
    if not path.exists():
        prepare_data()


#    return pd.read_csv(path)
