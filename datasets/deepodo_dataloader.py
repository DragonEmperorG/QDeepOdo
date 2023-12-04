import math
import os

import numpy as np
import pandas as pd
import scipy

from datasets.dataset_trace import DatasetTrace
from utils.logs.log_utils import get_logger

DEEPODO_DATALOADER_CONFIG_TRAIN_LABEL = 'TRAIN'
DEEPODO_DATALOADER_CONFIG_TEST_LABEL = 'TEST'


def load_chongqin_dataset(root_folder_path, dataset_classification_config):
    train_list = []
    train_preprocessor = []
    test_list = []

    for i in range(len(dataset_classification_config)):
        datasets_collector_data_time_folder_name = dataset_classification_config[i][1]
        datasets_collector_data_time_folder_path = os.path.join(
            root_folder_path,
            datasets_collector_data_time_folder_name
        )

        datasets_preprocess_reorganized_folder_name = "Reorganized"
        datasets_preprocess_reorganized_folder_path = os.path.join(
            datasets_collector_data_time_folder_path,
            datasets_preprocess_reorganized_folder_name
        )

        datasets_collector_track_folder_name = dataset_classification_config[i][2]
        datasets_collector_track_folder_path = os.path.join(
            datasets_preprocess_reorganized_folder_path,
            datasets_collector_track_folder_name
        )

        dataset_collector_phone_folder_name = dataset_classification_config[i][3]
        dataset_collector_phone_folder_path = os.path.join(
            datasets_collector_track_folder_path,
            dataset_collector_phone_folder_name
        )

        if dataset_classification_config[i][0] == DEEPODO_DATALOADER_CONFIG_TRAIN_LABEL:
            train_list.append(dataset_collector_phone_folder_path)
            raw_data = load_chongqin_deepodo_raw_data(dataset_collector_phone_folder_path)
            train_preprocessor.append(raw_data[:, 1:8])
        elif dataset_classification_config[i][0] == DEEPODO_DATALOADER_CONFIG_TEST_LABEL:
            test_list.append(dataset_collector_phone_folder_path)

    train_preprocessor_concatenate = np.concatenate(train_preprocessor)
    train_preprocessor_mean = np.mean(train_preprocessor_concatenate, axis=0)
    train_preprocessor_std = np.std(train_preprocessor_concatenate, axis=0)
    normalize_factors = {
        'mean': train_preprocessor_mean,
        'std': train_preprocessor_std,
    }

    return train_list, test_list, normalize_factors


def load_chongqin_deepodo_raw_data(folder_path):
    dataset_deepodo_folder_name = "DATASET_DEEPODO"
    dataset_deepodo_folder_path = os.path.join(
        folder_path,
        dataset_deepodo_folder_name
    )

    dataset_deepodo_numpy_file_name = "DeepOdoTrainData.npy"
    dataset_deepodo_numpy_file_path = os.path.join(
        dataset_deepodo_folder_path,
        dataset_deepodo_numpy_file_name
    )
    loaded_deepodo_raw_data = []
    if os.path.isfile(dataset_deepodo_numpy_file_path):
        loaded_deepodo_raw_data = np.load(dataset_deepodo_numpy_file_path)
    else:
        dataset_deepodo_raw_file_name = "DeepOdoTrainData.csv"
        dataset_deepodo_raw_file_path = os.path.join(
            dataset_deepodo_folder_path,
            dataset_deepodo_raw_file_name
        )
        if os.path.isfile(dataset_deepodo_raw_file_path):
            raw_data_panda = pd.read_csv(dataset_deepodo_raw_file_path, header=None,
                                         names=DeepOdoDataset._DEEP_ODO_TRAIN_DATA_NAMES_LIST)
            loaded_deepodo_raw_data = raw_data_panda.to_numpy()
            np.save(dataset_deepodo_numpy_file_path, loaded_deepodo_raw_data)
        else:
            print(f"Not have file {dataset_deepodo_raw_file_path=}")

    return loaded_deepodo_raw_data


def load_sdc2023_dataset(root_folder_path, dataset_classification_config):
    train_list = []
    test_list = []
    train_preprocessor = []
    for i in range(len(dataset_classification_config)):
        datasets_level2_drive_id_folder_name = dataset_classification_config[i][1]
        datasets_level2_drive_id_folder_path = os.path.join(
            root_folder_path,
            datasets_level2_drive_id_folder_name
        )

        datasets_level3_phone_name_folder_name = dataset_classification_config[i][2]
        datasets_level3_phone_name_folder_path = os.path.join(
            datasets_level2_drive_id_folder_path,
            datasets_level3_phone_name_folder_name
        )

        if dataset_classification_config[i][0] == DEEPODO_DATALOADER_CONFIG_TRAIN_LABEL:
            train_list.append(datasets_level3_phone_name_folder_path)
            raw_data = load_sdc2023_deepodo_raw_data(datasets_level3_phone_name_folder_path)
            train_preprocessor.append(raw_data[:, 4:])
        elif dataset_classification_config[i][0] == DEEPODO_DATALOADER_CONFIG_TEST_LABEL:
            test_list.append(datasets_level3_phone_name_folder_path)

    train_preprocessor_concatenate = np.concatenate(train_preprocessor)
    train_preprocessor_mean = np.mean(train_preprocessor_concatenate, axis=0)
    train_preprocessor_std = np.std(train_preprocessor_concatenate, axis=0)
    normalize_factors = {
        'mean': train_preprocessor_mean,
        'std': train_preprocessor_std,
    }

    return train_list, test_list, normalize_factors


def load_sdc2023_deepodo_raw_data(folder_path):
    logger_load = get_logger()
    recompute_flag = False

    dataset_deepodo_folder_name = "dataset_deepodo"
    dataset_deepodo_folder_path = os.path.join(
        folder_path,
        dataset_deepodo_folder_name
    )

    if not os.path.isdir(dataset_deepodo_folder_path):
        os.makedirs(dataset_deepodo_folder_path)

    dataset_deepodo_numpy_file_name = "train_data_deepodo.npy"
    dataset_deepodo_numpy_file_path = os.path.join(
        dataset_deepodo_folder_path,
        dataset_deepodo_numpy_file_name
    )

    loaded_deepodo_raw_data = []
    if (not recompute_flag) & os.path.isfile(dataset_deepodo_numpy_file_path):
        loaded_deepodo_raw_data = np.load(dataset_deepodo_numpy_file_path)
    else:
        dataset_deepodo_raw_file_name = "interp_gt_velocity_imu.csv"
        dataset_deepodo_raw_file_path = os.path.join(
            folder_path,
            dataset_deepodo_raw_file_name
        )
        if os.path.isfile(dataset_deepodo_raw_file_path):
            _DATA_UNIX_TIME_MILLIS = 'DATA_UNIX_TIME_MILLIS'
            _DATA_ZERO_OCLOCK_TIME = 'DATA_ZERO_OCLOCK_TIME'
            _DATA_ELAPSE_REALTIME_NANOS = 'DATA_ELAPSE_REALTIME_NANOS'
            _GROUND_TRUTH_VELOCITY_FORWARD = 'GROUND_TRUTH_VELOCITY_FORWARD'
            _PHONE_GYROSCOPE_X = 'PHONE_GYROSCOPE_X'
            _PHONE_GYROSCOPE_Y = 'PHONE_GYROSCOPE_Y'
            _PHONE_GYROSCOPE_Z = 'PHONE_GYROSCOPE_Z'
            _PHONE_ACCELEROMETER_X = 'PHONE_ACCELEROMETER_X'
            _PHONE_ACCELEROMETER_Y = 'PHONE_ACCELEROMETER_Y'
            _PHONE_ACCELEROMETER_Z = 'PHONE_ACCELEROMETER_Z'
            _PHONE_MAGNETOMETER_X = 'PHONE_MAGNETOMETER_X'
            _PHONE_MAGNETOMETER_Y = 'PHONE_MAGNETOMETER_Y'
            _PHONE_MAGNETOMETER_Z = 'PHONE_MAGNETOMETER_Z'

            _DEEP_ODO_TRAIN_DATA_NAMES_LIST = [
                _DATA_UNIX_TIME_MILLIS,
                _DATA_ZERO_OCLOCK_TIME,
                _DATA_ELAPSE_REALTIME_NANOS,
                _GROUND_TRUTH_VELOCITY_FORWARD,
                _PHONE_GYROSCOPE_X,
                _PHONE_GYROSCOPE_Y,
                _PHONE_GYROSCOPE_Z,
                _PHONE_ACCELEROMETER_X,
                _PHONE_ACCELEROMETER_Y,
                _PHONE_ACCELEROMETER_Z,
                _PHONE_MAGNETOMETER_X,
                _PHONE_MAGNETOMETER_Y,
                _PHONE_MAGNETOMETER_Z,
            ]
            raw_data_panda = pd.read_csv(dataset_deepodo_raw_file_path, names=_DEEP_ODO_TRAIN_DATA_NAMES_LIST)
            loaded_deepodo_raw_data = raw_data_panda.to_numpy()
            np.save(dataset_deepodo_numpy_file_path, loaded_deepodo_raw_data)
            logger_load.info("Save file {}", dataset_deepodo_numpy_file_path)
        else:
            logger_load.error("Not have file {}", dataset_deepodo_raw_file_path)

    return loaded_deepodo_raw_data


def load_sdc2023_deepodo_normalize_data(data_path, normalize_factors, sample_rate):
    raw_data = load_sdc2023_deepodo_raw_data(data_path)
    raw_data_length = len(raw_data)
    raw_data_head_time = raw_data[0, 0]
    raw_data_tail_time = raw_data[raw_data_length - 1, 0]
    resampled_data_head_time = math.ceil(raw_data_head_time)
    resampled_data_tail_time = math.floor(raw_data_tail_time)
    resampled_data_duration = resampled_data_tail_time - resampled_data_head_time
    resampled_data_linear_space_num = resampled_data_duration * sample_rate + 1
    resampled_data_time = np.linspace(
        resampled_data_head_time,
        resampled_data_tail_time,
        resampled_data_linear_space_num
    )

    resampled_data = resample_sdc2023_deepodo_data(raw_data, resampled_data_time)

    resampled_phone_sensor_data_centered = resampled_data[1:, 1:] - normalize_factors['mean']
    resampled_phone_sensor_data_standard = resampled_phone_sensor_data_centered / normalize_factors['std']

    resampled_phone_6axis_imu_data_standard = resampled_phone_sensor_data_standard[:, :6]

    phone_sensor_data_sequence = resampled_phone_6axis_imu_data_standard.reshape(
        (resampled_data_duration, sample_rate, 6)
    )
    ground_truth_data_index = np.arange(
        sample_rate, resampled_data_linear_space_num, sample_rate
    )
    ground_truth_data_sequence = resampled_data[ground_truth_data_index, 0] \
        .reshape((ground_truth_data_index.shape[0], 1))

    dataset_trace = DatasetTrace(data_path, phone_sensor_data_sequence, ground_truth_data_sequence)

    return dataset_trace


def resample_sdc2023_deepodo_data(data, output_time):
    interp1d_time = data[:, 0]
    interp1d_data = data[:, 3:]
    func = scipy.interpolate.interp1d(interp1d_time, interp1d_data, axis=0)
    output_data = func(output_time)
    return output_data
