import math

import numpy as np
import pandas as pd
import scipy
from torch.utils.data import Dataset


class DeepOdoDataset(Dataset):
    _SAMPLE_RATE_50HZ = 50

    _DATA_TIMESTAMP = 'DATA_TIMESTAMP'
    _PHONE_GYROSCOPE_X = 'PHONE_GYROSCOPE_X'
    _PHONE_GYROSCOPE_Y = 'PHONE_GYROSCOPE_Y'
    _PHONE_GYROSCOPE_Z = 'PHONE_GYROSCOPE_Z'
    _PHONE_ACCELEROMETER_X = 'PHONE_ACCELEROMETER_X'
    _PHONE_ACCELEROMETER_Y = 'PHONE_ACCELEROMETER_Y'
    _PHONE_ACCELEROMETER_Z = 'PHONE_ACCELEROMETER_Z'
    _PHONE_PRESSURE = 'PHONE_PRESSURE'
    _GROUND_TRUTH_VELOCITY_Y = 'GROUND_TRUTH_VELOCITY_Y'

    _DEEP_ODO_TRAIN_DATA_NAMES_LIST = [
        _DATA_TIMESTAMP,
        _PHONE_GYROSCOPE_X,
        _PHONE_GYROSCOPE_Y,
        _PHONE_GYROSCOPE_Z,
        _PHONE_ACCELEROMETER_X,
        _PHONE_ACCELEROMETER_Y,
        _PHONE_ACCELEROMETER_Z,
        _PHONE_PRESSURE,
        _GROUND_TRUTH_VELOCITY_Y
    ]

    def __init__(self, file_path, seq_len=60):
        self.seq_len = seq_len
        self.raw_data_panda = pd.read_csv(file_path, header=None, names=DeepOdoDataset._DEEP_ODO_TRAIN_DATA_NAMES_LIST)
        self.raw_data_numpy = self.raw_data_panda.to_numpy()
        self.raw_data_length = len(self.raw_data_numpy)
        self.raw_data_head_time = self.raw_data_numpy[0, 0]
        self.raw_data_tail_time = self.raw_data_numpy[self.raw_data_length - 1, 0]
        self.raw_50Hz_data_head_time = math.ceil(self.raw_data_head_time)
        self.raw_50Hz_data_tail_time = math.floor(self.raw_data_tail_time)
        self.raw_50Hz_data_duration = self.raw_50Hz_data_tail_time - self.raw_50Hz_data_head_time
        self.raw_50Hz_data_linspace_num = self.raw_50Hz_data_duration * DeepOdoDataset._SAMPLE_RATE_50HZ + 1
        self.raw_50Hz_data_time = np.linspace(
            self.raw_50Hz_data_head_time,
            self.raw_50Hz_data_tail_time,
            self.raw_50Hz_data_linspace_num
        )
        self.raw_50Hz_data = self.resample_train_data(self.raw_data_numpy, self.raw_50Hz_data_time)

        self.raw_50Hz_phone_sensor_data = self.raw_50Hz_data[1:self.raw_50Hz_data_linspace_num, 0:7]
        self.phone_sensor_data_sequence = self.raw_50Hz_phone_sensor_data.reshape((self.raw_50Hz_data_duration, 50, 7))

        self.ground_truth_data_index = np.arange(50, self.raw_50Hz_data_linspace_num, 50)
        self.ground_truth_data_sequence = self.raw_50Hz_data[self.ground_truth_data_index, 7]\
            .reshape((self.ground_truth_data_index.shape[0], 1))

    def __len__(self):
        return self.raw_50Hz_data_duration - self.seq_len + 1

    def __getitem__(self, idx):
        sequence = np.arange(idx, idx + self.seq_len)
        return self.phone_sensor_data_sequence[sequence, :, :], self.ground_truth_data_sequence[sequence, :]

    def resample_train_data(self, data, output_time):
        interp1d_time = data[:, 0]
        interp1d_data = data[:, 1:]
        func = scipy.interpolate.interp1d(interp1d_time, interp1d_data, axis=0)
        output_data = func(output_time)
        return output_data
