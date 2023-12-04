import math
import os

import numpy as np
import pandas as pd
import scipy
from torch.utils.data import Dataset

from datasets.deepodo_dataloader import load_chongqin_deepodo_raw_data


class DeepOdoChongQinDataset(Dataset):
    _SAMPLE_RATE_50HZ = 50

    def __init__(self, file_path_list, normalize_factors):
        self.dataset_len = len(file_path_list)
        self.input_sensor_data = []
        self.output_velocity_data = []
        for i in range(self.dataset_len):
            phone_sensor_data_sequence, ground_truth_data_sequence = DeepOdoChongQinDataset.load_deepodo_normalize_data(
                file_path_list[i], normalize_factors
            )
            self.input_sensor_data.append(phone_sensor_data_sequence)
            self.output_velocity_data.append(ground_truth_data_sequence)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        return self.input_sensor_data[idx], self.output_velocity_data[idx]

    @staticmethod
    def resample_deepodo_data(data, output_time):
        interp1d_time = data[:, 0]
        interp1d_data = data[:, 1:]
        func = scipy.interpolate.interp1d(interp1d_time, interp1d_data, axis=0)
        output_data = func(output_time)
        return output_data

    @staticmethod
    def load_deepodo_normalize_data(data_path, normalize_factors):
        raw_data = load_chongqin_deepodo_raw_data(data_path)
        raw_data_length = len(raw_data)
        raw_data_head_time = raw_data[0, 0]
        raw_data_tail_time = raw_data[raw_data_length - 1, 0]
        resampled_data_head_time = math.ceil(raw_data_head_time)
        resampled_data_tail_time = math.floor(raw_data_tail_time)
        resampled_data_duration = resampled_data_tail_time - resampled_data_head_time
        resampled_data_linear_space_num = resampled_data_duration * DeepOdoChongQinDataset._SAMPLE_RATE_50HZ + 1
        resampled_data_time = np.linspace(
            resampled_data_head_time,
            resampled_data_tail_time,
            resampled_data_linear_space_num
        )

        resampled_data = DeepOdoChongQinDataset.resample_deepodo_data(raw_data, resampled_data_time)

        resampled_phone_sensor_data_centered = resampled_data[1:, :7] - normalize_factors['mean']
        resampled_phone_sensor_data_standard = resampled_phone_sensor_data_centered / normalize_factors['std']

        phone_sensor_data_sequence = resampled_phone_sensor_data_standard.reshape(
            (resampled_data_duration, DeepOdoChongQinDataset._SAMPLE_RATE_50HZ, 7)
        )
        ground_truth_data_index = np.arange(
            DeepOdoChongQinDataset._SAMPLE_RATE_50HZ, resampled_data_linear_space_num, DeepOdoChongQinDataset._SAMPLE_RATE_50HZ
        )
        ground_truth_data_sequence = resampled_data[ground_truth_data_index, 7] \
            .reshape((ground_truth_data_index.shape[0], 1))

        return phone_sensor_data_sequence, ground_truth_data_sequence
