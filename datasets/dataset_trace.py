import math
import os

import numpy as np
import pandas as pd
import scipy
from torch.utils.data import Dataset


class DatasetTrace:

    def __init__(self, folder_path, phone_sensor_data_sequence, ground_truth_data_sequence):
        self.folder_path = folder_path
        self.phone_sensor_data_sequence = phone_sensor_data_sequence
        self.ground_truth_data_sequence = ground_truth_data_sequence




