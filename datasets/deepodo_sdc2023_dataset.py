import math

from torch.utils.data import Dataset

from datasets.dataset_trace import DatasetTrace
from datasets.deepodo_dataloader import load_sdc2023_deepodo_normalize_data


class DeepOdoSdcDataset(Dataset):
    _SAMPLE_RATE_50HZ = 50

    def __init__(self, type, file_path_list, normalize_factors, window_time_duration, window_time_hop):
        self.dataset_trace_list = []
        for i in range(len(file_path_list)):
            file_path = file_path_list[i]
            dataset_trace = load_sdc2023_deepodo_normalize_data(
                file_path, normalize_factors, DeepOdoSdcDataset._SAMPLE_RATE_50HZ
            )

            if type == 'result':
                self.dataset_trace_list.append(dataset_trace)
            else:
                dataset_subtrace_counts = math.floor((dataset_trace.global_end_index - window_time_duration) / window_time_hop)
                for j in range(dataset_subtrace_counts):
                    dataset_subtrace_start_index = window_time_hop * j
                    dataset_subtrace_end_index = dataset_subtrace_start_index + window_time_duration - 1
                    dataset_subtrace_section_index = range(dataset_subtrace_start_index, (dataset_subtrace_end_index+1))
                    dataset_subtrace_phone_sensor_data_sequence = dataset_trace.phone_sensor_data_sequence[dataset_subtrace_section_index, :, :]
                    dataset_subtrace_ground_truth_data_sequence = dataset_trace.ground_truth_data_sequence[dataset_subtrace_section_index, :]
                    dataset_subtrace = DatasetTrace(
                        file_path,
                        dataset_subtrace_start_index,
                        dataset_subtrace_end_index,
                        dataset_subtrace_phone_sensor_data_sequence,
                        dataset_subtrace_ground_truth_data_sequence
                    )
                    self.dataset_trace_list.append(dataset_subtrace)

    def __len__(self):
        return len(self.dataset_trace_list)

    def __getitem__(self, idx):
        return self.dataset_trace_list[idx]

