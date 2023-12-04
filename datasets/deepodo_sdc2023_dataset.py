from torch.utils.data import Dataset

from datasets.deepodo_dataloader import load_sdc2023_deepodo_normalize_data


class DeepOdoSdcDataset(Dataset):
    _SAMPLE_RATE_50HZ = 50

    def __init__(self, file_path_list, normalize_factors):
        self.dataset_len = len(file_path_list)
        self.dataset_trace_list = []
        for i in range(self.dataset_len):
            dataset_trace = load_sdc2023_deepodo_normalize_data(
                file_path_list[i], normalize_factors, DeepOdoSdcDataset._SAMPLE_RATE_50HZ
            )
            self.dataset_trace_list.append(dataset_trace)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        return self.dataset_trace_list[idx]

