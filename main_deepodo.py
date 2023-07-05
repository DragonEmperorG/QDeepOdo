import math
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.deepodo_dataset import DeepOdoDataset
from graphs.models.deepodo_model import DeepOdoModel
from test_deepodo import test
from train_deepodo import train


def main():
    datasets_root_folder_path = "E:\\DoctorRelated\\20230410重庆VDR数据采集"

    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    datasets_collector_data_time_folder_name = "2023_04_10"
    datasets_collector_data_time_folder_path = os.path.join(
        datasets_root_folder_path,
        datasets_collector_data_time_folder_name
    )

    datasets_preprocess_reorganized_folder_name = "Reorganized"
    datasets_preprocess_reorganized_folder_path = os.path.join(
        datasets_collector_data_time_folder_path,
        datasets_preprocess_reorganized_folder_name
    )

    datasets_collector_track_folder_name = "0016"
    datasets_collector_track_folder_path = os.path.join(
        datasets_preprocess_reorganized_folder_path,
        datasets_collector_track_folder_name
    )

    dataset_collector_phone_folder_name = "HUAWEI_Mate30"
    dataset_collector_phone_folder_path = os.path.join(
        datasets_collector_track_folder_path,
        dataset_collector_phone_folder_name
    )

    dataset_deepodo_folder_name = "DATASET_DEEPODO"
    dataset_deepodo_folder_path = os.path.join(
        dataset_collector_phone_folder_path,
        dataset_deepodo_folder_name
    )

    dataset_deepodo_file_name = "DeepOdoTrainData.csv"
    dataset_deepodo_file_path = os.path.join(
        dataset_deepodo_folder_path,
        dataset_deepodo_file_name
    )

    batch_size = 5
    train_dataset = DeepOdoDataset(dataset_deepodo_file_path, True, 459)
    test_dataset = DeepOdoDataset(dataset_deepodo_file_path, False)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset)

    deepodo_model = DeepOdoModel()

    file_path = os.path.join(os.path.abspath('.'), 'experiments', 'checkpoints', 'model_deepodo_wang.p')
    min_loss = -1
    if os.path.isfile(file_path):
        min_loss = 7.158216
        deepodo_model.load_state_dict(torch.load(file_path))

    loss_fn = nn.MSELoss(reduction="mean")

    learn_rate = 0.0001
    optimizer = torch.optim.Adam(deepodo_model.parameters(), lr=learn_rate)

    epochs = 1000
    file_name_train_head_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))

    writer = SummaryWriter()

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        min_loss = train(train_dataloader, deepodo_model, loss_fn, optimizer, t, epochs, file_name_train_head_time,
                         min_loss, writer, device)

    test(test_dataloader, deepodo_model, loss_fn, device)

    writer.flush()
    print("Done!")


if __name__ == '__main__':
    main()
