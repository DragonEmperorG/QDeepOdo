import math
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.deepodo_dataset import DeepOdoDataset
from models.deepodo_model import DeepOdoModel

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


def train(dataloader, model, loss_fn1, optimizer1, t1, epoch, time1, min_loss1, writer1):
    size = len(dataloader.dataset)
    model.train()

    train_min_loss = min_loss1

    for batch, (sensor_data, ground_truth) in enumerate(dataloader):
        sensor_data, ground_truth = sensor_data.to(device), ground_truth.to(device)
        input_float = sensor_data.float()
        output_float = ground_truth.float()

        # Compute prediction error
        pred = model(input_float)
        loss = loss_fn1(pred, output_float)

        # Backpropagation
        if loss < 1.8:
            loss.backward()
            optimizer1.step()
            optimizer1.zero_grad()

        writer.add_scalar("Loss/train", loss, t1)
        loss, current = loss.item(), (batch + 1) * len(sensor_data)

        is_save_model = False
        if train_min_loss == -1:
            train_min_loss = loss
            is_save_model = True
        else:
            if loss < train_min_loss:
                train_min_loss = loss
                is_save_model = True

        if is_save_model:
            file_name_loss = math.floor(loss * 1e6)
            file_name = "model_deepodo_wang_train_schedule_{}_epoch_{}_{}_batch_{}_{}_Loss_{}.p".format(time1,
                                                                                                        t1, epoch, current,
                                                                                                        size,
                                                                                                        file_name_loss)
            file_path1 = os.path.join(os.path.abspath('.'), 'experiments', 'checkpoints', file_name)
            torch.save(model.state_dict(), file_path1)
            file_path2 = os.path.join(os.path.abspath('.'), 'experiments', 'checkpoints', 'model_deepodo_wang.p')
            torch.save(model.state_dict(), file_path2)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    return train_min_loss


if __name__ == '__main__':
    datasets_root_folder_path = "E:\\DoctorRelated\\20230410重庆VDR数据采集"

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

    datasets_collector_track_folder_name = "0008"
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
    train_dataset = DeepOdoDataset(dataset_deepodo_file_path, 120)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    deepodo_model = DeepOdoModel()
    file_path = os.path.join(os.path.abspath('.'), 'experiments', 'checkpoints', 'model_deepodo_wang.p')
    min_loss = -1
    if os.path.isfile(file_path):
        min_loss = 1.275825
        deepodo_model.load_state_dict(torch.load(file_path))

    loss_fn = nn.MSELoss(reduction="mean")

    learn_rate = 0.001
    optimizer = torch.optim.Adam(deepodo_model.parameters(), lr=learn_rate, weight_decay=0.8)

    epochs = 1000
    file_name_train_head_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))

    writer = SummaryWriter()

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        min_loss = train(train_dataloader, deepodo_model, loss_fn, optimizer, t, epochs, file_name_train_head_time,
                         min_loss, writer)

    writer.flush()
    print("Done!")
