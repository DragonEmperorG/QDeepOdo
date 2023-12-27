import math
import os

import numpy as np
import torch

from utils.logs.log_utils import get_logger


def train(dataloader, model, loss_fn1, optimizer1, t1, epoch, time1, min_loss1, writer1, device):
    logger_train = get_logger()
    size = len(dataloader.dataset)
    model.train()
    optimizer1.zero_grad()
    train_min_loss = min_loss1

    epoch_loss_num = 0
    epoch_loss_sum = 0
    batch_data_len = len(dataloader)
    for batch, batch_data in enumerate(dataloader):
        batch_loss_num = 0
        batch_loss_sum = 0
        batch_counter = 0
        for dataset_trace in batch_data:
            sensor_data_numpy = dataset_trace.phone_sensor_data_sequence
            ground_truth_numpy = dataset_trace.ground_truth_data_sequence
            sensor_data_tensor, ground_truth_tensor = torch.tensor(sensor_data_numpy), torch.tensor(ground_truth_numpy)
            sensor_data, ground_truth = sensor_data_tensor.to(device), ground_truth_tensor.to(device)
            input_float = sensor_data.float()
            output_float = ground_truth.float()

            # input_float_len = len(input_float)
            # random_sample0 = int(np.random.randint(0, input_float_len - 60))
            # random_sample1 = random_sample0 + 60
            # random_sample_input_float = input_float[random_sample0:random_sample1, :]
            # random_sample_output_float = output_float[random_sample0:random_sample1, :]

            random_sample_input_float = input_float
            random_sample_output_float = output_float

            # Compute prediction error
            pred = model(device, random_sample_input_float)
            sequence_loss = loss_fn1(pred, random_sample_output_float)

            sequence_len = len(ground_truth_numpy)
            batch_loss_num = batch_loss_num + sequence_len
            batch_loss_sum = batch_loss_sum + sequence_loss * sequence_len

        # Backpropagation
        epoch_loss_num = epoch_loss_num + batch_loss_num
        epoch_loss_sum = epoch_loss_sum + batch_loss_sum

        batch_loss_avg = batch_loss_sum / batch_loss_num
        batch_loss_avg.backward()
        optimizer1.step()
        optimizer1.zero_grad()

        logger_train.info(f"batch loss: {batch_loss_avg:>7f}  [{(batch + 1):>2d}/{batch_data_len:>2d}]")

        writer1.add_scalar("Loss/train", batch_loss_avg, t1)
        batch_loss_sum, current = batch_loss_sum.item(), (batch + 1) * batch_data_len

        is_save_model = False
        if train_min_loss == -1:
            train_min_loss = batch_loss_sum
            is_save_model = True
        else:
            if batch_loss_sum < train_min_loss:
                train_min_loss = batch_loss_sum
                is_save_model = True

        if is_save_model:
            file_name_loss = math.floor(batch_loss_sum * 1e6)
            file_name = "model_deepodo_wang_train_schedule_{}_epoch_{}_{}_batch_{}_{}_Loss_{}.p".format(time1,
                                                                                                        t1, epoch, current,
                                                                                                        size,
                                                                                                        file_name_loss)
            file_path1 = os.path.join(os.path.abspath('.'), 'experiments', 'checkpoints', file_name)
            torch.save(model.state_dict(), file_path1)
            file_path2 = os.path.join(os.path.abspath('.'), 'experiments', 'checkpoints', 'model_deepodo_wang.p')
            torch.save(model.state_dict(), file_path2)

    epoch_loss_avg = epoch_loss_sum / epoch_loss_num
    logger_train.info(f"epoch loss: {epoch_loss_avg:>7f}")

    return train_min_loss
