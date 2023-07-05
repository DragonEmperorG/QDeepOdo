import math
import os

import torch


def train(dataloader, model, loss_fn1, optimizer1, t1, epoch, time1, min_loss1, writer1, device):
    size = len(dataloader.dataset)
    model.train()
    optimizer1.zero_grad()
    train_min_loss = min_loss1

    for batch, (sensor_data, ground_truth) in enumerate(dataloader):
        sensor_data, ground_truth = sensor_data.to(device), ground_truth.to(device)
        input_float = sensor_data.float()
        output_float = ground_truth.float()

        # Compute prediction error
        pred = model(input_float)
        loss = loss_fn1(pred, output_float)

        # Backpropagation
        loss.backward()
        optimizer1.step()
        optimizer1.zero_grad()

        writer1.add_scalar("Loss/train", loss, t1)
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
