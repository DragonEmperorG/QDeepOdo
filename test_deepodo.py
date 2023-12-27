import math
import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from utils.logs.log_utils import get_logger


def test(dataloader, model, loss_fn1, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch, batch_data in enumerate(dataloader):
            for dataset_trace in batch_data:
                folder_path = dataset_trace.folder_path
                sensor_data_numpy = dataset_trace.phone_sensor_data_sequence
                ground_truth_numpy = dataset_trace.ground_truth_data_sequence
                sensor_data_tensor, ground_truth_tensor = torch.tensor(sensor_data_numpy), torch.tensor(ground_truth_numpy)
                X, y = sensor_data_tensor.to(device), ground_truth_tensor.to(device)
                input_float = X.float()
                output_float = y.float()
                pred = model(device, input_float)
                test_loss += loss_fn1(pred, output_float).item()

                pred_numpy = pred.cpu().numpy()
                output_float_numpy = output_float.cpu().numpy()
                pred_numpy_length = pred_numpy.shape[0]
                t = np.linspace(0, pred_numpy_length-1, pred_numpy_length)
                fig, ax = plt.subplots(figsize=(9/2.54, 6.75/2.54), dpi=600)
                ax.plot(t, output_float_numpy, color="red", linestyle="-")
                ax.plot(t, pred_numpy, color="blue", linestyle="--")
                ax.set(xlabel='Time (s)', ylabel='Forward Velocity (m/s)')
                ax.legend(['Ground Truth', 'DeepOdo Network'], loc='upper right')
                ylim_numpy = np.array(ax.get_ylim())
                ylim_fixed_numpy = ylim_numpy + [0, 2]
                ylim_fixed_tuple = tuple(ylim_fixed_numpy)
                ax.set_ylim(ylim_fixed_tuple)
                plt.tight_layout()

                level3_folder_path, phone_name = os.path.split(folder_path)
                q1, drive_id = os.path.split(level3_folder_path)
                save_fig_name = drive_id + "_" + phone_name + "_forward_velocity_result.png"
                fig.savefig(os.path.join(os.path.abspath('.'), save_fig_name))
                plt.show()

                # np.savetxt('DeepOdoPredictData.txt', pred_numpy)

    test_loss /= num_batches
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def test_loss(dataloader, model, loss_fn1, device, writer1, t1):
    logger_test = get_logger()
    model.eval()

    test_loss_num = 0
    test_loss_sum = 0
    with torch.no_grad():
        for batch, batch_data in enumerate(dataloader):
            for dataset_trace in batch_data:
                sensor_data_numpy = dataset_trace.phone_sensor_data_sequence
                ground_truth_numpy = dataset_trace.ground_truth_data_sequence
                sensor_data_tensor, ground_truth_tensor = torch.tensor(sensor_data_numpy), torch.tensor(ground_truth_numpy)
                X, y = sensor_data_tensor.to(device), ground_truth_tensor.to(device)
                input_float = X.float()
                output_float = y.float()
                pred = model(device, input_float)
                sequence_loss = loss_fn1(pred, output_float)

                sequence_len = len(ground_truth_numpy)
                test_loss_num = test_loss_num + sequence_len
                test_loss_sum = test_loss_sum + sequence_loss.item() * sequence_len

    test_loss_avg = test_loss_sum / test_loss_num

    writer1.add_scalar("Loss/test", test_loss_avg, t1)
    logger_test.info(f"Test loss: {test_loss_avg:>8f}")
