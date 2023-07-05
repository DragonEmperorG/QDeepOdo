import math
import os

import numpy as np
import torch
from matplotlib import pyplot as plt


def test(dataloader, model, loss_fn1, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            input_float = X.float()
            output_float = y.float()
            pred = model(input_float)
            test_loss += loss_fn1(pred, output_float).item()

            pred_numpy = pred.cpu().numpy()
            output_float_numpy = output_float.cpu().numpy()
            pred_numpy_length = pred_numpy.shape[1]
            t = np.linspace(0, pred_numpy_length-1, pred_numpy_length)
            fig, ax = plt.subplots(figsize=(9/2.54, 6.75/2.54), dpi=600)
            ax.plot(t, output_float_numpy[0, :, 0], color="red", linestyle="-")
            ax.plot(t, pred_numpy[0, :, 0], color="blue", linestyle="--")
            ax.set(xlabel='Time (s)', ylabel='Forward Velocity (m/s)')
            ax.legend(['Ground Truth', 'DeepOdo Network'], loc='upper right')
            ylim_numpy = np.array(ax.get_ylim())
            ylim_fixed_numpy = ylim_numpy + [0, 2]
            ylim_fixed_tuple = tuple(ylim_fixed_numpy)
            ax.set_ylim(ylim_fixed_tuple)
            plt.tight_layout()
            fig.savefig(os.path.join(os.path.abspath('.'), "f.png"))
            plt.show()

    test_loss /= num_batches
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
