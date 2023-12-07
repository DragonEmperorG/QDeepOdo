import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.deepodo_dataloader import load_chongqin_dataset, load_sdc2023_dataset

from datasets.deepodo_sdc2023_dataset import DeepOdoSdcDataset
from graphs.models.deepodo_6axis_imu_model import DeepOdo6AxisImuModel
from graphs.models.deepodo_model import DeepOdoModel
from test_deepodo import test, test_loss
from train_deepodo import train
from utils.ScriptArgs import load_args
from utils.logs.log_utils import get_logger, init_logger


ARGS_INPUT_MODE = 0


def collate_fn(data):
    return data


def main(args):
    logger_main = get_logger()

    # train_list, test_list = load_chongqin_dataset(args.datasets_base_folder_path, args.datasets_train_test_config)
    # deepodo_model = DeepOdoModel()

    train_list, test_list, normalize_factors = load_sdc2023_dataset(args.datasets_base_folder_path, args.datasets_train_test_config)
    train_dataset = DeepOdoSdcDataset(train_list, normalize_factors)
    test_dataset = DeepOdoSdcDataset(test_list, normalize_factors)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn)
    deepodo_model = DeepOdo6AxisImuModel()

    deepodo_model.to(args.device)

    loss_fn = nn.MSELoss(reduction="mean")

    if args.train_filter:
        if args.continue_training:
            file_path = os.path.join(os.path.abspath('.'), 'experiments', 'checkpoints', args.model_file_name)
            deepodo_model.load_state_dict(torch.load(file_path))
            logger.info("Loaded model path: {}", file_path)
        else:
            logger.info("New train task")

        optimizer = torch.optim.Adam(deepodo_model.parameters(), lr=args.learning_rate)

        file_name_train_head_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))

        layout = {
            "SDC2023": {
                "loss": ["Multiline", ["Loss/train", "Loss/test"]],
            },
        }
        writer = SummaryWriter()
        writer.add_custom_scalars(layout)

        min_loss = -1
        for t in range(args.epochs):
            logger_main.info(f"Epoch {t + 1}/{args.epochs}")
            min_loss = train(
                train_dataloader, deepodo_model, loss_fn, optimizer,
                t, args.epochs, file_name_train_head_time, min_loss, writer, args.device)

            if t % 1 == 0:
                test_loss(test_dataloader, deepodo_model, loss_fn, args.device, writer, t)

    if args.test_filter:
        file_path = os.path.join(os.path.abspath('.'), 'experiments', 'checkpoints', args.model_file_name)
        deepodo_model.load_state_dict(torch.load(file_path))
        logger.info("Loaded model path: {}", file_path)
        test(test_dataloader, deepodo_model, loss_fn, args.device)
    logger_main.info(f"Finish task!")


if __name__ == '__main__':
    #
    init_logger()
    logger = get_logger()

    loaded_args = load_args(ARGS_INPUT_MODE)
    logger.info("Loaded task configuration")
    if loaded_args.device == "cuda":
        cuda_device_count = torch.cuda.device_count()
        cuda_current_device = torch.cuda.current_device()
        cuda_current_device_name = torch.cuda.get_device_name(cuda_current_device)
        logger.info("Device: {}:{} | {}, total {}",
                    loaded_args.device,
                    cuda_current_device,
                    cuda_current_device_name,
                    cuda_device_count
                    )
    else:
        logger.info("Device: {}", loaded_args.device)

    main(loaded_args)
