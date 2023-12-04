import os

import torch


class ScriptArgs:
    # Global
    reference_folder_path = os.path.abspath('.')
    datasets_base_folder_path = os.path.normpath(os.path.join(reference_folder_path, 'datas', 'sdc2023', 'train'))
    datasets_train_test_config = []

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    # Train
    train_filter = False
    epochs = 1000
    learning_rate = 0.00001
    batch_size = 20
    seq_len = 3000
    max_loss = 500
    max_grad_norm = 500
    continue_training = False
    model_file_name = "filter_schedule_20231119_142127_epoch_771_1000_loss_72058.p"

    test_filter = False


def load_args(args_input_mode):
    if args_input_mode == 1:
        return load_terminal_args()
    else:
        return load_default_args()


def load_terminal_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-04)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--max_loss', type=float, default=2e1)
    parser.add_argument('--max_grad_norm', type=float, default=1e0)

    parser.add_argument('--continue_from', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--model_path', type=str, default=None)

    args = parser.parse_args()
    return args


def load_default_args():
    # default_args = load_default_chongqin_datasets_args()
    default_args = load_default_sdc2023_datasets_args()
    # default_args = load_test_sdc2023_datasets_args()
    return default_args


def load_default_chongqin_datasets_args():
    chongqin_datasets_args = ScriptArgs()
    chongqin_datasets_args.datasets_base_folder_path = "E:\\DoctorRelated\\20230410重庆VDR数据采集"
    chongqin_datasets_args.datasets_train_test_config = [
        ['TRAIN', '2023_04_10', '0008', 'HUAWEI_Mate30'],
        ['TRAIN', '2023_04_10', '0009', 'HUAWEI_Mate30'],
        ['TRAIN', '2023_04_10', '0010', 'HUAWEI_Mate30'],
        ['TRAIN', '2023_04_10', '0011', 'HUAWEI_Mate30'],
        ['TRAIN', '2023_04_10', '0012', 'HUAWEI_Mate30'],
        ['TRAIN', '2023_04_10', '0013', 'HUAWEI_Mate30'],
        ['TRAIN', '2023_04_10', '0014', 'HUAWEI_Mate30'],
        ['TRAIN', '2023_04_10', '0015', 'HUAWEI_Mate30'],
        ['TRAIN', '2023_04_10', '0016', 'HUAWEI_Mate30'],
        ['TRAIN', '2023_04_10', '0017', 'HUAWEI_Mate30'],
        ['TEST', '2023_04_10', '0018', 'HUAWEI_Mate30'],
    ]
    chongqin_datasets_args.epochs = 3000
    chongqin_datasets_args.learning_rate = 0.0001
    chongqin_datasets_args.seq_len = 6000
    chongqin_datasets_args.max_loss = 100
    chongqin_datasets_args.max_grad_norm = 100
    chongqin_datasets_args.continue_training = True
    chongqin_datasets_args.model_file_name = "filter_schedule_20231120_144536_epoch_602_1000_loss_71825.p"
    return chongqin_datasets_args


def load_default_sdc2023_datasets_args():
    sdc2023_datasets_args = ScriptArgs()
    # sdc2023_datasets_args.datasets_base_folder_path = "E:\\GitHubRepositories\\QGSDC2023\\Data\\sdc2023\\train"
    sdc2023_datasets_args.datasets_train_test_config = [
        ['TRAIN', '2020-08-04-00-20-us-ca-sb-mtv-101', 'pixel5'],
        ['TEST', '2020-08-13-21-42-us-ca-mtv-sf-280', 'pixel5'],
        ['TRAIN', '2020-12-10-22-52-us-ca-sjc-c', 'pixel5'],
        ['TRAIN', '2021-01-04-21-50-us-ca-e1highway280driveroutea', 'pixel5'],
        ['TRAIN', '2021-01-04-22-40-us-ca-mtv-a', 'pixel5'],
        ['TRAIN', '2021-03-10-23-13-us-ca-mtv-h', 'pixel5'],
        ['TRAIN', '2021-03-16-18-59-us-ca-mtv-a', 'pixel5'],
        ['TRAIN', '2021-03-16-20-40-us-ca-mtv-b', 'pixel5'],
        ['TEST', '2021-04-02-20-43-us-ca-mtv-f', 'pixel5'],
        ['TRAIN', '2021-04-08-21-28-us-ca-mtv-k', 'pixel5'],
        ['TRAIN', '2021-07-14-20-50-us-ca-mtv-e', 'pixel5'],
        ['TRAIN', '2021-07-19-20-49-us-ca-mtv-a', 'pixel5'],
        ['TRAIN', '2021-07-27-19-49-us-ca-mtv-b', 'pixel5'],
        ['TEST', '2021-08-24-20-32-us-ca-mtv-h', 'pixel5'],
        ['TRAIN', '2021-12-07-19-22-us-ca-lax-d', 'pixel5'],
        ['TRAIN', '2021-12-07-22-21-us-ca-lax-g', 'pixel5'],
        ['TRAIN', '2021-12-08-17-22-us-ca-lax-a', 'pixel5'],
        ['TRAIN', '2021-12-08-20-28-us-ca-lax-c', 'pixel5'],
        ['TEST', '2021-12-09-17-06-us-ca-lax-e', 'pixel5'],
        ['TEST', '2022-01-11-18-48-us-ca-mtv-n', 'pixel5'],
        ['TRAIN', '2022-01-26-20-02-us-ca-mtv-pe1', 'pixel5'],
        ['TRAIN', '2022-02-24-18-29-us-ca-lax-o', 'pixel5'],
        ['TRAIN', '2022-04-01-18-22-us-ca-lax-t', 'pixel5'],
        ['TRAIN', '2022-07-26-21-01-us-ca-sjc-s', 'pixel5'],
        ['TRAIN', '2022-08-04-20-07-us-ca-sjc-q', 'pixel5'],
        ['TRAIN', '2022-11-15-00-53-us-ca-mtv-a', 'pixel5'],
        ['TEST', '2023-03-08-21-34-us-ca-mtv-u', 'pixel5'],
        ['TRAIN', '2023-05-09-21-32-us-ca-mtv-pe1', 'pixel5'],
        ['TRAIN', '2023-05-16-19-54-us-ca-mtv-xe1', 'pixel5'],
        ['TRAIN', '2023-05-19-20-10-us-ca-mtv-ie2', 'pixel5'],
        ['TRAIN', '2023-05-23-19-16-us-ca-mtv-ie2', 'pixel5'],
        ['TRAIN', '2023-05-24-20-26-us-ca-sjc-ge2', 'pixel5'],
        ['TRAIN', '2023-05-25-19-10-us-ca-sjc-be2', 'pixel5'],
        ['TEST', '2023-05-25-20-11-us-ca-sjc-he2', 'pixel5'],
        ['TRAIN', '2023-05-26-18-51-us-ca-sjc-ge2', 'pixel5'],
        ['TRAIN', '2023-09-05-20-13-us-ca', 'pixel5'],
        ['TEST', '2023-09-05-23-07-us-ca-routen', 'pixel5'],
        ['TRAIN', '2023-09-06-18-04-us-ca', 'pixel5'],
        ['TRAIN', '2023-09-07-18-59-us-ca', 'pixel5'],
    ]
    # sdc2023_datasets_args.device = "cpu"
    sdc2023_datasets_args.train_filter = True
    sdc2023_datasets_args.epochs = 1000
    sdc2023_datasets_args.learning_rate = 0.0001
    sdc2023_datasets_args.batch_size = 32
    sdc2023_datasets_args.continue_training = True
    sdc2023_datasets_args.model_file_name = "model_deepodo_wang_train_schedule_20231124_193025_epoch_448_1000_batch_31_31_Loss_3096443.p"
    return sdc2023_datasets_args


def load_test_sdc2023_datasets_args():
    sdc2023_datasets_args = ScriptArgs()
    sdc2023_datasets_args.datasets_base_folder_path = "E:\\GitHubRepositories\\QGSDC2023\\Data\\sdc2023\\train"
    sdc2023_datasets_args.datasets_train_test_config = [
        ['TRAIN', '2020-08-04-00-20-us-ca-sb-mtv-101', 'pixel5'],
        ['TEST', '2020-08-13-21-42-us-ca-mtv-sf-280', 'pixel5'],
        ['TRAIN', '2020-12-10-22-52-us-ca-sjc-c', 'pixel5'],
        ['TRAIN', '2021-01-04-21-50-us-ca-e1highway280driveroutea', 'pixel5'],
        ['TRAIN', '2021-01-04-22-40-us-ca-mtv-a', 'pixel5'],
        ['TRAIN', '2021-03-10-23-13-us-ca-mtv-h', 'pixel5'],
        ['TRAIN', '2021-03-16-18-59-us-ca-mtv-a', 'pixel5'],
        ['TRAIN', '2021-03-16-20-40-us-ca-mtv-b', 'pixel5'],
        ['TEST', '2021-04-02-20-43-us-ca-mtv-f', 'pixel5'],
        ['TRAIN', '2021-04-08-21-28-us-ca-mtv-k', 'pixel5'],
        ['TRAIN', '2021-07-14-20-50-us-ca-mtv-e', 'pixel5'],
        ['TRAIN', '2021-07-19-20-49-us-ca-mtv-a', 'pixel5'],
        ['TRAIN', '2021-07-27-19-49-us-ca-mtv-b', 'pixel5'],
        ['TEST', '2021-08-24-20-32-us-ca-mtv-h', 'pixel5'],
        ['TRAIN', '2021-12-07-19-22-us-ca-lax-d', 'pixel5'],
        ['TRAIN', '2021-12-07-22-21-us-ca-lax-g', 'pixel5'],
        ['TRAIN', '2021-12-08-17-22-us-ca-lax-a', 'pixel5'],
        ['TRAIN', '2021-12-08-20-28-us-ca-lax-c', 'pixel5'],
        ['TEST', '2021-12-09-17-06-us-ca-lax-e', 'pixel5'],
        ['TEST', '2022-01-11-18-48-us-ca-mtv-n', 'pixel5'],
        ['TRAIN', '2022-01-26-20-02-us-ca-mtv-pe1', 'pixel5'],
        ['TRAIN', '2022-02-24-18-29-us-ca-lax-o', 'pixel5'],
        ['TRAIN', '2022-04-01-18-22-us-ca-lax-t', 'pixel5'],
        ['TRAIN', '2022-07-26-21-01-us-ca-sjc-s', 'pixel5'],
        ['TRAIN', '2022-08-04-20-07-us-ca-sjc-q', 'pixel5'],
        ['TRAIN', '2022-11-15-00-53-us-ca-mtv-a', 'pixel5'],
        ['TEST', '2023-03-08-21-34-us-ca-mtv-u', 'pixel5'],
        ['TRAIN', '2023-05-09-21-32-us-ca-mtv-pe1', 'pixel5'],
        ['TRAIN', '2023-05-16-19-54-us-ca-mtv-xe1', 'pixel5'],
        ['TRAIN', '2023-05-19-20-10-us-ca-mtv-ie2', 'pixel5'],
        ['TRAIN', '2023-05-23-19-16-us-ca-mtv-ie2', 'pixel5'],
        ['TRAIN', '2023-05-24-20-26-us-ca-sjc-ge2', 'pixel5'],
        ['TRAIN', '2023-05-25-19-10-us-ca-sjc-be2', 'pixel5'],
        ['TEST', '2023-05-25-20-11-us-ca-sjc-he2', 'pixel5'],
        ['TRAIN', '2023-05-26-18-51-us-ca-sjc-ge2', 'pixel5'],
        ['TRAIN', '2023-09-05-20-13-us-ca', 'pixel5'],
        ['TEST', '2023-09-05-23-07-us-ca-routen', 'pixel5'],
        ['TRAIN', '2023-09-06-18-04-us-ca', 'pixel5'],
        ['TRAIN', '2023-09-07-18-59-us-ca', 'pixel5'],
    ]
    sdc2023_datasets_args.device = "cpu"
    sdc2023_datasets_args.test_filter = True
    sdc2023_datasets_args.model_file_name = "model_deepodo_wang_train_schedule_20231124_193025_epoch_448_1000_batch_31_31_Loss_3096443.p"
    return sdc2023_datasets_args
