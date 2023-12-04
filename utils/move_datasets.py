import os
import shutil

from utils.ScriptArgs import load_args

if __name__ == '__main__':

    loaded_args = load_args(0)

    dataset_deepodo_folder_name = "dataset_deepodo"
    dataset_deepodo_numpy_file_name = "train_data_deepodo.npy"

    reference_folder_path = os.path.abspath('.')
    move_to_data_base_folder_path = os.path.normpath(os.path.join(reference_folder_path, '..', 'datas', 'sdc2023', 'train'))

    for i in range(len(loaded_args.datasets_train_test_config)):
        move_from_data_folder_path = os.path.join(
            loaded_args.datasets_base_folder_path,
            loaded_args.datasets_train_test_config[i][1],
            loaded_args.datasets_train_test_config[i][2],
            dataset_deepodo_folder_name,
        )

        move_from_data_file_path = os.path.join(
            move_from_data_folder_path,
            dataset_deepodo_numpy_file_name,
        )

        move_to_data_folder_path = os.path.join(
            move_to_data_base_folder_path,
            loaded_args.datasets_train_test_config[i][1],
            loaded_args.datasets_train_test_config[i][2],
            dataset_deepodo_folder_name,
        )

        move_to_data_file_path = os.path.join(
            move_to_data_folder_path,
            dataset_deepodo_numpy_file_name,
        )

        if not os.path.isdir(move_to_data_folder_path):
            os.makedirs(move_to_data_folder_path)
        shutil.move(move_from_data_file_path, move_to_data_file_path)
