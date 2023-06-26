import torch
from torch import nn
from torch.autograd import Variable


class DeepOdoModel(nn.Module):
    def __init__(self):
        super(DeepOdoModel, self).__init__()

        self.conv1d1 = nn.Conv1d(7, 128, 11)
        self.maxpool1d1 = nn.MaxPool1d(2)
        self.conv1d2 = nn.Conv1d(128, 256, 9)
        self.maxpool1d2 = nn.MaxPool1d(2)
        self.flatten1 = nn.Flatten(0)
        self.fc1 = nn.Linear(1536, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.gru = nn.GRU(512, 1)
        self.gru_cell = nn.GRUCell(512, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, phone_data):
        batch_length = phone_data.shape[0]
        batch_output = []
        for i in range(batch_length):
            sequence_output = []
            input_sequence_single = phone_data[i, :, :, :]
            input_sequence_single_length = input_sequence_single.shape[0]
            hx = Variable(torch.randn(512))
            for j in range(input_sequence_single_length):
                input_atom = input_sequence_single[j, :, :]
                input_atom_conv1d = input_atom.permute(1, 0)
                # https://stackoverflow.com/questions/66074684/runtimeerror-expected-scalar-type-double-but-found-float-in-pytorch-cnn-train
                cnn_module = self.conv1d1(input_atom_conv1d)
                cnn_module = self.maxpool1d1(cnn_module)
                cnn_module = self.conv1d2(cnn_module)
                cnn_module = self.maxpool1d2(cnn_module)
                cnn_module = self.flatten1(cnn_module)
                cnn_module = self.fc1(cnn_module)
                cnn_module = self.fc2(cnn_module)
                hx = self.gru_cell(cnn_module, hx)
                hx = hx.detach()
                gru_module = self.fc3(hx)
                sequence_output.append(gru_module)

            sequence_output_stack = torch.stack(sequence_output, dim=0)
            batch_output.append(sequence_output_stack)

        batch_output_stack = torch.stack(batch_output, dim=0)

        return batch_output_stack

    def forward_1(self, phone_data):
        batch_length = phone_data.shape[0]
        batch_output = []
        for i in range(batch_length):
            sequence_output = []
            input_sequence_single = phone_data[i, :, :, :]
            input_sequence_single_length = input_sequence_single.shape[0]
            h0 = Variable(torch.randn(1))
            for j in range(input_sequence_single_length):
                input_atom = input_sequence_single[j, :, :]
                input_atom_conv1d = input_atom.permute(1, 0)
                # https://stackoverflow.com/questions/66074684/runtimeerror-expected-scalar-type-double-but-found-float-in-pytorch-cnn-train
                cnn_module = self.conv1d1(input_atom_conv1d)
                cnn_module = self.maxpool1d1(cnn_module)
                cnn_module = self.conv1d2(cnn_module)
                cnn_module = self.maxpool1d2(cnn_module)
                cnn_module = self.flatten1(cnn_module)
                cnn_module = self.fc1(cnn_module)
                cnn_module = self.fc2(cnn_module)
                sequence_output.append(cnn_module)

            sequence_output_stack = torch.stack(sequence_output, dim=0)
            sequence_output_stack1, h = self.gru(sequence_output_stack)
            batch_output.append(sequence_output_stack1)

        batch_output_stack = torch.stack(batch_output, dim=0)

        return batch_output_stack
