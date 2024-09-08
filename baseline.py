import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

import numpy as np


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]#.contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        #self.init_weights()
        self.initialize_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, output_size, input_size=16, num_channels=[8, 32, 64], kernel_size=3, dropout=0.4):
        super(TCN, self).__init__()
        self._batch_norm0 = nn.BatchNorm1d(input_size)
        self._tcn1 = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        #self.linear = nn.Linear(num_channels[-1], output_size)
        #self._fc1 = nn.Linear(num_channels[-1]*400, 5120)

        self._output = nn.Linear(num_channels[-1], output_size)
        #print("Number Parameters: ", self.get_n_params())
        self.initialize_weights()
        print("Number Parameters: ", self.get_n_params())
    
    def _hook_before_epoch(self, epoch):
        pass

    def _hook_after_epoch(self, epoch):
        pass

    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, inputs, target=None):
        """Inputs have to have dimension (N, C_in, L_in)"""
        '''
        x = []
        for i in range(len(A)):
            x.append(torch.mean(inputs[:, A[i], :], dim=1))
        y1=self.tcn(torch.stack(x, dim=1))
        '''
        if inputs.shape[0] != 1:
            inputs = inputs.squeeze()
        temporal_features1 = self._tcn1(self._batch_norm0(inputs))  # input should have dimension (N, C, L)
        output = self._output(temporal_features1[:,:,-1])

        return output, None, None # F.log_softmax(o, dim=1)


class ECNN(nn.Module):
    def __init__(self, number_of_class=10, dropout=0.5, k_c=3):
        # k_c: kernel size of channel
        super(ECNN, self).__init__()
        # self._batch_norm0 = nn.BatchNorm2d(1)
        self._batch_norm0 = nn.BatchNorm2d(1)
        self._conv1 = nn.Conv2d(1, 32, kernel_size=(k_c, 5), bias=False, padding=(1,2))
        self._batch_norm1 = nn.BatchNorm2d(32)
        self._prelu1 = nn.PReLU(32)
        self._dropout1 = nn.Dropout2d(dropout)
        self._pool1 = nn.MaxPool2d(kernel_size=(1, 3))

        self._conv2 = nn.Conv2d(32, 64, kernel_size=(k_c, 5), bias=False, padding=(1,2))
        self._batch_norm2 = nn.BatchNorm2d(64)
        self._prelu2 = nn.PReLU(64)
        self._dropout2 = nn.Dropout2d(dropout)
        self._pool2 = nn.MaxPool2d(kernel_size=(1, 3))

        self._conv3 = nn.Conv2d(64, 128, kernel_size=(k_c, 5), bias=False, padding=(1,2))
        self._batch_norm3 = nn.BatchNorm2d(128)
        self._prelu3 = nn.PReLU(128)
        self._dropout3 = nn.Dropout2d(dropout)
        self._pool3 = nn.MaxPool2d(kernel_size=(1, 3))

        self._fc1 = nn.Linear(128*16*1, 1024)
        self._fc_batch_norm1 = nn.BatchNorm1d(1024)
        self._fc_prelu1 = nn.PReLU(1024)
        self._fc_dropout1 = nn.Dropout(dropout)

        self._fc2 = nn.Linear(1024, 256)
        self._fc_batch_norm2 = nn.BatchNorm1d(256)
        self._fc_prelu2 = nn.PReLU(256)
        self._fc_dropout2 = nn.Dropout(dropout)

        self._output = nn.Linear(256, number_of_class)
        self.initialize_weights()

        print("Number Parameters: ", self.get_n_params())

    def _hook_before_epoch(self, epoch):
        pass

    def _hook_after_epoch(self, epoch):
        pass

    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    def init_weights(self):
        for m in self.modules():
            torch.nn.init.kaiming_normal(m.weight)
            m.bias.data.zero_()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, y=None):
        x = x.unsqueeze(1)
        conv1 = self._dropout1(self._prelu1(self._batch_norm1(self._conv1(self._batch_norm0(x)))))
        pool1 = self._pool1(conv1)

        conv2 = self._dropout2(
            self._prelu2(self._batch_norm2(self._conv2(pool1))))
        pool2 = self._pool2(conv2)

        conv3 = self._dropout3(
            self._prelu3(self._batch_norm3(self._conv3(pool2))))
        pool3 = self._pool3(conv3)
        flatten_tensor = pool3.view(pool3.size(0), -1)

        fc1 = self._fc_dropout1(
            self._fc_prelu1(self._fc_batch_norm1(self._fc1(flatten_tensor))))
        fc2 = self._fc_dropout2(
            self._fc_prelu2(self._fc_batch_norm2(self._fc2(fc1))))
        output = self._output(fc2)
        return output, None, None


class ECNN_Model(nn.Module):
    def __init__(self, number_of_class=12, dropout=0.5, k_c=3):
        # k_c: kernel size of channel
        super(ECNN_Model, self).__init__()
        # self._batch_norm0 = nn.BatchNorm2d(1)
        self._conv1 = nn.Conv2d(1, 32, kernel_size=(k_c, 5))
        self._batch_norm1 = nn.BatchNorm2d(32)
        self._prelu1 = nn.PReLU(32)
        self._dropout1 = nn.Dropout2d(dropout)
        self._pool1 = nn.MaxPool2d(kernel_size=(1, 3))

        self._conv2 = nn.Conv2d(32, 64, kernel_size=(k_c, 5))
        self._batch_norm2 = nn.BatchNorm2d(64)
        self._prelu2 = nn.PReLU(64)
        self._dropout2 = nn.Dropout2d(dropout)
        self._pool2 = nn.MaxPool2d(kernel_size=(1, 3))

        self._fc1 = nn.Linear((16 - 2 * k_c + 2) * 3 * 64, 500)
        # 8 = 12 channels - 2 -2 ;  53 = ((500-4)/3-4)/3
        self._batch_norm3 = nn.BatchNorm1d(500)
        self._prelu3 = nn.PReLU(500)
        self._dropout3 = nn.Dropout(dropout)

        self._output = nn.Linear(500, number_of_class)
        self.initialize_weights()

        print("Number Parameters: ", self.get_n_params())
    
    def _hook_before_epoch(self, epoch):
        pass

    def _hook_after_epoch(self, epoch):
        pass

    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    def init_weights(self):
        for m in self.modules():
            torch.nn.init.kaiming_normal(m.weight)
            m.bias.data.zero_()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x, y=None):
        # x = x.permute(0,1,3,2)  --> batch * 1 * 16 * 50
        # print(x.size())
        x = x.unsqueeze(1)
        conv1 = self._dropout1(self._prelu1(self._batch_norm1(self._conv1(x))))
        # conv1 = self._dropout1(
        # self._prelu1(self._batch_norm1(self._conv1(self._batch_norm0(x)))))
        pool1 = self._pool1(conv1)
        conv2 = self._dropout2(
            self._prelu2(self._batch_norm2(self._conv2(pool1))))
        pool2 = self._pool2(conv2)
        flatten_tensor = pool2.view(pool2.size(0), -1)
        fc1 = self._dropout3(
            self._prelu3(self._batch_norm3(self._fc1(flatten_tensor))))
        output = self._output(fc1)
        return output, None, None


class CNN2DEncoder(nn.Module):
    def __init__(self, classes=10, kernel_size=3, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(kernel_size, 5)), # shape(B,32,14,46)
            nn.BatchNorm2d(32),
            nn.PReLU(32),
            nn.Dropout2d(dropout),
            # nn.MaxPool2d(kernel_size=(1, 3)),
            nn.AvgPool2d(kernel_size=(1, 3)), # shape(B,32,16,15)
            nn.Conv2d(32, 64, kernel_size=(kernel_size, 5)), # shape(B,64,12,11)
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            nn.Dropout2d(dropout),
            # nn.MaxPool2d(kernel_size=(1, 3)),
            nn.AvgPool2d(kernel_size=(1, 3)), # shape(B,64,12,3)
            nn.Flatten(),
        )

        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(64*12*3, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, classes),
        )
    
    def _hook_before_epoch(self, epoch):
        pass

    def _hook_after_epoch(self, epoch):
        pass

    def forward(self, input, target=None):
        """
        shape: (N, C, L)
        """
        logits = self.fc(self.net(input.unsqueeze(1)))
        return logits, logits, None


class CNN(nn.Module):
    def __init__(self, classes=10, kernel_size=3, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(kernel_size, 5)), # shape(B,32,14,46)
            nn.BatchNorm2d(32),
            nn.PReLU(32),
            nn.Dropout2d(dropout),
            # nn.MaxPool2d(kernel_size=(1, 3)),
            nn.AvgPool2d(kernel_size=(1, 3)), # shape(B,32,16,15)
            nn.Conv2d(32, 64, kernel_size=(kernel_size, 5)), # shape(B,64,12,11)
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            nn.Dropout2d(dropout),
            # nn.MaxPool2d(kernel_size=(1, 3)),
            nn.AvgPool2d(kernel_size=(1, 3)), # shape(B,64,12,3)
            nn.Flatten(),
        )
        in_dim = 64*12*3
        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, classes),
        )

        self.initialize_weights()

        print("Number Parameters: ", self.get_n_params())

    def get_n_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        number_params = sum([np.prod(p.size()) for p in model_parameters])
        return number_params

    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    
    def _hook_before_epoch(self, epoch):
        pass

    def _hook_after_epoch(self, epoch):
        pass

    def forward(self, input, target=None):
        """
        shape: (N, C, L)
        """
        logits = self.fc(self.net(input.unsqueeze(1)))
        return logits, logits, None
    