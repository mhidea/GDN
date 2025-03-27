import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from util.consts import Tasks
from util.params import Params


class TimeDataset(Dataset):
    def __init__(self, raw_data, edge_index, param: Params, mode="train"):
        self.raw_data = raw_data
        self.param = param
        self.edge_index = edge_index.long()
        self.mode = mode

        x_data = raw_data[:-1]
        labels = raw_data[-1]

        data = x_data

        # to tensor
        data = torch.tensor(data).float()
        labels = torch.tensor(labels).float()

        self.x, self.y, self.labels = self.process(data, labels)

    def __len__(self):
        return len(self.x)

    def process(self, data, labels):
        x_arr, y_arr = [], []
        labels_arr = []

        is_train = self.mode == "train"

        node_num, total_time_len = data.shape

        rang = (
            range(self.param.window_length, total_time_len, self.param.stride)
            if is_train
            else range(self.param.window_length, total_time_len)
        )

        for i in rang:

            ft = data[:, i - self.param.window_length : i]
            x_arr.append(ft)
            tar = data[:, i]
            y_arr.append(tar)

            labels_arr.append(labels[i])

        x = torch.stack(x_arr).contiguous()
        y = torch.stack(y_arr).contiguous()

        labels = torch.Tensor(labels_arr).contiguous()

        return x, y, labels

    def __getitem__(self, idx):

        windowed_x = self.x[idx]
        y = self.y[idx]

        edge_index = self.edge_index

        next_label = self.labels[idx]

        return (
            windowed_x,
            y,
            next_label,
            edge_index,
        )
