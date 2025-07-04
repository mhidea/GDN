import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from util.consts import Tasks
from parameters.params import Params
import pandas as pd
from util.env import get_param
from util.data import sensorGroup_to_xy

_x_dim = 0
_y_dim = 0


def getDimensions():
    global _x_dim
    global _y_dim
    return _x_dim, _y_dim


def setDimensions(x_dim, y_dim):
    global _x_dim
    global _x_dim
    _x_dim = x_dim
    y_dim = _y_dim


class TimeDataset(Dataset):
    def __init__(
        self,
        data_frame: pd.DataFrame,
        column_groups: tuple,
        mode="train",
    ):
        global _x_dim
        global _y_dim
        self.device = get_param().device
        self.window_length = get_param().window_length
        self.stride = get_param().stride
        self.total_rows = data_frame.shape[0]
        if mode == "train" and "attack" not in data_frame.columns:
            data_frame["attack"] = torch.zeros(
                self.total_rows,
                dtype=torch.float32,
                # device=self.device,
                requires_grad=False,
            )

        self.label = torch.tensor(
            data_frame["attack"].to_numpy(),
            dtype=torch.float32,
            # device=self.device,
            requires_grad=False,
        )
        self.label = self.label.reshape(-1, 1).contiguous().pin_memory()

        xlist, ylist, next = sensorGroup_to_xy(column_groups, get_param().task)

        self.next_shifter: int = 0 if next else -1
        self.x = (
            torch.tensor(
                data_frame[xlist].to_numpy(),
                dtype=torch.float32,
                # device=self.device,
                requires_grad=False,
            )
            .contiguous()
            .pin_memory()
        )
        if len(ylist) == 0:
            self.y = self.label
        else:
            self.y = (
                torch.tensor(
                    data_frame[ylist].to_numpy(),
                    dtype=torch.float32,
                    # device=self.device,
                    requires_grad=False,
                )
                .contiguous()
                .pin_memory()
            )
        _x_dim = self.x.shape[1]
        _y_dim = self.y.shape[1]

    def __len__(self):
        return (
            self.total_rows - self.window_length - (self.next_shifter + 1)
        ) // self.stride

    def __getitem__(self, idx):
        start = idx * self.stride
        return (
            self.x[start : start + self.window_length, :]
            .t()
            .clone()
            .to(device=self.device),
            self.y[start + self.window_length + self.next_shifter]
            .clone()
            .to(device=self.device),
            self.label[start + self.window_length + self.next_shifter, :]
            .clone()
            .to(device=self.device),
        )
