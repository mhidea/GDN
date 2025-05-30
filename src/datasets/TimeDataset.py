import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from util.consts import Tasks
from parameters.params import Params
import pandas as pd
from util.env import get_param


class TimeDataset(Dataset):
    def __init__(
        self,
        data_frame: pd.DataFrame,
        sensor_list: list,
        actuator_list: list,
        mode="train",
    ):
        self.device = get_param().device
        self.window_length = get_param().window_length
        self.stride = get_param().stride
        self.df_sensor = (
            torch.tensor(
                data_frame[sensor_list].to_numpy(),
                dtype=torch.float32,
                # device=self.device,
                requires_grad=False,
            )
            .contiguous()
            .pin_memory()
        )
        # self.df_actuator = torch.tensor(
        #     data_frame[actuator_list].to_numpy(), dtype=torch.float32
        # )
        if mode == "train" and "attack" not in data_frame.columns:
            self.label = torch.zeros(
                self.df_sensor.shape[0],
                dtype=torch.float32,
                # device=self.device,
                requires_grad=False,
            )
        else:
            self.label = torch.tensor(
                data_frame["attack"].to_numpy(),
                dtype=torch.float32,
                # device=self.device,
                requires_grad=False,
            )
        self.label = self.label.reshape(-1, 1).contiguous().pin_memory()

    def __len__(self):
        return (self.df_sensor.shape[0] - self.window_length - 1) // self.stride

    def __getitem__(self, idx):
        start = idx * self.stride
        return (
            self.df_sensor[start : start + self.window_length, :]
            .t()
            .to(device=self.device),
            self.df_sensor[start + self.window_length].to(device=self.device),
            self.label[start + self.window_length, :].to(device=self.device),
            # .unsqueeze()
            # .requires_grad_(False).to(device=self.device, non_blocking=False),
        )
