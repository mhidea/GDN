import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from util.consts import Tasks
from util.params import Params
import pandas as pd
from util.env import get_param


class CurrentLabelDataset(Dataset):
    def __init__(self, data_frame: pd.DataFrame, sensor_list: list, mode="train"):
        self.df_sensor = torch.tensor(
            data_frame[sensor_list].to_numpy(), dtype=torch.float32
        ).contiguous()
        if mode == "train" and "attack" not in data_frame.columns:
            self.label = torch.zeros(self.df_sensor.shape[0]).float()
        else:
            self.label = torch.tensor(
                data_frame["attack"].to_numpy(), dtype=torch.float32
            )
        self.label = self.label.reshape(-1, 1).contiguous()
        self.device = get_param().device

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):

        return (
            self.df_sensor[idx]
            .unsqueeze(-1)
            .to(self.device, non_blocking=True)
            .contiguous(),
            None,
            self.label[idx, :]
            # .unsqueeze()
            .to(self.device, non_blocking=True),
        )


class CurrentDataset(Dataset):
    def __init__(
        self,
        data_frame: pd.DataFrame,
        sensor_list: list,
        actuator_list: list,
        mode="train",
    ):
        self.df_sensor = torch.tensor(
            data_frame[sensor_list].to_numpy(), dtype=torch.float32
        ).contiguous()
        self.df_actuator = torch.tensor(
            data_frame[actuator_list].to_numpy(), dtype=torch.float32
        ).contiguous()

        if mode == "train" and "attack" not in data_frame.columns:
            self.label = torch.zeros(self.df_sensor.shape[0]).float()
        else:
            self.label = torch.tensor(
                data_frame["attack"].to_numpy(), dtype=torch.float32
            )
        self.label = self.label.reshape(-1, 1).contiguous()
        self.device = get_param().device

    def __len__(self):
        return self.df_sensor.shape[0]

    def __getitem__(self, idx):

        return (
            self.df_sensor[idx]
            .unsqueeze(-1)
            .to(self.device, non_blocking=True)
            .contiguous(),
            self.df_actuator[idx]
            .unsqueeze(-1)
            .to(self.device, non_blocking=True)
            .contiguous(),
            self.label[idx, :]
            # .unsqueeze()
            .to(self.device, non_blocking=True),
        )
