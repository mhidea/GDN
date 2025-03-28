import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time

import torch.utils

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F
import tqdm

from util.data import *
from util.preprocess import *
from util.time import *
from util.env import *
from util.consts import Tasks


def test(model, dataloader: DataLoader, threshold=0.5):
    # test
    param = get_param()
    loss_func = nn.MSELoss(reduce=False)
    device = get_param().device

    predicted_tensor = None
    ground_sensor_tensor = []
    ground_labels_tensor = []
    loss_tensor = []

    model.eval()

    acu_loss = 0
    t = tqdm.tqdm(dataloader, desc="TESTING ", leave=False, position=1)
    for windowed_x, y, next_label, edge_index in t:

        with torch.no_grad():
            predicted = model(
                windowed_x.to(device, non_blocking=True),
                edge_index.to(device, non_blocking=True),
            )
            if param.task is Tasks.next_sensors:
                y_truth = y.to(param.device, non_blocking=True)
            elif param.task is Tasks.next_label:
                y_truth = next_label.to(param.device, non_blocking=True)
                predicted = predicted.squeeze(-1)
            loss = loss_func(predicted, y_truth)
            # loss = loss.mean(-1)

            # next_label = next_label.unsqueeze(1).repeat(1, predicted.shape[1])

            if predicted_tensor is None:
                predicted_tensor = predicted
                ground_sensor_tensor = y
                ground_labels_tensor = next_label
                loss_tensor = loss
            else:
                predicted_tensor = torch.cat((predicted_tensor, predicted), dim=0)
                ground_sensor_tensor = torch.cat((ground_sensor_tensor, y), dim=0)
                ground_labels_tensor = torch.cat(
                    (ground_labels_tensor, next_label), dim=0
                )
                loss_tensor = torch.cat((loss_tensor, loss), dim=0)
            acu_loss += loss.sum()

    t.close()

    avg_loss = acu_loss / dataloader.dataset.__len__()

    return avg_loss, [
        predicted_tensor,
        ground_sensor_tensor,
        ground_labels_tensor,
        loss_tensor,
    ]
