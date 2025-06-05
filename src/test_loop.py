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
from torchmetrics.classification import BinaryStatScores
from typing import Tuple
from parameters.StatisticalParameters import StatisticalParameters
from parameters.MetricsParameters import MetricsParameters
from evaluate import MyConfusuion


def test(model, dataloader: DataLoader, confusion: MyConfusuion = None):
    """This is not batched

    Args:
        model (_type_): _description_
        dataloader (DataLoader): Test data loader which is supposed not to be batched.

    Returns:
        tuple: predicted, y, labels, all_losses
    """
    model.eval()
    total_samples = dataloader.dataset.__len__()
    param = get_param()

    all_losses = torch.zeros(
        total_samples, model.node_num, device=param.device, requires_grad=False
    )
    all_ys = torch.zeros(
        total_samples, model.node_num, device=param.device, requires_grad=False
    )
    all_labels = torch.zeros(total_samples, device=param.device, requires_grad=False)
    acu_loss = 0

    t = tqdm.tqdm(dataloader, desc="TESTING ", leave=False, position=1)
    y_truth_index = param.y_truth_index()
    with torch.no_grad():
        # data = (windowed_x, y, labels)
        for b, data in enumerate(t):
            loss: torch.Tensor = model.loss(
                data[0],
                data[y_truth_index],
            )

            all_losses[b * param.batch : b * param.batch + data[0].shape[0], :] = (
                loss.detach().clone().squeeze(-1)
            )
            all_ys[b * param.batch : b * param.batch + data[0].shape[0], :] = (
                data[1].detach().clone().squeeze(-1)
            )
            all_labels[b * param.batch : b * param.batch + data[0].shape[0]] = (
                data[2].detach().clone().squeeze(-1)
            )
            if confusion is not None:
                confusion.update(
                    loss,
                    data[2].squeeze(-1),
                )

    t.close()
    return all_losses, all_ys, all_labels
