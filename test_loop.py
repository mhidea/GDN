import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time

import torch.utils
from util.time import *
from util.env import *

import argparse
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F
import tqdm

from util.data import *
from util.preprocess import *
from util.env import get_param


def test(model, dataloader: DataLoader):
    # test
    loss_func = nn.MSELoss(reduction="mean")
    device = get_param().device

    test_loss_list = []
    now = time.time()

    test_predicted_list = []
    test_ground_list = []
    test_labels_list = []

    t_test_predicted_list = []
    t_test_ground_list = []
    t_test_labels_list = []

    test_len = len(dataloader)

    model.eval()

    acu_loss = 0
    t = tqdm.tqdm(dataloader, desc="TESTING ", leave=False, position=1)
    for windowed_x, y, next_label, edge_index in t:
        # x, y, labels, edge_index = [
        #     item.to(device) for item in [x, y, labels, edge_index]
        # ]

        with torch.no_grad():
            predicted = model(
                windowed_x.to(device, non_blocking=True),
                edge_index.to(device, non_blocking=True),
            )
            loss = loss_func(predicted, y.to(device, non_blocking=True))

            next_label = next_label.unsqueeze(1).repeat(1, predicted.shape[1])

            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = predicted
                t_test_ground_list = y
                t_test_labels_list = next_label
            else:
                t_test_predicted_list = torch.cat(
                    (t_test_predicted_list, predicted), dim=0
                )
                t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, next_label), dim=0)

        test_loss_list.append(loss.item())
        acu_loss += loss.item()

        # if i % 10000 == 1 and i > 1:
        #     print(timeSincePlus(now, i / test_len))
    t.close()
    test_predicted_list = t_test_predicted_list.tolist()
    test_ground_list = t_test_ground_list.tolist()
    test_labels_list = t_test_labels_list.tolist()

    avg_loss = sum(test_loss_list) / len(test_loss_list)

    return avg_loss, [test_predicted_list, test_ground_list, test_labels_list]
