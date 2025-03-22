import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time

import torch.utils
import torch.utils
import torch.utils
from util.time import *
from util.env import *
from sklearn.metrics import mean_squared_error
from test import *
import torch.nn.functional as F
import numpy as np
from evaluate import (
    get_best_performance_data,
    get_val_performance_data,
    get_full_err_scores,
)
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from torch.utils.data import DataLoader, random_split, Subset
from scipy.stats import iqr
from torch.utils.tensorboard.writer import SummaryWriter
import os
import tqdm


def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction="mean")

    return loss


def train(
    model=None,
    save_path="",
    config={},
    train_dataloader=None,
    val_dataloader=None,
    feature_map={},
    test_dataloader=None,
    test_dataset=None,
    dataset_name="swat",
    train_dataset=None,
):
    global _tensorborad_path
    seed = config["seed"]

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, weight_decay=config["decay"]
    )

    now = time.time()

    train_loss_list = []
    cmp_loss_list = []

    device = get_device()

    acu_loss = 0
    min_loss = 1e8
    min_f1 = 0
    min_pre = 0
    best_prec = 0

    i = 0
    epoch = config["epoch"]
    early_stop_win = 15

    model.train()

    log_interval = 1000
    stop_improve_count = 0

    dataloader = train_dataloader

    w = SummaryWriter(TensorBoardPath())
    acu_loss = 0
    for i_epoch in range(epoch):

        t = tqdm.tqdm(
            dataloader,
            desc="epoc {} / {}".format((i_epoch + 1), epoch),
            postfix="{loss}",
        )
        acu_loss = 0
        model.train()
        i = 0
        for x, labels, attack_labels, edge_index in t:
            i = i + 1
            _start = time.time()

            x, labels, edge_index = [
                item.float().to(device) for item in [x, labels, edge_index]
            ]

            optimizer.zero_grad()
            out = model(x, edge_index).float().to(device)
            loss = loss_func(out, labels)
            loss.backward()
            optimizer.step()

            train_loss_list.append(loss.item())
            acu_loss += loss.item()
            t.set_postfix({"loss": loss.item()})

            i += 1
        w.add_scalar("loss", acu_loss, i_epoch)
        t.set_postfix({"loss": (acu_loss / len(dataloader))})
        t.close()

        # use val dataset to judge
        if val_dataloader is not None:
            val_loss, val_result = test(model, val_dataloader)
            w.add_scalar("val_loss", val_loss, i_epoch)

            if val_loss < min_loss:
                torch.save(model.state_dict(), save_path)

                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1

            if stop_improve_count >= early_stop_win:
                break

        else:
            if acu_loss < min_loss:
                torch.save(model.state_dict(), save_path)
                min_loss = acu_loss

    return train_loss_list
