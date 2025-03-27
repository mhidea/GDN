import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
import torch.utils
from sklearn.metrics import mean_squared_error
from test_loop import *
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
import os
import tqdm
from util.time import *
from util.env import *


def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction="mean")

    return loss


def train(
    model=None,
    train_dataloader=None,
    val_dataloader=None,
    feature_map={},
    test_dataloader=None,
    test_dataset=None,
    dataset_name="swat",
    train_dataset=None,
):
    param = get_param()
    seed = param.random_seed

    optimizer = torch.optim.Adam(
        model.parameters(), lr=param.learning_rate, weight_decay=param.decay
    )

    now = time.time()

    train_loss_list = []
    cmp_loss_list = []

    acu_loss = 0
    min_loss = 1e8
    min_f1 = 0
    min_pre = 0
    best_prec = 0

    i = 0
    epoch = param.epoch
    early_stop_win = 15
    model.to(param.device)

    log_interval = 1000
    stop_improve_count = 0

    dataloader = train_dataloader

    acu_loss = 0
    for i_epoch in range(epoch):

        t = tqdm.tqdm(
            dataloader,
            desc="epoc {} / {}".format((i_epoch + 1), epoch),
            postfix="{loss}",
            position=0,
        )
        acu_loss = 0
        model.train()
        for windowed_x, y, next_label, edge_index in t:
            # _start = time.time()

            # x, labels, edge_index = [
            #     item.to(device, non_blocking=True) for item in [x, labels, edge_index]
            # ]

            out = model(
                windowed_x.to(param.device, non_blocking=True),
                edge_index.to(param.device, non_blocking=True),
            )
            loss = loss_func(out, y.to(param.device, non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss_list.append(loss.item())
            acu_loss += loss.item()
            t.set_postfix({"loss": loss.item()})

        t.set_postfix({"loss": (acu_loss / len(dataloader))})
        t.close()

        # use val dataset to judge
        if val_dataloader is not None:
            val_loss, val_result = test(model, val_dataloader)
            getWriter().add_scalars(
                main_tag=getTag("loss"),
                global_step=i_epoch,
                tag_scalar_dict={
                    "train": (acu_loss / len(dataloader)),
                    "val": val_loss,
                },
            )

            if val_loss < min_loss:
                torch.save(model.state_dict(), param.best_path)

                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1

            if stop_improve_count >= early_stop_win:
                break

        else:
            if acu_loss < min_loss:
                torch.save(model.state_dict(), param.best_path)
                min_loss = acu_loss

    return train_loss_list
