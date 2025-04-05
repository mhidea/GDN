import numpy as np
from sympy import false
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
import torch.utils
from test_loop import *
import torch.nn.functional as F
import numpy as np
from evaluate import createMetrics, createStats
from torch.utils.data import DataLoader, random_split, Subset
from scipy.stats import iqr
import os
import tqdm
from util.time import *
from util.env import *
from util.consts import Tasks
from models.BaseModel import BaseModel

# def loss_func(y_pred, y_true):
#     loss = F.mse_loss(y_pred, y_true, reduction="sum")

#     return loss

filter_keis = ["TP", "FP", "TN", "FN"]


def train(
    model: BaseModel = None,
    train_dataloader=None,
    val_dataloader=None,
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

    acu_loss = 0
    min_loss = 1e8

    epoch = param.epoch
    early_stop_win = 15
    model.to(param.device)

    stop_improve_count = 0

    dataloader = train_dataloader

    acu_loss = 0
    loss_func = torch.nn.L1Loss(reduction="none")  # param.loss_function()

    for i_epoch in range(epoch):

        t = tqdm.tqdm(
            dataloader,
            desc="epoc {} / {}".format((i_epoch + 1), epoch),
            postfix="{loss}",
            position=0,
        )
        acu_loss = 0
        model.train()
        i = len(dataloader)
        total_samples = dataloader.dataset.__len__()
        avg_loss = 0
        threshold = 0
        all_losses = torch.zeros(
            total_samples, model.node_num, device=param.device, requires_grad=False
        )
        for b, (windowed_x, y, next_label) in enumerate(t):
            i -= 1
            y_truth = param.y_truth(y, next_label)
            optimizer.zero_grad(set_to_none=True)
            out: torch.Tensor = model(windowed_x)
            loss: torch.Tensor = loss_func(out, y_truth)
            all_losses[b * param.batch : b * param.batch + windowed_x.shape[0], :] = (
                loss.detach().clone().squeeze(-1)
            )

            if out.dim() == 2:
                loss = loss.sum(-1)
                _m = loss.max()
            else:
                _m = out.max()
            threshold = _m if _m > threshold else threshold
            loss = loss.sum()
            loss.backward()
            optimizer.step()

            acu_loss += loss.item()
            t.set_postfix({"loss": (loss.item() / windowed_x.shape[0])})
            if i == 0:
                avg_loss = acu_loss / total_samples
                t.set_postfix({"loss": avg_loss})
                t.close()
        # use val dataset to judge
        # if param.task is Tasks.next_sensors:
        #     threshold = avg_loss

        stats = createStats(all_losses=all_losses)

        if val_dataloader is not None:
            scores: dict = test(model, val_dataloader, stats=stats)
            val_loss = scores["Loss"]
            stats_dict = {key: scores[key] for key in filter_keis}
            metrics_dict = {
                key: scores[key] for key in scores.keys() if key not in filter_keis
            }

            getWriter().add_scalars(
                main_tag=getTag("val_stats/epoch_"),
                global_step=i_epoch,
                tag_scalar_dict=stats_dict,
            )
            getWriter().add_scalars(
                main_tag=getTag("val_metrics/epoch_"),
                global_step=i_epoch,
                tag_scalar_dict=metrics_dict,
            )
            getWriter().add_scalars(
                main_tag=getTag("loss"),
                global_step=i_epoch,
                tag_scalar_dict={
                    "train": avg_loss,
                    "val": val_loss,
                },
            )

            if val_loss < min_loss:
                torch.save(model.state_dict(), param.best_path())

                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1

            if stop_improve_count >= early_stop_win:
                break

        else:
            if acu_loss < min_loss:
                torch.save(model.state_dict(), param.best_path())
                min_loss = acu_loss
