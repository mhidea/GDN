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
from evaluate import createIrqStats
from torch.utils.data import DataLoader, random_split, Subset
from scipy.stats import iqr
import os
import tqdm
from util.time import *
from util.env import *
from util.consts import Tasks
from models.BaseModel import BaseModel
from parameters.MetricsParameters import MetricsParameters
from evaluate import MyConfusuion

# def loss_func(y_pred, y_true):
#     loss = F.mse_loss(y_pred, y_true, reduction="sum")

#     return loss

filter_keis = ["TP", "FP", "TN", "FN"]


def train(model: BaseModel = None, train_dataloader=None, val_dataloader=None):
    param = get_param()
    seed = param.random_seed

    optimizer = torch.optim.Adam(
        model.parameters(), lr=param.learning_rate, weight_decay=param.decay
    )

    acu_loss = 0
    min_validation_loss = 1e8
    min_train_loss = 1e8

    early_stop_win = 15
    model.to(param.device)

    stop_improve_count = 0

    dataloader = train_dataloader

    acu_loss = 0
    # loss_func = torch.nn.L1Loss(reduction="none")  # param.loss_function()

    y_truth_index = param.y_truth_index()
    best_train_threshold = None
    for i_epoch in range(param.epoch):

        t = tqdm.tqdm(
            dataloader,
            desc="epoc {} / {}".format((i_epoch + 1), param.epoch),
            postfix="{Bloss} {Aculoss} {val_loss}",
            position=0,
        )
        acu_loss = 0
        model.train()
        train_iterations = len(dataloader)
        trained_samples = 0
        avg_loss = 0
        train_all_losses = torch.zeros(
            dataloader.dataset.__len__(),
            1 if param.task in [Tasks.current_label] else model.node_num,
            device=param.device,
            requires_grad=False,
        )
        # data = (windowed_x, y, next_label)
        for b, data in enumerate(t):
            train_iterations -= 1
            batch = data[0].shape[0]
            trained_samples += batch

            loss: torch.Tensor = model.loss(
                data[0],
                data[y_truth_index],
            )

            train_all_losses[trained_samples - batch : trained_samples, :] = (
                loss.narrow_copy(0, 0, batch).squeeze(-1)
            )

            # sum
            loss = loss.sum()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            acu_loss += loss.item()
            t.set_postfix(
                {
                    "Bloss": (loss.item() / batch),
                    "Aculoss": (acu_loss / trained_samples),
                    "val_loss": 0,
                }
            )

            #  Training is finished. Now we validate
            if train_iterations == 0:
                avg_loss = acu_loss / trained_samples

                # Compute the statistics of train losses so that evalution can do the classification
                thr = getThreshold()
                thr.fit(train_all_losses)

                if val_dataloader is not None:
                    conf = MyConfusuion(thr=thr).to(param.device)
                    #     conf = bs.compute()
                    #     result = MetricsParameters()
                    #     result.loadFromConfusion(conf)
                    #     result.loss = (acu_loss) / total_samples

                    # return result
                    all_validation_losses: torch.Tensor = test(
                        model, val_dataloader, confusion=conf
                    )
                    confusion_matrix = conf.compute()
                    metrics = MetricsParameters()
                    metrics.loadFromConfusion(confusion_matrix)
                    metrics.loss = all_validation_losses.sum(-1).mean()
                    val_loss = metrics.loss
                    t.set_postfix(
                        {
                            "Aculoss": (acu_loss / trained_samples),
                            "val_loss": val_loss,
                        }
                    )

                    confusion_metrics = {
                        key: metrics.toDict()[key] for key in filter_keis
                    }
                    _m = metrics.toDict()
                    performance_metrics_dict = {
                        key: _m[key] for key in _m.keys() if key not in filter_keis
                    }

                    # Save the metrics per epoch in tensorboard
                    getWriter().add_scalars(
                        main_tag=getTag("val_confusion/epoch_"),
                        global_step=i_epoch,
                        tag_scalar_dict=confusion_metrics,
                    )
                    getWriter().add_scalars(
                        main_tag=getTag("val_metrics/epoch_"),
                        global_step=i_epoch,
                        tag_scalar_dict=performance_metrics_dict,
                    )
                    getWriter().add_scalars(
                        main_tag=getTag("loss"),
                        global_step=i_epoch,
                        tag_scalar_dict={
                            "train": avg_loss,
                            "val": val_loss,
                        },
                    )

                    # Save the model with the least loss on train data
                    if acu_loss < min_train_loss:
                        torch.save(
                            model.state_dict(),
                            param.best_path().replace("best.pt", "best_train.pt"),
                        )
                        min_train_loss = acu_loss
                        torch.save(
                            thr,
                            param.best_path().replace(
                                "best.pt", "least_loss_threshold.pt"
                            ),
                        )
                        best_train_threshold = thr

                    # Save the model with the least loss on validation data
                    if val_loss < min_validation_loss:
                        torch.save(model.state_dict(), param.best_path())
                        torch.save(
                            thr,
                            param.best_path().replace(
                                "best.pt", "least_val_loss_threshold.pt"
                            ),
                        )

                        min_validation_loss = val_loss
                        stop_improve_count = 0
                    else:
                        stop_improve_count += 1

                    if stop_improve_count >= early_stop_win:
                        break

                else:
                    if acu_loss < min_validation_loss:
                        torch.save(model.state_dict(), param.best_path())
                        min_validation_loss = acu_loss
                t.close()
    return best_train_threshold
