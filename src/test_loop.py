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
from evaluate import createMetrics


def test(model, dataloader: DataLoader, stats: dict = None):
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
    loss_func = torch.nn.L1Loss(reduction="none")

    if stats is None:
        pass
        all_losses = torch.zeros(
            total_samples, model.node_num, device=param.device, requires_grad=False
        )
    else:
        bs = BinaryStatScores().to(param.device)
        acu_loss = 0

    t = tqdm.tqdm(dataloader, desc="TESTING ", leave=False, position=1)
    for b, (windowed_x, y, labels) in enumerate(t):
        with torch.no_grad():
            y_truth = param.y_truth(y, labels)
            predicted = model(windowed_x)
            loss: torch.Tensor = loss_func(predicted, y_truth)
            if stats is None:
                pass
                all_losses[
                    b * param.batch : b * param.batch + windowed_x.shape[0], :
                ] = (loss.detach().clone().squeeze(-1))
            else:
                predicted_labels = (
                    ((loss - stats["medians"]).abs() / stats["iqr"]).max(-1).values
                )
                acu_loss += predicted_labels.sum()
                pred = torch.where(
                    predicted_labels > stats["threshold"],
                    torch.tensor(1),
                    torch.tensor(0),
                )
                bs.update(pred, labels.squeeze(-1))
    if stats is None:
        result = all_losses
    else:
        conf = bs.compute()
        result = createMetrics(conf) | {"Loss": (acu_loss) / total_samples}

    t.close()
    return result
