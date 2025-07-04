# %%
import pickle
from parameters.params import Params, Datasets, Models, Tasks
from util.env import set_param
from main import Main
import torch
from models.mine.MSTGAT import MSTGAT
from test_loop import test
from train_loop import train
import pandas as pd
from evaluate import (
    IqrThreshold,
    MyConfusuion,
    IqrSensorThreshold,
    MinMaxThreshold,
    ZscoreThreshold,
)
import scipy.optimize as opt

# %%
path = "./snapshot/my_mstgat_batadal/25_06_16_10_06_41/0/"

param: Params = pickle.load(file=open(f"{path}param.pickle", "rb"))
param.val_ratio = 0
set_param(param)
model_parameters = pickle.load(file=open(f"{path}model_parameters.pickle", "rb"))
adj = torch.load(param.best_validationModel_path().replace("best.pt", "adj.pt"))


# %%
main = Main(param=param, modelParams=model_parameters, adj=adj)

# # main.load_model(path=param.least_trainLossModel_path())

# # %%
# print(len(main.train_dataloader),len(main.val_dataloader),len(main.test_dataloader))

# # %% [markdown]
# # ## Train losses
# # get all losses from train data loader

# # %%
# train_all_loss,y_train,label_train = test(main.model,main.train_dataloader,None)

# # %%
# print(train_all_loss.shape)
# print(label_train.sum())

# # %% [markdown]
# # ## Test losses

# # %%
# test_all_losses,ys,lbls = test(main.model, main.test_dataloader,confusion=None)
