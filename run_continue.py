from parameters.params import Params
from util.env import set_param
import torch
import pickle
from main import Main

path = "./snapshot/my_mstgat_batadal_noconst/25_06_05_21_48_33/0/"

param: Params = pickle.load(file=open(f"{path}param.pickle", "rb"))()
print(param.summary())

main = Main(
    param=param,
)
