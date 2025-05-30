import pickle
from parameters.params import Params
from util.env import set_param
from main import Main

path = "./snapshot/gdn_swat/25_03_28_09_47_29/0/"
param: Params = pickle.load(file=open(f"{path}param.pickle", "rb"))
set_param(param)
print(param.task)
scores = Main(param=param).run()
print(scores)
