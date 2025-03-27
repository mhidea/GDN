import pickle
from util.params import Params
from util.env import set_param
from main import Main

path = "./snapshot/gdn_batadal/25_03_27_23_18_23/0/"
param: Params = pickle.load(file=open(f"{path}param.pickle", "rb"))
set_param(param)
print(param.task)
scores = Main(param=param).run()
print(scores)
