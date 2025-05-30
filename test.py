from main import Main, DatasetLoader
from parameters.params import Params
from util.consts import Datasets, Models
from util.env import set_param

param = Params()
set_param(param)
param.dataset = Datasets.batadal
param.model = Models.my_mstgat
main = Main(param=param, dataset_loader=DatasetLoader.findModality)
