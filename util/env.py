import torch
import numpy as np
import datetime
import os
_device = None 
_tensorborad_path = None 

def get_device():
    # return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return _device

def set_device(dev):
    global _device
    _device = dev

def init_work(worker_id, seed):
    np.random.seed(seed + worker_id)
def TensorBoardPath():
    return _tensorborad_path
def createTensorBoardPath(model_name):
    global _tensorborad_path
    now=datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    _tensorborad_path=f"./board/{model_name}/{now}"
    if not os.path.exists(_tensorborad_path):
        os.makedirs(_tensorborad_path)
        