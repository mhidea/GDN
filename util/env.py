import torch
import numpy as np
import datetime
import os

_device = None
_tensorborad_path = None
_snapshot_path = None


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
    if _tensorborad_path == None:
        now = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
        _tensorborad_path = f"./board/{model_name}/{now}"
        if not os.path.exists(_tensorborad_path):
            os.makedirs(_tensorborad_path)


def getSnapShotPath():
    global _tensorborad_path
    global _snapshot_path
    if _snapshot_path == None:
        if _tensorborad_path == None:
            createTensorBoardPath()
        _snapshot_path = _tensorborad_path.replace("board", "snapshot")
        if not os.path.exists(_snapshot_path):
            os.makedirs(_snapshot_path)
    return _snapshot_path


def clearPAths():
    global _tensorborad_path
    global _snapshot_path
    _tensorborad_path = None
    _snapshot_path = None
