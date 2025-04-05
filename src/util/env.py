import numpy as np
import datetime
import os
from util.consts import Datasets, Models
from util.params import Params
from torch.utils.tensorboard.writer import SummaryWriter

_tensorborad_path = None
_snapshot_path = None
_param: Params = None
_writer: SummaryWriter = None
_run_time = None
_tag_suffix = None


def setTag(run: int):
    global _tag_suffix
    _tag_suffix = run


def getTag(name: str):
    global _tag_suffix
    return f"{name}_{_tag_suffix}"


def createWriter(dir: str):
    global _writer
    if _writer is not None:
        _writer.flush()
        _writer.close()
    _writer = SummaryWriter(log_dir=dir)


def getWriter() -> SummaryWriter:
    global _writer
    return _writer


def get_param() -> Params:
    return _param


def set_param(param: Params):
    global _param
    _param = param


def init_work(worker_id, seed):
    np.random.seed(seed + worker_id)


def TensorBoardPath():
    global _tensorborad_path
    return _tensorborad_path


def createPaths(model: Models, dataset: Datasets):
    global _tensorborad_path
    global _snapshot_path
    global _run_time
    if _run_time is None:
        _run_time = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    if _tensorborad_path == None:
        _tensorborad_path = f"./board/{model.name}_{dataset.value}/{_run_time}"
        if not os.path.exists(_tensorborad_path):
            os.makedirs(_tensorborad_path)


def getSnapShotPath():
    global _snapshot_path
    global _tensorborad_path
    global _tag_suffix
    _snapshot_path = _tensorborad_path.replace("board", "snapshot") + f"/{_tag_suffix}"
    if not os.path.exists(_snapshot_path):
        os.makedirs(_snapshot_path)
    return _snapshot_path


def clearPAths():
    global _tensorborad_path
    global _snapshot_path
    _tensorborad_path = None
    _snapshot_path = None
