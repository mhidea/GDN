from main import Main, DatasetLoader
from parameters.params import Params
from util.consts import Datasets, Models
from util.env import set_param
import numpy as np
import pandas as pd
from util.consts import Tasks
from datasets.TimeDataset import TimeDataset
from util.preprocess import findSensorActuator
from parameters.params import Params
from util.env import set_param

rows = 10
s = {
    "s1": np.arange(0, rows, 1),
    "s2": np.random.randn(rows),
    "s3": np.random.randn(rows),
}

a = {"a1": np.random.randint(0, 2, (rows,)), "a2": np.random.randint(0, 2, (rows,))}
c = {"c1": np.zeros(rows), "c2": np.zeros(rows) + 3}

df = pd.DataFrame(s | a | c)
sg = findSensorActuator(df)
print(sg)
param = Params(window_length=2, stride=1, task=Tasks.s_current_a)
set_param(param)
ds = TimeDataset(df, sg)
print(ds.__getitem__(0))

print(param.summary())
