# %%
from models.mine.MSTGAT import MSTGAT
from parameters.params import Params
from util.env import set_param
import torch
from torch_geometric.utils import dense_to_sparse
import tqdm
from datasets.TimeDataset import TimeDataset
from util.consts import Tasks, Datasets
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from util.preprocess import findSensorActuator

param = Params()
param.learning_rate = 0.001
param.batch = 32
param.epoch = 60
param.embedding_dimension = 128
param.topk = 15
kernel_size = 16
gamma1 = 0.5
gamma2 = 0.8
param.task = Tasks.s_next_s
param.dataset = Datasets.batadal
set_param(param)


# %%
def prepareDF(scale, dataset):
    _train_original = pd.read_csv(f"./data/{dataset}/train.csv", sep=",", index_col=0)
    _test_original = pd.read_csv(f"./data/{dataset}/test.csv", sep=",", index_col=0)
    _columns = findSensorActuator(_train_original)
    sensors, actuators, consts = _columns
    print("#####################################")
    print("sensors count: ", len(sensors))
    print("actuators count: ", len(actuators))
    print("consts count: ", len(consts))
    print("consts: ", consts)
    print("#####################################")
    print(sensors)
    if scale:
        scaler = MinMaxScaler()
        # Initialize the MinMaxScaler

        # Fit the scaler on the first dataset
        scaler.fit(_train_original[sensors])

        # Transform both datasets using the same scaler
        _train_original[sensors] = scaler.transform(_train_original[sensors])

        _test_original[sensors] = scaler.transform(_test_original[sensors])
    # if "attack" in _train_original.columns:
    #     _train_original = _train_original.drop(columns=["attack"])
    return _train_original, _test_original, _columns


# %% [markdown]
# ### Load dataset

# %%
train, test, (sensors, actuators, consts) = prepareDF(True, param.dataset.value)

train_dataset = TimeDataset(
    train,
    sensor_list=sensors,
    actuator_list=actuators,
    mode="train",
)


# %% [markdown]
# ### Create dataLoader

# %%
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=param.batch
)

# %% [markdown]
# ### Create model

# %%
node_num = len(sensors)
inter_adj = torch.randint(low=0, high=2, size=(node_num, node_num))
modal_adjacency, _ = dense_to_sparse(torch.eye(node_num))
m = MSTGAT(
    node_num=node_num,
    modal_adjacency=modal_adjacency.cuda(),
    gamma1=gamma1,
    gamma2=gamma2,
    kernel_size=kernel_size,
)

# %% [markdown]
# ### TRAIN

# %%
opt = torch.optim.Adam(m.parameters(), lr=param.learning_rate)
m = m.train().cuda()
total_samples = train_dataloader.dataset.__len__()
avg_loss = 0
for epoch in range(param.epoch):
    t = tqdm.tqdm(
        train_dataloader,
        desc="epoc {} / {}".format((epoch + 1), param.epoch),
        postfix="{loss} {val_loss}",
        position=0,
    )
    i = len(train_dataloader)
    acu_loss = 0
    for b, (windowed_x, y, next_label) in enumerate(t):
        i -= 1
        opt.zero_grad(set_to_none=True)
        loss = m(windowed_x.cuda(), y.cuda())
        loss.backward()
        t.set_postfix(
            {"loss": (loss.item() / windowed_x.shape[0]), "val_loss": avg_loss}
        )
        opt.step()
        acu_loss += loss.detach().clone().item()
        if i == 0:
            avg_loss = acu_loss / total_samples
