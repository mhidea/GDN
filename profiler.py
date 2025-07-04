import torch
import torch.profiler
from torch.utils.tensorboard import SummaryWriter
from main import Main
from parameters.params import Params
from util.consts import *
from util.env import *
from evaluate import IqrThreshold

# Initializing parameters
param = Params()
param.device = "cuda" if torch.cuda.is_available() else "cpu"
set_param(param)

# main define task
param.task = Tasks.sa_next_sa
param.dataset = Datasets.batadal
param.model = Models.my_mstgat2
param.window_length = 30

setThreshold(IqrThreshold())
createPaths(param.model, param.dataset)

model_parameters = param.model.getClass().getParmeters()
model_parameters = {key: [model_parameters[key]] for key in model_parameters.keys()}
print("#### Default model extra parameters : \n", model_parameters)

# params per specific task
if param.task in [Tasks.s_current_l]:
    param.window_length = 1

if param.model == Models.gnn_tam:
    model_parameters = {
        # "relu", "directed", "unidirected", "undirected", "tanh"
        "gsl_type": ["undirected"],
        "alpha": [0.1],
    }
elif param.model == Models.my_fw:
    model_parameters = {"sparsification_method": ["topk", "dropout"]}
elif param.model == Models.diffpool:
    model_parameters = {"max_nodes": [150]}
elif param.model in [Models.my_mstgat, Models.my_mstgat2, Models.my_mstgat_lstm]:
    model_parameters = {
        "gamma1": 0.5,
        "gamma2": 0.8,
        "kernel_size": 16,
    }
createWriter(TensorBoardPath())
setTag(0)
param.save_path = getSnapShotPath()
with torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=0, warmup=0, active=5, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(TensorBoardPath()),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    main = Main(param, modelParams=model_parameters, adj=None)
    opt = torch.optim.Adam(params=main.model.parameters(), lr=0.001)
    prof.step()
    for x, y, labels in main.train_dataloader:
        opt.zero_grad()
        loss = main.model.loss(x, y)
        loss.mean().backward()
        opt.step()
        prof.step()

getWriter().close()
