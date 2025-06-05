from tkinter.dnd import dnd_start
from main import Main, DatasetLoader
from util.consts import *
from util.env import *
import itertools
from evaluate import IqrThreshold

import torch
import pickle

if __name__ == "__main__":

    # Initializing parameters
    param = Params()
    param.device = "cuda" if torch.cuda.is_available() else "cpu"
    set_param(param)

    # main define task
    param.task = Tasks.next_sensors
    param.dataset = Datasets.batadal_noconst
    param.model = Models.my_mstgat
    param.datasetLoader = DatasetLoader.findModality
    setThreshold(IqrThreshold())
    createPaths(param.model, param.dataset)

    model_parameters = param.model.getClass().getParmeters()
    model_parameters = {key: [model_parameters[key]] for key in model_parameters.keys()}
    print("#### Default model extra parameters : \n", model_parameters)

    # params per specific task
    if param.task in [Tasks.current_label]:
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
    elif param.model in [Models.my_mstgat, Models.my_mstgat2]:
        model_parameters = {"gamma1": [0.5], "gamma2": [0.8], "kernel_size": [16]}

    # Creating grid search
    # This python dictionary is flexible.you can change the keys as you wish.
    grid = {
        "epoch": [60],
        "batch": [32],
        "window_length": [5],
        "embedding_dimension": [32],
        "topk": [30],
        "out_layer_inter_dim": [32],
        "out_layer_num": [2],
        # use stride for dabase summerizatoion . default = 1 (no summerization)
        "stride": [1],
    }

    # merge parameters
    grid = grid | model_parameters  # merge dicts
    createWriter(TensorBoardPath())
    _g = []
    total = 1
    for key in grid.keys():
        temp = [{key: v} for v in grid[key]]
        total *= len(temp)
        _g.append(temp)
    # create a text for list of parameters in the grid
    grid_strig = f"# Grid Params\n\n**Total runs: {total}**"
    grid_strig += "\n\n".join(
        [
            f"\n\n- **{key}**: " + " , ".join([str(value) for value in grid[key]])
            for key in grid.keys()
        ]
    )
    grid_strig += (
        f"\n\n## TensorBoard Path \n\n```console\n\n{TensorBoardPath()}\n\n```"
    )
    getWriter().add_text(tag="Grid", text_string=grid_strig, global_step=0)

    param_keies = param.toDict().keys()
    for i, (x) in enumerate(itertools.product(*_g)):
        setTag(i)
        param.save_path = getSnapShotPath()
        model_dict = {}
        for z in x:
            key = list(z.keys())[0]
            if key in param_keies:
                setattr(param, key, z[key])
            else:
                model_dict = model_dict | z
        print(param.summary(extra_dict=model_dict))
        # adj = torch.zeros((43, 43)).float()
        # modals = [[6, 25, 26, 27, 28], [0, 7, 8, 9, 10], [3, 19, 20]]
        # for modal in modals:
        #     for s1 in modal:
        #         for s2 in modal:
        #             adj[s1][s2] = 1
        adj = None
        main = Main(param, modelParams=model_dict, adj=adj)
        # main.model = torch.compile(main.model)
        # main.profile()
        # continue
        main.run(i)
