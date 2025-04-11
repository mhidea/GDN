from tkinter.dnd import dnd_start
from main import Main
from util.consts import *
from util.env import *
import itertools

import torch
import pickle

if __name__ == "__main__":

    # Initializing parameters
    param = Params()
    param.device = "cuda" if torch.cuda.is_available() else "cpu"
    set_param(param)

    param.task = Tasks.next_sensors
    param.dataset = Datasets.swat
    param.model = Models.gnn_tam
    createPaths(param.model, param.dataset)

    model_parameters = param.model.getClass().getParmeters()
    model_parameters = {key: [model_parameters[key]] for key in model_parameters.keys()}
    print(model_parameters)

    if param.task in [Tasks.current_label]:
        param.window_length = 1

    if param.model == Models.gnn_tam:
        model_parameters = {
            "n_gnn": [1],
            "gsl_type": ["relu", "directed", "unidirected", "undirected", "tanh"],
            "alpha": [0.1],
        }
    elif param.model == Models.my_fw:
        model_parameters = {"sparsification_method": ["topk", "dropout"]}
    elif param.model == Models.diffpool:
        model_parameter = {"max_nodes": [150]}

    # Creating grid search
    # This python dictionary is flexible.you can change the keys as you wish.
    grid = {
        "epoch": [50],
        "batch": [128],
        "window_length": [5, 10],
        "embedding_dimension": [64],
        "topk": [5, 10, 12, 15],
        "out_layer_inter_dim": [64],
        "out_layer_num": [1],
        "stride": [10],
    }
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
        main = Main(param, debug=False, modelParams=model_dict)
        # main.model = torch.compile(main.model)
        # main.profile()
        # continue

        main.run(i)
