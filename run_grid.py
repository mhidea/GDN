from main import Main
from util.consts import *
from util.env import *
import itertools

import torch
import pickle

if __name__ == "__main__":

    # Initializing parameters
    param = Params()
    set_param(param)
    param.task = Tasks.next_label
    param.dataset = Datasets.dummy
    param.model = Models.gdn
    param.device = "cuda" if torch.cuda.is_available() else "cpu"
    createPaths(param.model, param.dataset)
    # Creating grid search
    grid = {
        "epoch": [2],
        "batch": [2],
        "window_length": [5, 6],
        "embedding_dimension": [64],
        "topk": [2, 3],
        "out_layer_inter_dim": [64],
        "osso": [2, 3],
    }

    createWriter(TensorBoardPath())

    # create a text for list of parameters in the grid
    grid_strig = "# Grid Params"
    grid_strig += "\n\n".join(
        [
            f"\n\n- *{key}*: " + " , ".join([str(value) for value in grid[key]])
            for key in grid.keys()
        ]
    )
    grid_strig += (
        f"\n\n## TensorBoard Path \n\n```console\n\n{TensorBoardPath()}\n\n```"
    )
    getWriter().add_text(tag="Grid", text_string=grid_strig, global_step=0)

    _g = [[{key: v} for v in grid[key]] for key in grid.keys()]
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

        # main.profile()
        # continue

        main.run(i)
