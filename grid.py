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
    param.task = Tasks.next_features
    param.dataset = Datasets.batadal
    param.device = "cuda" if torch.cuda.is_available() else "cpu"
    param.epoch = 40

    # Creating grid search
    batches = [128]
    windows = [5, 10, 15]
    dims = [64, 128]
    topks = [10, 12, 14]
    out_layers = [64]

    # batches=[64]
    # windows=[8]
    # dims=[32]
    # topks=[12]
    # out_layers=[128]
    grid = itertools.product(batches, windows, dims, topks, out_layers)
    createPaths(param.model, param.dataset)
    createWriter(TensorBoardPath())
    grid_strig = "# Grid Params"
    grid_strig += "\n\n- *batches*: " + " , ".join([str(item) for item in batches])
    grid_strig += "\n\n- *windows*: " + " , ".join([str(item) for item in windows])
    grid_strig += "\n\n- *dims*: " + " , ".join([str(item) for item in dims])
    grid_strig += "\n\n- *topks*: " + " , ".join([str(item) for item in topks])
    grid_strig += "\n\n- *out_layers*: " + " , ".join(
        [str(item) for item in out_layers]
    )
    grid_strig += (
        f"\n\n## TensorBoard Path \n\n```console\n\n{TensorBoardPath()}\n\n```"
    )
    getWriter().add_text(tag="Grid", text_string=grid_strig, global_step=0)
    for i, (batch, window, dim, topk, out_layer) in enumerate(grid):
        setTag(i)
        print(
            i,
            "batch",
            batch,
            "window",
            window,
            "dim",
            dim,
            "topk",
            topk,
            "out_layer",
            out_layer,
        )
        param.batch = batch
        param.window_length = window
        param.embedding_dimension = dim
        param.topk = topk
        param.out_layer_inter_dim = out_layer
        param.save_path = getSnapShotPath()

        print(param.summary())
        main = Main(param, debug=False)

        # main.profile()
        # continue

        scores = main.run(i)
