# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset

from sklearn.preprocessing import MinMaxScaler

from util.env import *
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fully_connected_graph_struc

from datasets.TimeDataset import TimeDataset
from sklearn.preprocessing import MinMaxScaler

from models.GDN import GDN
import pickle
from train_loop import train, loss_func
from test_loop import test
from evaluate import (
    get_err_scores,
    get_best_performance_data,
    get_val_performance_data,
    get_full_err_scores,
)

from datetime import datetime

import os
from pathlib import Path

import matplotlib.pyplot as plt

import random
from util.preprocess import findSensorActuator


import itertools
from torch.profiler import profile, record_function, ProfilerActivity
from util.consts import Datasets, Models, Tasks
from util.params import Params


class Main:

    def __init__(self, param: Params, debug=False, scale=True):

        self.param = param
        self.datestr = None

        dataset_name = self.param.dataset.value

        train, test = self._prepareDF(scale, dataset=dataset_name)

        feature_map = get_feature_map(dataset_name)
        fc_struc = get_fully_connected_graph_struc(dataset_name)

        fc_edge_index = build_loc_net(
            fc_struc, list(train.columns), feature_map=feature_map
        )
        fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)

        self.feature_map = feature_map

        train_dataset_indata = construct_data(train, feature_map, labels=0)
        test_dataset_indata = construct_data(
            test, feature_map, labels=test.attack.tolist()
        )

        train_dataset = TimeDataset(
            train_dataset_indata,
            fc_edge_index,
            mode="train",
            param=self.param,
        )
        test_dataset = TimeDataset(
            test_dataset_indata,
            fc_edge_index,
            mode="test",
            param=self.param,
        )

        train_dataloader, val_dataloader = self.get_loaders(train_dataset)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.param.batch,
            shuffle=False,
            pin_memory=False,
        )

        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)

        self.model = self.param.model.getClass()(
            edge_index_sets,
            len(feature_map),
            embeding_dim=self.param.embedding_dimension,
            window_size=self.param.window_length,
            out_layer_num=self.param.out_layer_num,
            out_layer_inter_dim=self.param.out_layer_inter_dim,
            topk=self.param.topk,
        ).to(self.param.device)

    def _prepareDF(self, scale, dataset):
        train_orig = pd.read_csv(f"./data/{dataset}/train.csv", sep=",", index_col=0)
        test_orig = pd.read_csv(f"./data/{dataset}/test.csv", sep=",", index_col=0)
        col, _, _ = findSensorActuator(train_orig)
        print(col)
        if scale:
            # Initialize the MinMaxScaler
            scaler = MinMaxScaler()

            # Fit the scaler on the first dataset
            scaler.fit(train_orig)

            # Transform both datasets using the same scaler
            train_orig[train_orig.columns] = scaler.transform(
                train_orig[train_orig.columns]
            )

            test_orig[train_orig.columns] = scaler.transform(
                test_orig[train_orig.columns]
            )
        if "attack" in train_orig.columns:
            train_orig = train_orig.drop(columns=["attack"])
        return train_orig, test_orig

    def profile(self):
        print(f"Profiling the device {self.param.device}")
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

        self.model.train()
        self.model.cuda()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,
            weight_decay=self.param.decay,
        )
        # profile(f"Kineto ? {torch.profiler.kineto_available()}")
        if not os.path.exists("./profiler"):
            os.mkdir("profiler")
        with torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=1, warmup=4, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            use_cuda=True,
        ) as prof:
            for step, batch_data in enumerate(self.train_dataloader):
                x, labels, attack_labels, edge_index = batch_data

                x, labels, edge_index = [
                    item.float().to(self.param.device)
                    for item in [x, labels, edge_index]
                ]
                if step >= (1 + 4 + 3) * 1:
                    break
                out = (
                    self.model(
                        x.to(self.param.device), edge_index.to(self.param.device)
                    ).float()
                    # .to(get_device())
                )
                loss = loss_func(out, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                prof.step()

    def _loadBestModel(self):
        self.model.load_state_dict(torch.load(open(self.param.best_path, "rb")))

    def run(self):
        wasnt_trained = False
        if not self.param.trained:
            wasnt_trained = True
            print("Model Not Found. Training new model.")
            self.train_log = train(
                model=self.model,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
                feature_map=self.feature_map,
                test_dataloader=self.test_dataloader,
                test_dataset=self.test_dataset,
                train_dataset=self.train_dataset,
                dataset_name=self.param.dataset.value,
            )

        # test
        best_model = self.model.to(self.param.device)
        _, self.test_result = test(best_model, self.test_dataloader)
        _, self.val_result = test(best_model, self.val_dataloader)

        if wasnt_trained:
            # TensorBoardPath()
            # w.add_pr_curve(
            #     tag="pr",
            #     labels=np.array(self.val_result[1]),
            #     predictions=np.array(self.val_result[0]),
            # )
            return self.get_score(self.test_result, self.val_result)

            # w.flush()

    def get_loaders(self, train_dataset):
        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - self.param.val_ratio))
        val_use_len = int(dataset_len * self.param.val_ratio)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat(
            [indices[:val_start_index], indices[val_start_index + val_use_len :]]
        )
        train_subset = Subset(train_dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index : val_start_index + val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)

        train_dataloader = DataLoader(
            train_subset,
            batch_size=self.param.batch,
            shuffle=True,
            pin_memory=False,
        )

        val_dataloader = DataLoader(
            val_subset,
            batch_size=self.param.batch,
            shuffle=False,
            pin_memory=False,
        )

        return train_dataloader, val_dataloader

    def get_score(self, test_result, val_result):
        feature_num = len(test_result[0][0])
        np_test_result = np.array(test_result)
        np_val_result = np.array(val_result)

        test_labels = np_test_result[2, :, 0].tolist()

        test_scores, normal_scores = get_full_err_scores(test_result, val_result)

        top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1)
        top1_val_info = get_val_performance_data(
            test_scores, normal_scores, test_labels, topk=1
        )

        print("=========================** Result **============================\n")

        info = None
        # if self.env_config["report"] == "best":
        if True:
            info = top1_best_info
        elif self.env_config["report"] == "val":
            info = top1_val_info
        return {
            "F1_score": info[0],
            "precision_score": info[1],
            "recall_score": info[2],
        }

    def get_save_path(self, feature_name=""):

        dir_path = self.env_config["save_path"]

        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime("%m_%d_%H_%M_%S")
        datestr = self.datestr

        paths = [
            f"./pretrained/{dir_path}/best_{datestr}.pt",
            f"./results/{dir_path}/{datestr}.csv",
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths


if __name__ == "__main__":

    # Initializing parameters
    param = Params()
    set_param(param)
    param.dataset = Datasets.dummy
    param.device = "cuda" if torch.cuda.is_available() else "cpu"
    param.epoch = 10

    # Creating grid search
    batches = [256]
    windows = [5, 6]
    dims = [64]
    topks = [2]
    out_layers = [64]

    # batches=[64]
    # windows=[8]
    # dims=[32]
    # topks=[12]
    # out_layers=[128]
    grid = itertools.product(batches, windows, dims, topks, out_layers)
    createPaths(param.model, param.dataset)
    createWriter(TensorBoardPath())
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

        # train_config = createTrainConfig(args)
        # env_config = createEnvConfig(args)

        print(param.summary())
        main = Main(param, debug=False)

        # main.profile()
        # continue

        with open(f"{getSnapShotPath()}/param.pickle", "wb") as file:
            # Serialize and save the object to the file
            pickle.dump(param, file)
        scores = main.run()
        if isinstance(scores, dict):
            pass
            getWriter().add_hparams(
                # run_name=f"{param.model.name}_{param.dataset.value}",
                hparam_dict=param.toDict(),
                metric_dict=scores,
                # global_step=i,
            )
        param.save_path = getSnapShotPath()
        getWriter().add_text("summary", param.summary(), global_step=i)
        # clearPAths()
        getWriter().flush()
