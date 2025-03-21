# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset

from sklearn.preprocessing import MinMaxScaler

from util.env import (
    get_device,
    set_device,
    TensorBoardPath,
    createTensorBoardPath,
    getSnapShotPath,
    clearPAths,
)
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fully_connected_graph_struc
from util.iostream import printsep

from datasets.TimeDataset import TimeDataset
from sklearn.preprocessing import MinMaxScaler

from models.GDN import GDN
import pickle
from train import train
from test import test
from evaluate import (
    get_err_scores,
    get_best_performance_data,
    get_val_performance_data,
    get_full_err_scores,
)

import sys
from datetime import datetime

import os
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import random
from util.preprocess import findSensorActuator
from torch.utils.tensorboard.writer import SummaryWriter
import itertools


class Main:
    def __init__(self, train_config, env_config, debug=False, scale=True):

        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None

        dataset = self.env_config["dataset"]
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

        train, test = train_orig, test_orig

        if "attack" in train.columns:
            train = train.drop(columns=["attack"])

        feature_map = get_feature_map(dataset)
        fc_struc = get_fully_connected_graph_struc(dataset)

        set_device(env_config["device"])
        self.device = get_device()

        fc_edge_index = build_loc_net(
            fc_struc, list(train.columns), feature_map=feature_map
        )
        fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)

        self.feature_map = feature_map

        train_dataset_indata = construct_data(train, feature_map, labels=0)
        test_dataset_indata = construct_data(
            test, feature_map, labels=test.attack.tolist()
        )

        cfg = {
            "slide_win": train_config["slide_win"],
            "slide_stride": train_config["slide_stride"],
        }

        train_dataset = TimeDataset(
            train_dataset_indata, fc_edge_index, mode="train", config=cfg
        )
        test_dataset = TimeDataset(
            test_dataset_indata, fc_edge_index, mode="test", config=cfg
        )

        train_dataloader, val_dataloader = self.get_loaders(
            train_dataset,
            train_config["seed"],
            train_config["batch"],
            val_ratio=train_config["val_ratio"],
        )

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=train_config["batch"],
            shuffle=False,
            num_workers=1,
            pin_memory=False,
        )

        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)

        self.model = GDN(
            edge_index_sets,
            len(feature_map),
            dim=train_config["dim"],
            input_dim=train_config["slide_win"],
            out_layer_num=train_config["out_layer_num"],
            out_layer_inter_dim=train_config["out_layer_inter_dim"],
            topk=train_config["topk"],
        ).to(self.device)

    def run(self):

        if len(self.env_config["load_model_path"]) > 0:
            model_save_path = self.env_config["load_model_path"]
        else:
            model_save_path = self.env_config["save_path"] + "/best.pt"

            self.train_log = train(
                self.model,
                model_save_path,
                config=train_config,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
                feature_map=self.feature_map,
                test_dataloader=self.test_dataloader,
                test_dataset=self.test_dataset,
                train_dataset=self.train_dataset,
                dataset_name=self.env_config["dataset"],
            )

        # test
        w = SummaryWriter(TensorBoardPath())

        best_model = self.model.to(self.device)
        _, self.test_result = test(best_model, self.test_dataloader)
        _, self.val_result = test(best_model, self.val_dataloader)
        w.add_pr_curve(
            tag="pr",
            labels=np.array(self.val_result[1]),
            predictions=np.array(self.val_result[0]),
        )
        scores = self.get_score(self.test_result, self.val_result)
        w.add_hparams(hparam_dict=self.train_config, metric_dict=scores)
        w.flush()

    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1):
        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)
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
            batch_size=batch,
            shuffle=True,
            pin_memory=False,
            num_workers=1,
        )

        val_dataloader = DataLoader(
            val_subset, batch_size=batch, shuffle=False, pin_memory=False, num_workers=1
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
        if self.env_config["report"] == "best":
            info = top1_best_info
        elif self.env_config["report"] == "val":
            info = top1_val_info

        # print(f"F1 score: {info[0]}")
        # print(f"precision: {info[1]}")
        # print(f"recall: {info[2]}\n")
        # w=SummaryWriter(TensorBoardPath())
        # table="| F1 | precision | recall |\n|----|-----------|--------|\n"
        # table=f"{table}|  {info[0]:.2f}  |     {info[1]:.2f}      |   {info[2]:.2f}     |"
        # w.add_text("scores",table,0)
        # w.flush()
        return {"F1": info[0], "precision": info[1], "recall": info[2]}

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
    parser = argparse.ArgumentParser()

    parser.add_argument("-batch", help="batch size", type=int, default=128)
    parser.add_argument("-epoch", help="train epoch", type=int, default=50)
    parser.add_argument("-slide_win", help="slide_win", type=int, default=10)
    parser.add_argument("-dim", help="dimension", type=int, default=64)
    parser.add_argument("-slide_stride", help="slide_stride", type=int, default=1)
    parser.add_argument(
        "-save_path_pattern", help="save path pattern", type=str, default=""
    )
    parser.add_argument("-dataset", help="wadi / swat", type=str, default="batadal")
    parser.add_argument("-device", help="cuda / cpu", type=str, default="cuda")
    parser.add_argument("-random_seed", help="random seed", type=int, default=0)
    parser.add_argument("-comment", help="experiment comment", type=str, default="")
    parser.add_argument("-out_layer_num", help="outlayer num", type=int, default=1)
    parser.add_argument(
        "-out_layer_inter_dim", help="out_layer_inter_dim", type=int, default=256
    )
    parser.add_argument("-decay", help="decay", type=float, default=0)
    parser.add_argument("-val_ratio", help="val ratio", type=float, default=0.1)
    parser.add_argument("-topk", help="topk num", type=int, default=16)
    parser.add_argument("-report", help="best / val", type=str, default="best")
    parser.add_argument(
        "-load_model_path", help="trained model path", type=str, default=""
    )
    args = parser.parse_args()
    batches = [64, 128]
    windows = [8, 10, 12]
    dims = [32, 64, 128]
    topks = [12, 14, 16, 18]
    out_layers = [128, 256]

    # batches=[64]
    # windows=[8]
    # dims=[32]
    # topks=[12]
    # out_layers=[128]

    grid = itertools.product(batches, windows, dims, topks, out_layers)
    for i, (batch, window, dim, topk, out_layer) in enumerate(grid):
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
        args.batch = batch
        args.slide_win = window
        args.dim = dim
        args.topk = topk
        args.out_layer_inter_dim = out_layer

        createTensorBoardPath(args.dataset)
        print(TensorBoardPath())
        args.save_path_pattern = getSnapShotPath()
        args.comment = args.dataset
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ["PYTHONHASHSEED"] = str(args.random_seed)

        train_config = {
            "batch": args.batch,
            "epoch": args.epoch,
            "slide_win": args.slide_win,
            "dim": args.dim,
            "slide_stride": args.slide_stride,
            "comment": args.comment,
            "seed": args.random_seed,
            "out_layer_num": args.out_layer_num,
            "out_layer_inter_dim": args.out_layer_inter_dim,
            "decay": args.decay,
            "val_ratio": args.val_ratio,
            "topk": args.topk,
        }

        env_config = {
            "save_path": args.save_path_pattern,
            "dataset": args.dataset,
            "report": args.report,
            "device": args.device,
            "load_model_path": "",
        }
        with open(f"{getSnapShotPath()}/args.pickle", "wb") as file:
            # Serialize and save the object to the file
            pickle.dump(args, file)
        main = Main(train_config, env_config, debug=False)
        main.run()
        clearPAths()
