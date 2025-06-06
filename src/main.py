# -*- coding: utf-8 -*-
import os
import pickle
import random
import re

import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Subset

# Local Imports
from datasets.TimeDataset import TimeDataset
from parameters.params import Params
from test_loop import test
from train_loop import train
from util.env import getSnapShotPath, getWriter
from util.preprocess import findSensorActuator
from evaluate import MyConfusuion, MetricsParameters
from util.data import sensorGroup_to_xy

# Globals for data caching (consider refactoring later)
_train_original: pd.DataFrame | None = None
_test_original: pd.DataFrame | None = None
_columns = None


class Main:

    def __init__(self, param: Params, modelParams={}, adj=None):
        self.param = param
        self.modelParams = modelParams
        self.scaler = MinMaxScaler()

        print(f"# DATASET \n\n*{self.param.dataset}*")
        train, test = self._load_param_DF(param)
        sensors, actuators, consts = findSensorActuator(train)
        print("#####################################")
        print("sensors count: ", len(sensors))
        print("actuators count: ", len(actuators))
        print("consts count: ", len(consts))
        print("consts: ", consts)
        print("#####################################")
        print(sensors)
        xlist, ylist, next = sensorGroup_to_xy((sensors, actuators, consts), param.task)
        if adj is None:
            adj = self.create_adj(xlist)

        # if self.param.task in [Tasks.s_next_l, Tasks.s_next_s]:
        train_dataset = TimeDataset(
            train,
            column_groups=(sensors, actuators, consts),
            mode="train",
        )
        test_dataset = TimeDataset(
            test,
            column_groups=(sensors, actuators, consts),
            mode="test",
        )
        # elif self.param.task in [Tasks.s_current_l, Tasks.s_current_a]:
        #     train_dataset = CurrentDataset(
        #         train,
        #         sensor_list=sensors,
        #         actuator_list=actuators,
        #         mode="train",
        #     )
        #     test_dataset = CurrentDataset(
        #         test,
        #         sensor_list=sensors,
        #         actuator_list=actuators,
        #         mode="test",
        #     )
        self.num_test_samples = test_dataset.__len__()
        self.num_anomalies = test["attack"].sum()

        train_dataloader, val_dataloader = self.train_validation_loaders(train_dataset)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=param.batch,
            shuffle=False,
            # pin_memory=True,
            # num_workers=4,
        )
        self.model: torch.nn.Module = self.param.model.getClass()(
            adj=adj, **modelParams
        ).to(self.param.device)
        if self.param.trained():
            print("Model is trained. Loading from file .....")
            self.load_model(self.param.best_validationModel_path())
        # self.model = torch.compile(self.model, mode="reduce-overhead")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))

    def create_adj(self, columns: list):
        stripped_columns = [re.sub(r"\d+", "", s) for s in columns]
        adj = torch.zeros(len(stripped_columns), len(stripped_columns))
        for i in range(len(stripped_columns)):
            for j in range(len(stripped_columns)):
                if stripped_columns[i] == stripped_columns[j]:
                    adj[i][j] = 1
        return adj

    def _load_param_DF(self, param: Params):
        global _train_original
        global _test_original
        global _columns
        if _train_original is None:
            _train_original = pd.read_csv(
                f"./data/{param.dataset.value}/train.csv", sep=",", index_col=0
            )
            _test_original = pd.read_csv(
                f"./data/{param.dataset.value}/test.csv", sep=",", index_col=0
            )
            columns = {col: col.strip() for col in _train_original.columns}
            _train_original = _train_original.rename(columns=columns)

            columns = {col: col.strip() for col in _test_original.columns}
            _test_original = _test_original.rename(columns=columns)
            for i, col in enumerate(_train_original.columns):
                # print(_train_original.columns[i], _test_original.columns[i])
                assert _train_original.columns[i] == _test_original.columns[i]
            print(" - ".join(_train_original.columns))
            # Fit the scaler on the first dataset
            self.scaler.fit(_train_original)

            # Transform both datasets using the same scaler
            _train_original[_train_original.columns] = self.scaler.transform(
                _train_original[_train_original.columns]
            )

            _test_original[_train_original.columns] = self.scaler.transform(
                _test_original[_train_original.columns]
            )
        return _train_original, _test_original

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
            os.makedirs("./profiler", exist_ok=True)  # Use makedirs with exist_ok=True
        with torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=1, warmup=4, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler"),
            record_shapes=True,
            profile_memory=True,  # Note: profile_memory adds overhead
            with_stack=True,  # Note: with_stack adds overhead
            use_cuda=(self.param.device == "cuda"),  # Set use_cuda based on device
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
                loss = self.param.loss_function(out, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                prof.step()

    def _loadBestModel(self):
        self.model.load_state_dict(
            torch.load(open(self.param.best_validationModel_path(), "rb"))
        )

    def run(self, step: int = 0):
        wasnt_trained = False
        best_threshold = None
        if not self.param.trained():
            wasnt_trained = True
            print("###############################")
            print("Model Not Found. Training new model.")
            best_threshold = train(
                model=self.model,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
            )
            print("###############################")
        else:
            print("Model already trained. Loading from saved file.")
            self.model.load_state_dict(
                torch.load(self.param.best_validationModel_path(), weights_only=True)
            )
            best_threshold = torch.load(
                self.param.best_validationModel_path().replace(
                    "best.pt", "least_loss_threshold.pt"
                ),
            )

        # Test the best model
        print("###############################")
        print("Testing on test dataset")
        # best_model_losses = test(best_model, self.train_dataloader)
        # best_model_stats = createStats(best_model_losses)
        best_model = self.model.to(self.param.device)
        conf = MyConfusuion(thr=best_threshold).to(device=self.param.device)
        all_losses, _, _ = test(best_model, self.test_dataloader, confusion=conf)
        confusion_matrix = conf.compute()
        metrics = MetricsParameters()
        metrics.loadFromConfusion(confusion_matrix)
        metrics.loss = all_losses.sum(-1).mean()
        print("scores")
        print(
            f"Total test dataset samples: {self.num_test_samples} . Total anomalies (label=1) : {self.num_anomalies} "
        )

        print(metrics.summary())
        # Log results if the model was trained in this run
        if wasnt_trained:
            getWriter().add_hparams(
                # run_name=f"{param.model.name}_{param.dataset.value}",
                hparam_dict=self.param.toDict() | {"step": step} | self.modelParams,
                metric_dict=metrics.toDict(),
                # global_step=i,
            )
            with open(f"{getSnapShotPath()}/param.pickle", "wb") as file:
                # Serialize and save the object to the file
                pickle.dump(self.param, file)
            with open(f"{getSnapShotPath()}/model_parameters.pickle", "wb") as file:
                # Serialize and save the object to the file
                pickle.dump(self.modelParams, file)
            getWriter().add_text(
                "summary",
                best_threshold.summary()
                + "\n\n"
                + metrics.summary()
                + "\n\n"
                + self.param.summary(extra_dict=self.modelParams),
                global_step=step,
            )
            # clearPAths()
            getWriter().flush()

        return metrics

    def train_validation_loaders(self, train_dataset):
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
            # pin_memory=True,
            # num_workers=4,
        )

        val_dataloader = DataLoader(
            val_subset,
            batch_size=self.param.batch,
            shuffle=False,
            # pin_memory=True,
            # num_workers=4,
        )

        return train_dataloader, val_dataloader
