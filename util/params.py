from util.consts import Datasets, Models, Tasks
from tabulate import tabulate
import os
import torch
import numpy as np
import random


class Params:
    def __init__(
        self,
        batch: int = 128,
        epoch: int = 50,
        window_length: int = 5,
        embedding_dimension: int = 64,
        stride: int = 1,
        save_path: str = "",
        dataset: Datasets = Datasets.msl,
        random_seed: int = 0,
        out_layer_num: int = 1,
        out_layer_inter_dim: int = 64,
        val_ratio: float = 0.1,
        learning_rate: float = 0.001,
        decay: float = 0.0,
        topk: int = 5,
        model: Models = Models.gdn,
        task: Tasks = Tasks.next_label,
        device: str = "cuda",
    ):
        self._batch = batch
        self._epoch = epoch
        self._window_length = window_length
        self._embedding_dimension = embedding_dimension
        self._stride = stride
        self._save_path = save_path
        self._trained = False
        self._dataset = dataset
        self._random_seed = random_seed
        self._out_layer_num = out_layer_num
        self._out_layer_inter_dim = out_layer_inter_dim
        self._val_ratio = val_ratio
        self._learning_rate = learning_rate
        self._decay = decay
        self._topk = topk
        self._model = model
        self._task = task
        self._device = device

    @property
    def trained(self) -> bool:
        """
        Gets or sets the trained size.
        """
        self._trained = os.path.exists(self.best_path)
        return self._trained

    @property
    def batch(self) -> int:
        """
        Gets or sets the batch size.
        """
        return self._batch

    @batch.setter
    def batch(self, value: int):
        self._batch = value

    @property
    def epoch(self) -> int:
        """
        Gets or sets the number of epochs.
        """
        return self._epoch

    @epoch.setter
    def epoch(self, value: int):
        self._epoch = value

    @property
    def window_length(self) -> int:
        """
        Gets or sets the window length.
        """
        return self._window_length

    @window_length.setter
    def window_length(self, value: int):
        self._window_length = value

    @property
    def embedding_dimension(self) -> int:
        """
        Gets or sets the embedding dimension.
        """
        return self._embedding_dimension

    @embedding_dimension.setter
    def embedding_dimension(self, value: int):
        self._embedding_dimension = value

    @property
    def stride(self) -> int:
        """
        Gets or sets the stride value.
        """
        return self._stride

    @stride.setter
    def stride(self, value: int):
        self._stride = value

    @property
    def best_path(self) -> str:
        """
        Gets or sets the file save path.
        """
        return self._save_path + "/best.pt"

    @property
    def save_path(self) -> str:
        """
        Gets or sets the file save path.
        """
        return self._save_path

    @save_path.setter
    def save_path(self, value: str):
        self._save_path = value

    @property
    def dataset(self) -> Datasets:
        """
        Gets or sets the dataset object.
        """
        return self._dataset

    @dataset.setter
    def dataset(self, value: Datasets):
        self._dataset = value

    @property
    def random_seed(self) -> int:
        """
        Gets or sets the random seed for reproducibility.
        """
        return self._random_seed

    @random_seed.setter
    def random_seed(self, value: int):
        self._random_seed = value
        random.seed(self._random_seed)
        np.random.seed(self._random_seed)
        torch.manual_seed(self._random_seed)
        torch.cuda.manual_seed(self._random_seed)
        torch.cuda.manual_seed_all(self._random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ["PYTHONHASHSEED"] = str(self._random_seed)

    @property
    def out_layer_num(self) -> int:
        """
        Gets or sets the number of output layers.
        """
        return self._out_layer_num

    @out_layer_num.setter
    def out_layer_num(self, value: int):
        self._out_layer_num = value

    @property
    def out_layer_inter_dim(self) -> int:
        """
        Gets or sets the intermediate dimension of the output layer.
        """
        return self._out_layer_inter_dim

    @out_layer_inter_dim.setter
    def out_layer_inter_dim(self, value: int):
        self._out_layer_inter_dim = value

    @property
    def learning_rate(self) -> float:
        """
        Gets or sets the learning rate.
        """
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float):
        self._learning_rate = value

    @property
    def val_ratio(self) -> float:
        """
        Gets or sets the validation ratio.
        """
        return self._val_ratio

    @val_ratio.setter
    def val_ratio(self, value: float):
        self._val_ratio = value

    @property
    def decay(self) -> float:
        """
        Gets or sets the optimizer weight_decay.
        """
        return self._decay

    @decay.setter
    def decay(self, value: float):
        self._decay = value

    @property
    def topk(self) -> int:
        """
        Gets or sets the top-k value.
        """
        return self._topk

    @topk.setter
    def topk(self, value: int):
        self._topk = value

    @property
    def model(self) -> Models:
        """
        Gets or sets the model object.
        """
        return self._model

    @model.setter
    def model(self, value: Models):
        self._model = value

    @property
    def task(self) -> Tasks:
        """
        Gets or sets the task object.
        """
        return self._task

    @task.setter
    def task(self, value: Tasks):
        self._task = value

    @property
    def device(self) -> str:
        """
        Gets or sets the computation device (e.g., 'cpu', 'cuda').
        """

        return self._device

    @device.setter
    def device(self, value: str):
        self._device = value

    def toDict(self) -> dict:
        return {
            "window_length": self.window_length,
            "embedding_dimension": self.embedding_dimension,
            "out_layer_inter_dim": self.out_layer_inter_dim,
            "out_layer_num": self.out_layer_num,
            "batch": self.batch,
            "epoch": self.epoch,
            "stride": self.stride,
            "dataset": self.dataset.name,
            "random_seed": self.random_seed,
            "val_ratio": self.val_ratio,
            "learning_rate": self.learning_rate,
            "decay": self.decay,
            "topk": self.topk,
            "model": self.model.value,
            "task": self.task.name,
            "device": self.device,
        }

    def summary(self, tablefmt="github"):
        """
        Creates a summary table of all properties and their values using tabulate.
        Splits tables into chunks of at most five columns.
        """
        # Gather properties and values into a list
        d = self.toDict()

        # Define headers and prepare tables
        headers = [key for key in d.keys()]
        values = [
            str(d[key]) for key in headers
        ]  # Convert values to strings for display
        single_items = {"save_path": self.save_path}
        tables = []
        # Split into chunks of at most 5 columns
        for i in range(0, len(headers), 5):
            chunk_headers = headers[i : i + 5]
            chunk_values = [values[i : i + 5]]  # Single row for the values
            table = tabulate(
                chunk_values,
                headers=chunk_headers,
                tablefmt=tablefmt,
                stralign="center",
            )
            tables.append(table)

        # Join and return all tables
        summary_output = (
            "\n\n".join([f"# {key}\n\n{val}" for key, val in single_items.items()])
            + "\n\n"
        )

        summary_output += "\n\n".join(tables)
        return summary_output
