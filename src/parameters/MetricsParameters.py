from parameters.BaseParameter import BaseParameter
import torch


class MetricsParameters(BaseParameter):
    """docstring for MetricsParameters."""

    def __init__(
        self,
        TP: float = 0.0,
        FP: float = 0.0,
        TN: float = 0.0,
        FN: float = 0.0,
        FPR: float = 0.0,
        accuracy: float = 0.0,
        precision: float = 0.0,
        recall: float = 0.0,
        # F1: float = 0.0,
        loss: float = 0.0,
    ):
        super(MetricsParameters, self).__init__()

        self._TP = TP
        self._FP = FP
        self._TN = TN
        self._FN = FN
        self._FPR = FPR
        self._accuracy = accuracy
        self._precision = precision
        self._recall = recall
        # self._F1 = F1
        self._loss = loss

    @property
    def TP(self) -> float:
        """Gets or sets the TP."""
        return self._TP

    @TP.setter
    def TP(self, value: float):
        self._TP = value

    @property
    def FP(self) -> float:
        """Gets or sets the FP."""
        return self._FP

    @FP.setter
    def FP(self, value: float):
        self._FP = value

    @property
    def FPR(self) -> float:
        """Gets or sets the FPR."""
        return self._FPR

    @FPR.setter
    def FPR(self, value: float):
        self._FPR = value

    @property
    def TN(self) -> float:
        """Gets or sets the TN."""
        return self._TN

    @TN.setter
    def TN(self, value: float):
        self._TN = value

    @property
    def FN(self) -> float:
        """Gets or sets the FN."""
        return self._FN

    @FN.setter
    def FN(self, value: float):
        self._FN = value

    @property
    def accuracy(self) -> float:
        """Gets or sets the accuracy."""
        return self._accuracy

    @accuracy.setter
    def accuracy(self, value: float):
        self._accuracy = value

    @property
    def precision(self) -> float:
        """Gets or sets the precision."""
        return self._precision

    @precision.setter
    def precision(self, value: float):
        self._precision = value

    @property
    def recall(self) -> float:
        """Gets or sets the recall."""
        return self._recall

    @recall.setter
    def recall(self, value: float):
        self._recall = value

    @property
    def F1(self) -> float:
        """Gets or sets the F1."""
        if (self.precision + self.recall) != 0:
            return 2 * (self.precision * self.recall) / (self.precision + self.recall)
        else:
            return 0.0

    @F1.setter
    def F1(self, value: float):
        self._F1 = value

    @property
    def loss(self) -> float:
        """Gets or sets the loss."""
        return self._loss

    @loss.setter
    def loss(self, value: float):
        if isinstance(value, torch.Tensor):
            self._loss = value.item()
        else:
            self._loss = value

    def loadFromConfusion(self, metrics_tensor):
        self.TP = metrics_tensor[1][1].item()
        self.FN = metrics_tensor[1][0].item()
        self.TN = metrics_tensor[0][0].item()
        self.FP = metrics_tensor[0][1].item()
        # Accuracy
        self.accuracy = (self.TP + self.TN) / metrics_tensor.sum().item()

        # Precision
        self.precision = (
            self.TP / (self.TP + self.FP) if (self.TP + self.FP) != 0 else 0.0
        )

        # Recall
        self.recall = self.TP / (self.TP + self.FN) if (self.TP + self.FN) != 0 else 0.0

        self.FPR = self.FP / (self.TN + self.FP) if (self.TN + self.FP) != 0 else 0.0
