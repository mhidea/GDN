from enum import Enum
import importlib


class Datasets(Enum):
    swat = "swat"
    swat_filtered = "swat_filtered"
    batadal = "batadal"
    batadal_filtered = "batadal_filtered"
    msl = "msl"
    wadi = "wadi"
    dummy = "dummy"


class Tasks(Enum):
    next_label = 1
    next_sensors = 2
    current_label = 3
    current_actuators = 4


class Models(Enum):
    gdn = "models.GDN.GDN"
    gdn_lstm_start = "models.GDN.GDNLstmStart"
    gnn_tam_relu = "models.gnn_tam.GNN_TAM_RELU"
    gnn_tam = "models.gnn_tam.GNN_TAM"
    my_tr = "models.MyModels.TimeWindowGNN"
    my_fw = "models.MyModels.FeatureWindowGNN"
    my_gnn_lstm = "models.MyModels.GNN_LSTM_AnomalyDetector_Optimized"
    diffpool = "models.DiffPool.DiffPool"
    diffpool_lstm_start = "models.DiffPool.DiffPoolLstmStart"

    def getClass(self):

        # Split the string into module and class names
        module_path, class_name = self.value.rsplit(".", 1)

        # Dynamically import the module (including submodule)
        module = importlib.import_module(module_path)

        # Get the class from the module
        return getattr(module, class_name)
