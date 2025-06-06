from enum import Enum
import importlib


class Datasets(Enum):
    swat = "swat"
    swat_noconst = "swat_noconst"
    swat_filtered = "swat_filtered"
    batadal_noconst = "batadal_noconst"
    batadal = "batadal"
    batadal_filtered = "batadal_filtered"
    msl = "msl"
    wadi = "wadi"
    dummy = "dummy"


class Tasks(Enum):
    """Defines which columns are for input and which is for prediction.

    s = sensors
    a = actuators
    c = constants
    l = labels
    all = sensors + actuators + constants
    next = next
    current = current

    Args:
        Enum (_type_): _description_
    """

    s_next_l = 1
    s_next_s = 2
    s_current_l = 3
    s_current_a = 4
    all_next_l = 5
    sc_next_s = 6
    sa_next_s = 7
    sa_next_sa = 8
    all_next_all = 9


class Models(Enum):
    gdn = "models.GDN.GDN"
    gdn_lstm_start = "models.GDN.GDNLstmStart"
    gnn_tam_relu = "models.gnn_tam.GNN_TAM_RELU"
    gnn_tam = "models.gnn_tam.GNN_TAM"
    my_tr = "models.mine.MyModels.TimeWindowGNN"
    my_fw = "models.mine.MyModels.FeatureWindowGNN"
    my_gat = "models.mine.MyModels.MyGAT"
    my_gat_embd = "models.mine.MyModels.MyGATEmbd"
    my_gnn_lstm = "models.mine.MyModels.GNN_LSTM_AnomalyDetector_Optimized"
    my_gcn = "models.mine.MyGCN.MyGCN"
    my_mstgat = "models.mine.MSTGAT.MSTGAT"
    my_mstgat2 = "models.mine.MSTGAT2.MSTGAT2"
    diffpool = "models.DiffPool.DiffPool"
    diffpool_lstm_start = "models.DiffPool.DiffPoolLstmStart"

    def getClass(self):

        # Split the string into module and class names
        module_path, class_name = self.value.rsplit(".", 1)

        # Dynamically import the module (including submodule)
        module = importlib.import_module(module_path)

        # Get the class from the module
        return getattr(module, class_name)
