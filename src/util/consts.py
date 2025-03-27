from enum import Enum
import importlib


class Datasets(Enum):
    swat = "swat"
    batadal = "batadal"
    msl = "msl"
    wadi = "wadi"
    dummy = "dummy"


class Tasks(Enum):
    next_label = 1
    next_sensors = 2
    current_labels = 3


class Models(Enum):
    gdn = "models.GDN.GDN"

    def getClass(self):

        # Split the string into module and class names
        module_path, class_name = self.value.rsplit(".", 1)

        # Dynamically import the module (including submodule)
        module = importlib.import_module(module_path)

        # Get the class from the module
        return getattr(module, class_name)
