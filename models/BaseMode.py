import torch


class BaseClass(torch.nn.Module):
    """docstring for BaseClass."""

    def __init__(self, arg):
        super(BaseClass, self).__init__()

    def pre_forward(self):
        pass

    def forrward(self):
        pass
