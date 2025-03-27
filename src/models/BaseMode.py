import torch
from util.consts import Tasks


class BaseClass(torch.nn.Module):
    """docstring for BaseClass."""

    def __init__(
        self,
        node_num,
        task: Tasks,
    ):
        super(BaseClass, self).__init__()
        self.node_num = node_num
        self.task = task

    def pre_forward(self) -> torch.Tensor:
        pass

    def forward(self, data, org_edge_index):
        out = self.pre_forward(data, org_edge_index)
        if self.task == Tasks.next_features:
            out = out.view(-1, self.node_num)
        return out
