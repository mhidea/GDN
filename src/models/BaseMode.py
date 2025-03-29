import torch
from util.consts import Tasks


class BaseModel(torch.nn.Module):
    """docstring for BaseModel."""

    def __init__(
        self,
        node_num,
        task: Tasks,
    ):
        super(BaseModel, self).__init__()
        self.node_num = node_num
        self.task = task
        if self.task == Tasks.next_label:
            self.nodes_to_label = torch.nn.Linear(self.node_num, 1)

    def pre_forward(self, data: torch.Tensor, org_edge_index) -> torch.Tensor:
        if data.dim() == 3:
            return torch.randn([data.shape[0], self.node_num])
        return torch.randn(self.node_num)

    def forward(self, data, org_edge_index):
        out = self.pre_forward(data, org_edge_index)
        # Out shape is (batch,nodes=sensors,windows)

        if self.task == Tasks.next_sensors:
            out = out.squeeze(-1)
        elif self.task == Tasks.next_label:
            out = out.squeeze(-1)
            out = self.nodes_to_label(out)
            out = torch.nn.functional.sigmoid(out)
        return out
