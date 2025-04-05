import torch
from util.consts import Tasks
from util.env import get_param


class BaseModel(torch.nn.Module):
    """docstring for BaseModel."""

    def __init__(self, node_num, **kwargs):
        super(BaseModel, self).__init__()
        self.param = get_param()
        self.node_num = node_num
        self.task = self.param.task
        if self.task == Tasks.next_label:
            self.nodes_to_label = torch.nn.Linear(self.node_num, 1)

    def pre_forward(self, data: torch.Tensor) -> torch.Tensor:
        if data.dim() == 3:
            return torch.randn([data.shape[0], self.node_num])
        return torch.randn(self.node_num)

    def getParmeters() -> dict:
        return {}

    def forward(self, data):
        out = self.pre_forward(data)
        assert out.shape[-2] == self.node_num
        # Out shape is (batch,nodes=sensors,windows)

        if self.task in [Tasks.next_sensors, Tasks.current_actuators]:
            out = out.squeeze(-1)
        elif self.task in [Tasks.next_label, Tasks.current_label]:
            # TODO: 25/04/04 17:30:49 maybe replace with mean(-1)
            out = out.squeeze(-1)
            out = self.nodes_to_label(out)
            out = torch.nn.functional.sigmoid(out)
        return out
