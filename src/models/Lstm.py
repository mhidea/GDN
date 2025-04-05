import torch
from util.env import get_param


class LstmStart(torch.nn.Module):
    """Apply LSTM anywhere in your model.
    Inputs to forward method is supposed to be a tensor of shape:
        (Batch, nodes, window)
    Normally lstm accepts input of shape:
        (Batch, window, nodes)
    This transformation is done in forward method automatically

    Args:
        torch (_type_): _description_
    """

    def __init__(self, in_num: int, out_num: int = -1):
        super(LstmStart, self).__init__()

        if out_num == -1:
            out_num = get_param().lstm_hidden_dim

        self.lstm = torch.nn.LSTM(
            in_num,
            out_num,
            batch_first=True,
            num_layers=get_param().lstm_layers_num,
        )
        self.lin = torch.nn.Linear(out_num, in_num)

    def forward(self, x):
        x = x.transpose(-1, -2)
        x, _ = self.lstm(x)
        x = self.lin(x)
        x = torch.nn.functional.sigmoid(x)
        x = x.transpose(-1, -2).contiguous()
        # x = x[:, -1, :].unsqueeze(-1)
        return x
