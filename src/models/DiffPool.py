import os.path as osp
from math import ceil

import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch_geometric.nn import DenseGCNConv as GCNConv, dense_diff_pool
from models.BaseModel import BaseModel
from util.preprocess import fully_connected_nonSparse
from torch_geometric.nn.aggr import LSTMAggregation
from torch.nn import LSTM


class GNN(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, normalize=False, lin=True
    ):
        super(GNN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(GCNConv(in_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        self.convs.append(GCNConv(hidden_channels, out_channels, normalize))
        self.bns.append(torch.nn.BatchNorm1d(out_channels))

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        for step in range(len(self.convs)):
            x = self.convs[step](x, adj, mask)
            x = F.relu(x)
            x = x.view(batch_size, -1, num_nodes)
            x = self.bns[step](x)
            x = x.view(batch_size, num_nodes, -1)

            # x = torch.stack([self.bns[step](slice) for slice in torch.unbind(x, dim=0)])

        return x


class DiffPool(BaseModel):
    """_summary_

    Args:
        BaseModel (_type_): _description_
    """

    def __init__(self, **kwargs):
        super(DiffPool, self).__init__(**kwargs)
        self.adj = fully_connected_nonSparse(self.node_num).to(
            self.param.device, non_blocking=True
        )
        self.max_nodes = kwargs["max_nodes"]

        self.aggr = LSTMAggregation(self.param.window_length, 1)

        num_nodes = ceil(0.25 * self.max_nodes)
        self.gnn1_pool = GNN(1, self.param.out_layer_inter_dim, num_nodes)
        self.gnn1_embed = GNN(
            1,
            self.param.out_layer_inter_dim,
            self.param.out_layer_inter_dim,
        )

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(
            self.param.out_layer_inter_dim, self.param.out_layer_inter_dim, num_nodes
        )
        self.gnn2_embed = GNN(
            self.param.out_layer_inter_dim,
            self.param.out_layer_inter_dim,
            self.param.out_layer_inter_dim,
            lin=False,
        )

        self.gnn3_embed = GNN(
            self.param.out_layer_inter_dim,
            self.param.out_layer_inter_dim,
            self.param.out_layer_inter_dim,
            lin=False,
        )

        self.lin1 = torch.nn.Linear(
            self.param.out_layer_inter_dim, self.param.out_layer_inter_dim
        )
        self.lin2 = torch.nn.Linear(self.param.out_layer_inter_dim, self.node_num)

    def pre_forward(self, x: torch.Tensor, mask=None):
        b, v, w = x.shape
        x = x.view(-1, w)
        x = self.aggr(x, dim=-1)
        x = x.view(b, v, 1)
        s = self.gnn1_pool(x, self.adj, mask)
        x = self.gnn1_embed(x, self.adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, self.adj, s, mask)
        # x_1 = s_0.t() @ z_0
        # adj_1 = s_0.t() @ adj_0 @ s_0

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)  # , l1 + l2, e1 + e2

    def getParmeters() -> dict:
        return {"max_nodes": 150}


class DiffPoolLstm(BaseModel):
    """_summary_

    Args:
        BaseModel (_type_): _description_
    """

    def __init__(self, **kwargs):
        super(DiffPoolLstm, self).__init__(**kwargs)
        self.adj = fully_connected_nonSparse(self.node_num).to(
            self.param.device, non_blocking=True
        )
        self.max_nodes = kwargs["max_nodes"]

        self.lstm = LSTM(
            self.node_num,
            self.node_num,
        )

        num_nodes = ceil(0.25 * self.max_nodes)
        self.gnn1_pool = GNN(1, self.param.out_layer_inter_dim, num_nodes)
        self.gnn1_embed = GNN(
            1,
            self.param.out_layer_inter_dim,
            self.param.out_layer_inter_dim,
        )

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(
            self.param.out_layer_inter_dim, self.param.out_layer_inter_dim, num_nodes
        )
        self.gnn2_embed = GNN(
            self.param.out_layer_inter_dim,
            self.param.out_layer_inter_dim,
            self.param.out_layer_inter_dim,
            lin=False,
        )

        self.gnn3_embed = GNN(
            self.param.out_layer_inter_dim,
            self.param.out_layer_inter_dim,
            self.param.out_layer_inter_dim,
            lin=False,
        )

        self.lin1 = torch.nn.Linear(
            self.param.out_layer_inter_dim, self.param.out_layer_inter_dim
        )
        self.lin2 = torch.nn.Linear(self.param.out_layer_inter_dim, self.node_num)

    def pre_forward(self, x: torch.Tensor, mask=None):
        b, v, w = x.shape
        x = x.transpose(-1, -2)
        x, _ = self.lstm(x)
        x = x[:, -1, :].unsqueeze(-1)
        s = self.gnn1_pool(x, self.adj, mask)
        x = self.gnn1_embed(x, self.adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, self.adj, s, mask)
        # x_1 = s_0.t() @ z_0
        # adj_1 = s_0.t() @ adj_0 @ s_0

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)  # , l1 + l2, e1 + e2

    def getParmeters() -> dict:
        return {"max_nodes": 150}
