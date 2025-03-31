import torch
import torch.nn as nn

from models.gsl import GSL
from models.BaseMode import BaseModel
from util.env import *
from util.time import *


class GCLayer(nn.Module):
    """
    Graph convolution layer.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim)

    def forward(self, adj, X):
        adj = adj + torch.eye(adj.size(0)).to(adj.device)
        h = self.dense(X)
        norm = adj.sum(1) ** (-1 / 2)
        h = norm[None, :] * adj * norm[:, None] @ h
        return h


class GNN_TAM(BaseModel):
    """
    Model architecture from the paper "Graph Neural Networks with Trainable
    Adjacency Matrices for Fault Diagnosis on Multivariate Sensor Data".
    https://doi.org/10.1109/ACCESS.2024.3481331
    """

    def __init__(
        self, n_gnn: int = 1, gsl_type: str = "relu", alpha: float = 0.1, **kwargs
    ):
        """
        Args:
            n_nodes (int): The number of nodes/sensors.
            window_size (int): The number of timestamps in one sample.
            n_classes (int): The number of classes.
            n_gnn (int): The number of GNN modules.
            gsl_type (str): The type of GSL block.
            n_hidden (int): The number of hidden parameters in GCN layers.
            alpha (float): Saturation rate for GSL block.
            k (int): The maximum number of edges from one node.
            device (str): The name of a device to train the model. `cpu` and
                `cuda` are possible.
        """
        super(GNN_TAM, self).__init__(**kwargs)
        param = get_param()
        self.window_size = self.param.window_length
        self.nhidden = self.param.out_layer_inter_dim
        self.device = param.device
        self.idx = torch.arange(self.node_num).to(self.device)
        self.adj = [0 for i in range(n_gnn)]
        self.h = [0 for i in range(n_gnn)]
        self.skip = [0 for i in range(n_gnn)]
        self.z = (
            torch.ones(self.node_num, self.node_num) - torch.eye(self.node_num)
        ).to(self.device)
        self.n_gnn = n_gnn

        self.gsl = nn.ModuleList()
        self.conv1 = nn.ModuleList()
        self.bnorm1 = nn.ModuleList()
        self.conv2 = nn.ModuleList()
        self.bnorm2 = nn.ModuleList()

        for i in range(self.n_gnn):
            self.gsl.append(
                GSL(
                    gsl_type,
                    self.node_num,
                    self.param.window_length,
                    alpha,
                    self.param.topk,
                    self.device,
                )
            )
            self.conv1.append(
                GCLayer(self.param.window_length, self.param.out_layer_inter_dim)
            )
            self.bnorm1.append(nn.BatchNorm1d(self.node_num))
            self.conv2.append(
                GCLayer(self.param.out_layer_inter_dim, self.param.out_layer_inter_dim)
            )
            self.bnorm2.append(nn.BatchNorm1d(self.node_num))

        self.fc = nn.Linear(n_gnn * self.param.out_layer_inter_dim, self.node_num)

    def pre_forward(self, X):
        X = X.to(self.device)
        for i in range(self.n_gnn):
            self.adj[i] = self.gsl[i](self.idx)
            self.adj[i] = self.adj[i] * self.z
            self.h[i] = self.conv1[i](self.adj[i], X).relu()
            self.h[i] = self.bnorm1[i](self.h[i])
            self.skip[i], _ = torch.min(self.h[i], dim=1)
            self.h[i] = self.conv2[i](self.adj[i], self.h[i]).relu()
            self.h[i] = self.bnorm2[i](self.h[i])
            self.h[i], _ = torch.min(self.h[i], dim=1)
            self.h[i] = self.h[i] + self.skip[i]

        h = torch.cat(self.h, 1)
        output = self.fc(h)

        return output

    def get_adj(self):
        return self.adj

    def getParmeters():
        return {"n_gnn": 1, "gsl_type": "relu", "alpha": 0.1}


class GNN_TAM_RELU(GNN_TAM):
    def __init__(self, **kwargs):
        super(GNN_TAM_RELU, self).__init__(gsl_type="relu", **kwargs)
