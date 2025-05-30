import torch
from torch_geometric.utils import dense_to_sparse
from models.BaseModel import BaseModel
from torch_geometric.nn.models import GCN


class GSL(torch.nn.Module):
    """This module learns a adjacency matrix using embeddings"""

    def __init__(self, node_num, embedding_dimension, topK=None):
        super(GSL, self).__init__()
        self.embeddings = torch.nn.Embedding(
            node_num, embedding_dim=embedding_dimension
        )
        self.topK = topK

    def getAdjDense(self):
        adj = self.embeddings.weight @ self.embeddings.weight.T
        if self.topK:
            topk_values, topk_indices = torch.topk(adj, self.topK, 1)
            result = torch.zeros_like(adj)
            adj = result.scatter_(1, topk_indices, topk_values)
        return adj

    def getAdj(self):
        adj = self.getAdjDense()
        adj = dense_to_sparse(adj)
        return adj

    def forward(self, x):
        adj = self.getAdj()
        return adj


class MyGCN(BaseModel):
    """docstring for MyGCN."""

    def __init__(self, **kwargs):
        super(MyGCN, self).__init__(**kwargs)
        self.gsl = GSL(
            self.node_num, self.param.embedding_dimension, topK=self.param.topk
        )
        self.gcn = GCN(
            in_channels=self.param.window_length,
            hidden_channels=self.param.out_layer_inter_dim,
            num_layers=self.param.out_layer_num,
            out_channels=1,
        )

    def pre_forward(self, x: torch.Tensor):
        index, weight = self.gsl(x)
        x = self.gcn(x, index, weight)
        return x
