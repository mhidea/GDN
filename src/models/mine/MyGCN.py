import torch
from torch_geometric.utils import dense_to_sparse
from models.BaseModel import BaseModel, GraphLearner
from torch_geometric.nn.models import GCN, GAT, MLP
from torch_geometric.data import Data, Batch


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


class GATEncoder(BaseModel):
    """docstring for GATEncoder."""

    def __init__(self, **kwargs):
        super(GATEncoder, self).__init__(**kwargs)
        self.graph_learner = GraphLearner(
            num_nodes=self.node_num,
            num_embeddings=self.param.embedding_dimension,
            top_k=self.param.topk,
        )
        self.gat = GAT(
            in_channels=self.param.window_length,
            hidden_channels=self.param.out_layer_inter_dim,
            num_layers=self.param.out_layer_num,
            out_channels=1,
        )
        self.dt = torch.nn.Parameter(torch.ones(1))

    def loss(self, x: torch.Tensor, y_truth):

        batch = x.shape[0]
        trainable_adj = self.graph_learner()
        index, weight = dense_to_sparse(trainable_adj)
        data_list_attention = [
            Data(x=x_, edge_index=index, edge_weight=weight) for x_ in x
        ]
        batch_attention = Batch.from_data_list(data_list_attention)
        x_prime = self.gat(
            batch_attention.x,
            edge_index=batch_attention.edge_index,
            edge_weight=batch_attention.edge_weight,
        ).view(batch, self.node_num)

        x_next = x[:, :, -1] + x_prime * self.dt

        return torch.nn.functional.mse_loss(
            x_next.squeeze(-1), y_truth, reduction="none"
        )


class GATEncoderModal(BaseModel):
    """docstring for GATEncoderModal."""

    def __init__(self, **kwargs):
        super(GATEncoderModal, self).__init__(**kwargs)

        self.adj = self.adj.to(device=self.param.device)

        self.inter_index, self.inter_weigth = dense_to_sparse(self.adj.detach().clone())
        self.inter_index.requires_grad = False
        self.inter_weigth.requires_grad = False

        self.intra_index, self.intra_weigth = dense_to_sparse(
            (1 - self.adj).detach().clone()
        )
        self.intra_index.requires_grad = False
        self.intra_weigth.requires_grad = False
        self.graph_learner = GraphLearner(
            num_nodes=self.node_num,
            num_embeddings=self.param.embedding_dimension,
            top_k=self.param.topk,
        )
        self.gat = GAT(
            in_channels=self.param.window_length,
            hidden_channels=self.param.out_layer_inter_dim,
            num_layers=self.param.out_layer_num,
            out_channels=1,
        )

        self.gat_modal = GAT(
            in_channels=self.param.window_length,
            hidden_channels=self.param.out_layer_inter_dim,
            num_layers=self.param.out_layer_num,
            out_channels=1,
        )
        self.gat_antimodal = GAT(
            in_channels=self.param.window_length,
            hidden_channels=self.param.out_layer_inter_dim,
            num_layers=self.param.out_layer_num,
            out_channels=1,
        )

        self.mlp = MLP(
            in_channels=3,
            out_channels=1,
            num_layers=1,
            bias=False,
            act=None,
        )
        self.dt = torch.nn.Parameter(torch.ones(1))

    def loss(self, x: torch.Tensor, y_truth):

        batch = x.shape[0]
        trainable_adj = self.graph_learner()
        index, weight = dense_to_sparse(trainable_adj)
        data_list_attention = [
            Data(x=x_, edge_index=index, edge_weight=weight) for x_ in x
        ]
        batch_attention = Batch.from_data_list(data_list_attention)
        x1 = self.gat(
            batch_attention.x,
            edge_index=batch_attention.edge_index,
            edge_weight=batch_attention.edge_weight,
        ).view(batch, self.node_num, 1)

        data_list_inter = [
            Data(x=x_, edge_index=self.inter_index, edge_weight=self.inter_weigth)
            for x_ in x
        ]
        batch_inter = Batch.from_data_list(data_list_inter)
        x2 = self.gat_modal(
            batch_inter.x,
            edge_index=batch_inter.edge_index,
            edge_weight=batch_inter.edge_weight,
        ).view(batch, self.node_num, 1)
        data_list_intra = [
            Data(x=x_, edge_index=self.intra_index, edge_weight=self.intra_weigth)
            for x_ in x
        ]
        batch_intra = Batch.from_data_list(data_list_intra)
        x3 = self.gat_antimodal(
            batch_intra.x,
            edge_index=batch_intra.edge_index,
            edge_weight=batch_intra.edge_weight,
        ).view(batch, self.node_num, 1)

        x_prime = self.mlp(torch.concat((x1, x2, x3), dim=-1)).squeeze(-1)

        x_next = x[:, :, -1] + x_prime * self.dt

        return torch.nn.functional.mse_loss(
            x_next.squeeze(-1), y_truth, reduction="none"
        )
