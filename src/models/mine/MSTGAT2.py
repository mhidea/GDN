import torch
from torch_geometric.nn.models import GCN, GAT, VGAE, MLP
from torch_geometric.nn.conv import GCNConv
from models.BaseModel import BaseModel, GraphLearner
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, Batch
from torch_geometric.utils import negative_sampling
from typing import Optional, Tuple

EPS = 1e-15


class MyVGAE(VGAE):
    """docstring for MyVGAE."""

    def recon_loss(
        self,
        z: torch.Tensor,
        pos_edge_index: torch.Tensor,
        neg_edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        pos_loss = -torch.log(self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean(
            dim=-1
        )

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(
            1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS
        ).mean(dim=-1)

        return pos_loss + neg_loss


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight).relu()
        return self.conv_mu(x, edge_index, edge_weight), self.conv_logstd(
            x, edge_index, edge_weight
        )


class MSTGAT2(BaseModel):
    """docstring for MSTGAT2."""

    def __init__(self, **kwargs):
        super(MSTGAT2, self).__init__(**kwargs)
        self.gamma1 = kwargs["gamma1"]
        self.gamma2 = kwargs["gamma2"]
        kernel_size = kwargs["kernel_size"]

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

        self.lin1 = MLP(
            in_channels=self.param.window_length,
            out_channels=self.param.embedding_dimension,
            num_layers=1,
            bias=False,
            act=None,
        )

        self.multi_head_attention = GAT(
            in_channels=1,
            hidden_channels=self.param.lstm_hidden_dim,
            out_channels=self.param.embedding_dimension,
            num_layers=self.param.out_layer_num,
            act="relu",
        )

        self.inter_modal = GAT(
            in_channels=1,
            hidden_channels=self.param.lstm_hidden_dim,
            out_channels=self.param.embedding_dimension,
            num_layers=self.param.out_layer_num,
            act="relu",
        )

        self.intra_modal = GAT(
            in_channels=1,
            hidden_channels=self.param.lstm_hidden_dim,
            out_channels=self.param.embedding_dimension,
            num_layers=self.param.out_layer_num,
            act="relu",
        )
        self.lin_spatial_to_temporal = MLP(
            in_channels=3 * self.param.embedding_dimension,
            out_channels=1,
            num_layers=1,
        )

        self.conv1d_out = self.param.window_length - kernel_size + 1
        # Linear layer for each node (sensor)
        self.conv1d = torch.nn.Conv1d(
            in_channels=self.node_num,
            out_channels=self.node_num,
            kernel_size=kernel_size,
        )

        self.lin_predictor = MLP(
            in_channels=self.conv1d_out, out_channels=1, num_layers=1
        )
        self.vae = MyVGAE(
            (
                VariationalGCNEncoder(
                    self.conv1d_out,
                    out_channels=self.param.out_layer_inter_dim,
                )
            )
        )

    def getParmeters():
        return {"kernel_size": 16, "gamma1": 0.5, "gamma2": 0.5}

    def loss(self, x, x_truth):
        # x=(Node,Window)
        batch = x.shape[0]
        # get learnable adjacency matrix
        trainable_adj = self.graph_learner()
        index, weight = dense_to_sparse(trainable_adj)

        x = x.permute(0, 2, 1).reshape(-1, self.node_num).unsqueeze(-1).contiguous()

        data_list_attention = [
            Data(x=x_, edge_index=index, edge_weight=weight) for x_ in x
        ]
        batch_attention = Batch.from_data_list(data_list_attention)
        x1 = self.multi_head_attention(
            batch_attention.x,
            edge_index=batch_attention.edge_index,
            edge_weight=batch_attention.edge_weight,
        ).view(-1, self.node_num, self.param.embedding_dimension)

        data_list_inter = [
            Data(x=x_, edge_index=self.inter_index, edge_weight=self.inter_weigth)
            for x_ in x
        ]
        batch_inter = Batch.from_data_list(data_list_inter)
        # x2 = self.inter_modal(h0, self.iter_adjacency)
        x2 = self.inter_modal(
            batch_inter.x,
            edge_index=batch_inter.edge_index,
            edge_weight=batch_inter.edge_weight,
        ).view(-1, self.node_num, self.param.embedding_dimension)

        data_list_intra = [
            Data(x=x_, edge_index=self.intra_index, edge_weight=self.intra_weigth)
            for x_ in x
        ]
        batch_intra = Batch.from_data_list(data_list_intra)
        # x3 = self.intra_modal(h0, self.itra_adjacency)
        x3 = self.intra_modal(
            batch_intra.x,
            edge_index=batch_intra.edge_index,
            edge_weight=batch_intra.edge_weight,
        ).view(-1, self.node_num, self.param.embedding_dimension)

        x = torch.concat((x1, x2, x3), dim=-1).relu()
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        x = self.lin_spatial_to_temporal(x)  # (Node ,window_length)
        x = (
            x.squeeze(-1)
            .reshape(batch, -1, self.node_num)
            .permute(0, 2, 1)
            .contiguous()
        )

        # time convolution
        # x: [batch, nodes, time] → [batch, time, nodes]
        x: torch.Tensor = self.conv1d(x).relu()

        # Predictor
        # Apply the transformation to each channel separately
        prediction = self.lin_predictor(x)
        loss_prediction = torch.nn.functional.mse_loss(
            prediction.squeeze(-1), x_truth, reduction="none"
        )
        # VAE
        data_list_vae = [Data(x=x_, edge_index=index, edge_weight=weight) for x_ in x]

        batch_vae = Batch.from_data_list(data_list_vae)
        vae_out = self.vae.encode(
            batch_vae.x, batch_vae.edge_index, batch_vae.edge_weight
        )

        loss_reconstruction = self.vae.recon_loss(vae_out, index)
        return self.gamma1 * loss_reconstruction + (1 - self.gamma2) * loss_prediction

        #         for i, x_ in enumerate(x):
        # # batch_vae = Batch.from_data_list(data_list_vae)
        # vae_out = self.vae.encode(x_, index, weight)

        # loss_reconstruction = self.vae.recon_loss(vae_out, index)
        # loss_prediction[i] = (
        #     self.gamma1 * loss_reconstruction
        #     + (1 - self.gamma2) * loss_prediction[i]
        # )

        return loss_prediction
