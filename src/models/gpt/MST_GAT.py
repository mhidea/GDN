import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse


class GraphLearner(nn.Module):
    def __init__(self, num_nodes, top_k):
        super(GraphLearner, self).__init__()
        self.num_nodes = num_nodes
        self.top_k = top_k
        self.embedding = nn.Parameter(torch.randn(num_nodes, num_nodes))

    def forward(self):
        # Compute similarity (symmetric)
        sim = F.relu(torch.matmul(self.embedding, self.embedding.T))  # [N, N]

        # For top-k, mask diagonals
        sim.fill_diagonal_(-float("inf"))

        # Select top-k neighbors for each node
        topk_vals, topk_indices = torch.topk(sim, self.top_k, dim=-1)

        # Create edge index and weights
        node_indices = torch.arange(self.num_nodes).unsqueeze(1).expand(-1, self.top_k)
        edge_index = torch.stack(
            [node_indices.reshape(-1), topk_indices.reshape(-1)], dim=0
        )  # [2, E]
        edge_weight = topk_vals.reshape(-1)

        return edge_index, edge_weight


class MGATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.2):
        super(MGATLayer, self).__init__()
        self.gat = GATConv(in_channels, out_channels, heads=heads, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight, modality_groups):
        # Create modality-aware attention mask
        mask = torch.zeros(edge_index.shape[1], device=x.device)

        # Mask out edges not within the same modality group
        modality_sets = [set(group) for group in modality_groups]
        node_to_modality = {}
        for idx, group in enumerate(modality_sets):
            for node in group:
                node_to_modality[node] = idx

        src, dst = edge_index
        for i in range(edge_index.shape[1]):
            if node_to_modality.get(src[i].item()) == node_to_modality.get(
                dst[i].item()
            ):
                mask[i] = 1.0

        edge_weight = edge_weight * mask  # zero out cross-modality edges

        # Remove zero-weighted edges
        valid = edge_weight.nonzero(as_tuple=True)[0]
        edge_index = edge_index[:, valid]
        edge_weight = edge_weight[valid]

        return self.gat(x, edge_index, edge_weight)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super(TemporalConv, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=1)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=1)

    def forward(self, x):
        # x: [batch, nodes, time] → [batch, time, nodes]
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x.permute(0, 2, 1)  # [batch, nodes, time]


class VAEReconstruction(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAEReconstruction, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2 * latent_dim),  # mean and logvar
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.ran
