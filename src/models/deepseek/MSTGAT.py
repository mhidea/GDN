import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_batch
from models.BaseModel import BaseModel
from util.env import get_param


class MGATLayer(nn.Module):
    """Multimodal Graph Attention Layer (intra + inter-modal)"""

    def __init__(self, in_channels, out_channels, heads, num_modalities):
        super().__init__()
        self.intra_convs = nn.ModuleList(
            [
                GATConv(in_channels, out_channels, heads=heads, concat=False)
                for _ in range(num_modalities)
            ]
        )
        self.inter_conv = GATConv(in_channels, out_channels, heads=heads, concat=False)

    def forward(self, x, intra_edges, inter_edges):
        # Intra-modal aggregation
        intra_out = torch.zeros_like(x)
        for i, conv in enumerate(self.intra_convs):
            mask = intra_edges[0] == i  # Assuming edge attributes contain modality info
            intra_out += conv(x, intra_edges[:, mask])

        # Inter-modal aggregation
        inter_out = self.inter_conv(x, inter_edges)

        return intra_out + inter_out


class TCNBlock(nn.Module):
    """Temporal Convolution Block with dilation"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(dilation * (kernel_size - 1)) // 2,
            dilation=dilation,
        )
        self.res = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        residual = x
        x = self.conv(x)
        if self.res is not None:
            residual = self.res(residual)
        x = self.norm(x + residual)
        return F.relu(x)


class MSTGAT(BaseModel):
    def __init__(self, num_modalities, **kwargs):
        super().__init__(**kwargs)

        self.window_size = get_param().window_length
        hidden_dim = get_param().out_layer_inter_dim
        # Spatial encoder
        self.mgat1 = MGATLayer(
            self.window_size, hidden_dim, heads=4, num_modalities=num_modalities
        )
        self.mgat2 = MGATLayer(
            hidden_dim, hidden_dim, heads=4, num_modalities=num_modalities
        )

        # Temporal encoder
        self.tcn = nn.Sequential(
            TCNBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=1),
            TCNBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2),
            TCNBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=4),
        )

        # Reconstruction module (VAE)
        self.fc_mu = nn.Linear(hidden_dim, hidden_dim)
        self.fc_logvar = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.window_size),
        )

        # Prediction module
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # Predict next time step
        )

    def encode(self, x, intra_edges, inter_edges):
        # Spatial aggregation
        x = self.mgat1(x, intra_edges, inter_edges)
        x = F.relu(x)
        x = self.mgat2(x, intra_edges, inter_edges)

        # Temporal processing
        x = x.permute(0, 2, 1)  # [batch, features, time]
        x = self.tcn(x)
        return x.permute(0, 2, 1)  # Back to [batch, time, features]

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def pre_forward(self, x, intra_edges, inter_edges):

        batch_size = x.size(0)

        # Encode spatial-temporal patterns
        h = self.encode(x, intra_edges, inter_edges)

        # Reconstruction
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)

        # Prediction
        pred = self.predictor(h[:, -1, :])  # Use last time step's features

        return x_recon, pred, mu, logvar

    def anomaly_score(self, x, intra_edges, inter_edges):
        x_recon, pred, _, _ = self.forward(x, intra_edges, inter_edges)
        recon_loss = F.mse_loss(x_recon, x, reduction="none").mean((1, 2))
        pred_loss = F.mse_loss(pred, x[:, :, -1:], reduction="none").mean((1, 2))
        return recon_loss + pred_loss
