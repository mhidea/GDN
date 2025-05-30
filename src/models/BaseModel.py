import torch
from util.consts import Tasks
from util.env import get_param
import warnings
from torch_geometric.nn.inits import glorot, zeros
import math


class GraphLearner(torch.nn.Module):
    def __init__(self, num_nodes, num_embeddings, top_k=None):
        super(GraphLearner, self).__init__()
        self.num_nodes = num_nodes
        self.top_k = top_k
        self.num_embeddings = num_embeddings  # Save as property
        self.embedding = torch.nn.Embedding(
            num_embeddings=num_nodes, embedding_dim=num_embeddings
        )
        # self.embedding = torch.nn.Parameter(
        #     torch.randn(num_nodes, num_embeddings)
        # )  # Trainable similarity matrix
        self.reset_parameters()

    def reset_parameters(self):
        # glorot(self.embedding.weight)
        torch.nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self):
        # Compute similarity matrix (symmetric but not binary)
        all_embeddings = self.embedding(
            torch.arange(self.num_nodes, device=self.embedding.weight.device)
        )
        sim = torch.matmul(all_embeddings, all_embeddings.T)  # Shape: [N, N]

        if self.top_k:
            # Reconstruct sparse adjacency matrix using scatter logic
            adj_matrix = torch.zeros_like(sim)

            # Get top-k neighbors per node
            topk_vals, topk_indices = torch.topk(sim, self.top_k, dim=-1)

            adj_matrix.scatter_(-1, topk_indices, topk_vals)
        else:
            adj_matrix = sim

        adj_matrix = torch.nn.functional.normalize(adj_matrix, p=2, dim=-1)
        return adj_matrix


class BaseModel(torch.nn.Module):
    """docstring for BaseModel."""

    def __init__(self, node_num, adj=None, **kwargs):
        super(BaseModel, self).__init__()
        self.param = get_param()
        self.node_num = node_num
        self.adj = adj

        self.task = self.param.task
        if self.task in [Tasks.next_label, Tasks.current_label]:
            self.nodes_to_label = torch.nn.Linear(self.node_num, 1)

    def pre_forward(self, data: torch.Tensor) -> torch.Tensor:
        if data.dim() == 3:
            return torch.randn([data.shape[0], self.node_num])
        return torch.randn(self.node_num)

    def getParmeters() -> dict:
        return {}

    def denseAdj(self) -> torch.Tensor:
        """Adjacency matrix. Tensor of (shape node_num , node_num.)
        To avoid changing the gradients use "with torch.no_grad():"

        Returns:
            torch.Tensor: node_num x node_num
        """
        if self.__class__ == BaseModel:
            print("Warning: 'Has not been overridden!")
        return torch.rand(self.node_num, self.node_num)

    def weigthAdj(self) -> torch.Tensor:
        """Weigth of adjacency matrix. Tensor of (shape node_num x node_num,).
        To avoid changing the gradients use "with torch.no_grad():"

        Returns:
            torch.Tensor: node_num x node_num
        """
        if self.__class__ == BaseModel:
            print("Warning: 'Has not been overridden!")
        return torch.rand(self.node_num * self.node_num)

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

    def loss(self, x, y_truth):
        x = self.forward(x)
        return torch.nn.functional.l1_loss(x, y_truth, reduction="none")
