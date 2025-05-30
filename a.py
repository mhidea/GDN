# %%
import torch
import torch_geometric
import torch.nn.functional as F

# %%
num_nodes = 4
embedding_dim = 8
top_k = 2
directed = True

# Learnable time series embeddings
node_embeddings = torch.nn.Parameter(torch.Tensor(num_nodes, embedding_dim))


# %%
from torch_geometric.nn import GAT

m = GAT(2, 4, 1, dropout=0.5, add_self_loops=True, negative_slope=0.2, v2=True)

# %%
m(
    torch.rand(num_nodes, 2),
    edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
    #   ,return_attention_weights =True
)  # ,edge_attr=torch.rand(4,embedding_dim))

# %%
torch.nn.init.xavier_uniform_(node_embeddings)

# %%
node_embeddings

# %%
normalized_embeddings = F.normalize(node_embeddings, p=2, dim=1)
node_embeddings


# %%
# Compute cosine similarity matrix
sim_matrix = torch.mm(normalized_embeddings, normalized_embeddings.t())
sim_matrix


# %%
# Ensure self-similarity is not selected
sim_matrix.fill_diagonal_(-1)  # Set diagonal to -1 to exclude self-connections

# Get top-k neighbors for each node
topk_values, topk_indices = torch.topk(sim_matrix, k=top_k, dim=1)
topk_values, topk_indices


# %%
# Create binary adjacency mask
adj_mask = torch.zeros_like(sim_matrix)
adj_mask.scatter_(1, topk_indices, 1.0)
adj_mask


# %%
# For undirected graphs, ensure symmetric connections
if not directed:
    adj_mask = torch.max(adj_mask, adj_mask.t())
    # Remove self-loops
    adj_mask.fill_diagonal_(0)

# %%
torch.nn.Parameter(torch.empty(1, 2, 5))

# %%
torch.nn.Parameter()

# %%
node_embeddings.shape

# %%
x = torch.rand(4, 3)
x

# %%
import torch_geometric.utils


adj = torch.ones(4, 4)
adj, f = torch_geometric.utils.dense_to_sparse(adj)
adj, f = torch_geometric.utils.remove_self_loops(adj)

# %%
model = torch_geometric.nn.GAT(
    num_layers=1, hidden_channels=6, in_channels=-1, out_channels=2
)

# %%
model(x, adj)
