import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GCN
from models.BaseModel import BaseModel
from torch_geometric.nn import GCNConv, GAT, MLP, DenseGCNConv
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, Batch


class TimeWindowGNN(BaseModel):
    def __init__(self, **kwargs):
        """
        GNN model for time-windowed sensor data with trainable adjacency matrix.

        Parameters:
        -----------
        num_sensors : int
            Number of sensors (nodes) in the graph
        window_size : int
            Size of the time window for each sensor
        hidden_channels : int
            Number of hidden features
        sparsification_method : str
            Method to sparsify the adjacency matrix: 'dropout' or 'topk'
        dropout_rate : float
            Dropout rate if sparsification_method is 'dropout'
        k_ratio : float
            Ratio of edges to keep if sparsification_method is 'topk'
        """
        super(TimeWindowGNN, self).__init__(**kwargs)

        # Store parameters
        self.num_sensors = self.node_num
        self.window_size = self.param.window_length
        self.sparsification_method = kwargs["sparsification_method"]
        dropout_rate = self.param.topk / self.node_num
        k_ratio = self.param.topk / self.node_num

        self.dropout_rate = dropout_rate
        self.k_value = int(k_ratio * self.node_num * self.node_num)

        # Initialize trainable adjacency matrix for sensor relationships
        self.adj_weights = nn.Parameter(
            torch.rand(self.num_sensors, self.num_sensors) * 0.1
        )

        # Temporal feature extraction for each sensor's time window
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # Calculate the output size after temporal convolutions
        # For window_size W, after two MaxPool1d(2), it becomes W/4
        temporal_output_size = 32 * (self.window_size // 4)

        # Projection layer to convert temporal features to node features
        self.projection = nn.Linear(
            temporal_output_size, self.param.out_layer_inter_dim
        )

        # GNN layers
        self.conv1 = GCNConv(
            self.param.out_layer_inter_dim, self.param.out_layer_inter_dim
        )
        self.conv2 = GCNConv(
            self.param.out_layer_inter_dim, self.param.out_layer_inter_dim
        )

        # Output layer to produce a single value per sensor
        self.output_layer = nn.Linear(self.param.out_layer_inter_dim, 1)

    def get_adjacency_matrix(self, training=True):
        # Make adjacency matrix symmetric
        adj = self.adj_weights + self.adj_weights.t()

        # Apply sigmoid to constrain values between 0 and 1
        adj = torch.sigmoid(adj)

        # Apply sparsification during training
        if training:
            if self.sparsification_method == "dropout":
                # Apply dropout to adjacency matrix
                mask = torch.ones_like(adj).bernoulli_(1 - self.dropout_rate)
                adj = adj * mask

            elif self.sparsification_method == "topk":
                # Keep only top-k values
                values, _ = torch.topk(adj.flatten(), self.k_value)
                threshold = values[-1]
                adj = adj * (adj >= threshold).float()

        return adj

    def pre_forward(self, x, training=True):
        """
        Forward pass of the model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (N, W) where N is the number of sensors and
            W is the window size
        training : bool
            Whether the model is in training mode

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (N, 1) with one value per sensor
        """
        batch_size = x.size(0) if len(x.shape) > 2 else 1

        if len(x.shape) > 2:
            # If batched input: (B, N, W) -> reshape to (B*N, W)
            x = x.reshape(-1, self.window_size)

        # Process each sensor's time window
        # Reshape to (N, 1, W) for 1D convolution
        x = x.unsqueeze(1)

        # Apply temporal convolutions
        x = self.temporal_conv(x)

        # Flatten the temporal features
        x = x.reshape(x.size(0), -1)

        # Project to hidden dimension
        x = self.projection(x)

        # Get current adjacency matrix
        adj = self.get_adjacency_matrix(training=training)

        # Convert to edge_index format for PyTorch Geometric
        edge_index = adj.nonzero().t().contiguous()
        edge_weight = adj[edge_index[0], edge_index[1]]

        # If batched, we need to create a batch graph for each sample
        if batch_size > 1:
            # Create a list to store batch graphs
            batch_x = []
            batch_edge_index = []
            batch_edge_weight = []

            for b in range(batch_size):
                # Get node features for this batch
                start_idx = b * self.num_sensors
                end_idx = (b + 1) * self.num_sensors
                batch_x.append(x[start_idx:end_idx])

                # Adjust edge indices for this batch
                batch_edge_index.append(edge_index + b * self.num_sensors)
                batch_edge_weight.append(edge_weight)

            # Concatenate all batches
            x = torch.cat(batch_x, dim=0)
            edge_index = torch.cat(batch_edge_index, dim=1)
            edge_weight = torch.cat(batch_edge_weight, dim=0)

        # Apply GNN layers
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.5, training=training)
        x = F.relu(self.conv2(x, edge_index, edge_weight))

        # Apply output layer to get a single value per sensor
        x = self.output_layer(x)

        # Reshape back to (B, N, 1) if batched input
        if batch_size > 1:
            x = x.reshape(batch_size, self.num_sensors, 1)
        else:
            x = x.reshape(self.num_sensors, 1)

        return x

    def getParmeters():
        return {"sparsification_method": "topk"}


class FeatureWindowGNN(BaseModel):
    def __init__(self, **kwargs):
        """
        GNN model that treats time window values directly as node features.

        Parameters:
        -----------
        num_sensors : int
            Number of sensors (nodes) in the graph
        window_size : int
            Size of the time window for each sensor, used as node features
        hidden_channels : int
            Number of hidden features
        sparsification_method : str
            Method to sparsify the adjacency matrix: 'dropout' or 'topk'
        dropout_rate : float
            Dropout rate if sparsification_method is 'dropout'
        k_ratio : float
            Ratio of edges to keep if sparsification_method is 'topk'
        """
        super(FeatureWindowGNN, self).__init__(**kwargs)

        # Store parameters
        self.num_sensors = self.node_num
        self.window_size = self.param.window_length
        self.sparsification_method = kwargs["sparsification_method"]
        self.dropout_rate = self.param.topk / self.node_num
        self.k_ratio = self.param.topk / self.node_num

        # Initialize trainable adjacency matrix for sensor relationships
        self.adj_weights = nn.Parameter(
            torch.rand(self.num_sensors, self.num_sensors) * 0.1
        )

        # GNN layers - input features are directly the window_size values
        self.conv1 = GCNConv(self.window_size, self.param.out_layer_inter_dim)
        self.conv2 = GCNConv(
            self.param.out_layer_inter_dim, self.param.out_layer_inter_dim
        )
        self.conv3 = GCNConv(
            self.param.out_layer_inter_dim, self.param.out_layer_inter_dim
        )

        # Output layer to produce a single value per sensor
        self.output_layer = nn.Linear(self.param.out_layer_inter_dim, 1)

    def get_adjacency_matrix(self, training=True):
        # Make adjacency matrix symmetric
        adj = self.adj_weights + self.adj_weights.t()

        # Apply sigmoid to constrain values between 0 and 1
        adj = torch.sigmoid(adj)

        # Apply sparsification during training
        if training:
            if self.sparsification_method == "dropout":
                # Apply dropout to adjacency matrix
                mask = torch.ones_like(adj).bernoulli_(1 - self.dropout_rate)
                adj = adj * mask

            elif self.sparsification_method == "topk":
                # Keep only top-k values
                values, _ = torch.topk(adj.flatten(), self.param.topk)
                threshold = values[-1]
                adj = adj * (adj >= threshold).float()

        return adj

    def pre_forward(self, x, training=True):
        """
        Forward pass of the model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (B, N, W) for batched input or (N, W) for single sample
            where B is batch size, N is the number of sensors, and W is the window size
        training : bool
            Whether the model is in training mode

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (B, N, 1) for batched input or (N, 1) for single sample
            with one anomaly score per sensor
        """
        # Determine if input is batched
        # If x has 3 dimensions (B, N, W), it's batched; if 2 dimensions (N, W), it's a single sample
        batch_size = x.size(0) if len(x.shape) > 2 else 1
        # x shape: (B, N, W) for batched input or (N, W) for single sample

        # Get current adjacency matrix with applied sparsification
        adj = self.get_adjacency_matrix(training=training)
        # adj shape: (N, N)

        # Convert adjacency matrix to edge_index and edge_weight format for PyTorch Geometric
        edge_index = adj.nonzero().t().contiguous()
        # edge_index shape: (2, E) where E is the number of edges
        edge_weight = adj[edge_index[0], edge_index[1]]
        # edge_weight shape: (E)

        # Process each batch separately
        if batch_size > 1:
            # List to store outputs for each batch
            outputs = []

            for b in range(batch_size):
                # Get features for this batch
                # Each node's features are directly the W values from the time window
                batch_features = x[b]  # Shape: (N, W)

                # Apply GNN layers
                h = F.relu(self.conv1(batch_features, edge_index, edge_weight))
                # h shape: (N, hidden_channels)

                h = self.aggr(h, training=training)
                # h shape: (N, hidden_channels)

                h = F.relu(self.conv2(h, edge_index, edge_weight))
                # h shape: (N, hidden_channels)

                h = self.aggr(h, training=training)
                # h shape: (N, hidden_channels)

                h = F.relu(self.conv3(h, edge_index, edge_weight))
                # h shape: (N, hidden_channels)

                # Apply output layer to get a single anomaly score per sensor
                out = self.output_layer(h)
                # out shape: (N, 1)

                outputs.append(out)

            # Stack outputs from all batches
            x = torch.stack(outputs)
            # x shape: (B, N, 1)

        else:
            # For single sample, process directly
            # If input is (B, N, W) with B=1, reshape to (N, W)
            if len(x.shape) > 2:
                x = x.squeeze(0)
            # x shape: (N, W)

            # Apply GNN layers
            x = F.relu(self.conv1(x, edge_index, edge_weight))
            # x shape: (N, hidden_channels)

            x = self.aggr(x, training=training)
            # x shape: (N, hidden_channels)

            x = F.relu(self.conv2(x, edge_index, edge_weight))
            # x shape: (N, hidden_channels)

            x = self.aggr(x, training=training)
            # x shape: (N, hidden_channels)

            x = F.relu(self.conv3(x, edge_index, edge_weight))
            # x shape: (N, hidden_channels)

            # Apply output layer to get a single anomaly score per sensor
            x = self.output_layer(x)
            # x shape: (N, 1)

        return x

    def getParmeters():
        return {"sparsification_method": "topk"}

    def aggr(self, x, training):
        if self.sparsification_method == "dropout":
            return F.dropout(x, p=self.dropout_rate, training=training)
        if self.sparsification_method == "topk":
            return F.dropout(x, p=self.dropout_rate, training=training)


class GNN_LSTM_AnomalyDetector_Optimized(BaseModel):
    def __init__(self, **kwargs):
        """
        Optimized model that combines GNN for feature relationships and LSTM for temporal patterns.
        """
        super(GNN_LSTM_AnomalyDetector_Optimized, self).__init__(**kwargs)

        # Store parameters
        self.num_sensors = self.node_num
        self.window_size = self.param.window_length
        self.sparsification_method = kwargs["sparsification_method"]
        self.dropout_rate = self.param.topk / self.node_num
        self.k_ratio = self.param.topk / self.node_num

        # Initialize trainable adjacency matrix for feature relationships
        self.adj_weights = nn.Parameter(
            torch.rand(self.window_size, self.window_size) * 0.1
        )

        # GNN layers for learning relationships between features
        self.gnn_layers = nn.ModuleList()
        for i in range(self.param.out_layer_num):
            if i == 0:
                # First layer takes raw features
                self.gnn_layers.append(GCNConv(1, self.param.out_layer_inter_dim))
            else:
                # Subsequent layers
                self.gnn_layers.append(
                    GCNConv(
                        self.param.out_layer_inter_dim, self.param.out_layer_inter_dim
                    )
                )

        # LSTM for temporal patterns
        self.lstm = nn.LSTM(
            input_size=self.param.out_layer_inter_dim,
            hidden_size=self.param.out_layer_inter_dim,
            num_layers=kwargs["lstm_layers"],
            batch_first=True,
        )

        # Output layer
        self.output_layer = nn.Linear(self.param.out_layer_inter_dim, 1)

    def get_adjacency_matrix(self, training=True):
        # Make adjacency matrix symmetric
        adj = self.adj_weights + self.adj_weights.t()

        # Apply sigmoid to constrain values between 0 and 1
        adj = torch.sigmoid(adj)

        # Apply sparsification during training
        if training:
            if self.sparsification_method == "dropout":
                # Apply dropout to adjacency matrix
                mask = torch.ones_like(adj).bernoulli_(1 - self.dropout_rate)
                adj = adj * mask

            elif self.sparsification_method == "topk":
                # Keep only top-k values
                values, _ = torch.topk(adj.flatten(), self.k_value)
                threshold = values[-1]
                adj = adj * (adj >= threshold).float()

        return adj

    def pre_forward(self, x, training=True):
        """
        Forward pass of the model.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_sensors, window_size)
        training : bool
            Whether the model is in training mode

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, num_sensors)
        """
        # x shape: (batch_size, num_sensors, window_size)
        batch_size = x.size(0)

        # Get adjacency matrix for feature relationships
        adj = self.get_adjacency_matrix(training=training)
        # adj shape: (window_size, window_size)

        # Convert adjacency matrix to edge_index format
        edge_index = adj.nonzero().t().contiguous()
        # edge_index shape: (2, E) where E is number of edges
        edge_weight = adj[edge_index[0], edge_index[1]]
        # edge_weight shape: (E)

        # Reshape input to process all sensors in parallel
        # From (batch_size, num_sensors, window_size) to (batch_size * num_sensors, window_size)
        x_reshaped = x.reshape(-1, self.window_size)
        # Add feature dimension: (batch_size * num_sensors, window_size) -> (batch_size * num_sensors, window_size, 1)
        x_reshaped = x_reshaped.unsqueeze(2)
        # Transpose to get (batch_size * num_sensors, window_size, 1) -> (batch_size * num_sensors, 1, window_size)
        # This is needed because we want to apply GNN across the window dimension
        x_reshaped = x_reshaped.transpose(1, 2)

        # Process all sensors in parallel using GNN
        # We need to process each batch_size * num_sensors separately
        all_gnn_outputs = []

        for i in range(batch_size * self.num_sensors):
            # Get data for this instance
            instance_data = x_reshaped[i, 0, :].unsqueeze(1)  # Shape: (window_size, 1)

            # Apply GNN layers
            h = instance_data
            for gnn_layer in self.gnn_layers:
                h = gnn_layer(h, edge_index, edge_weight)
                h = F.relu(h)
                h = self.aggr(h, training=training)

            all_gnn_outputs.append(h)

        # Stack all GNN outputs
        gnn_outputs = torch.stack(all_gnn_outputs)
        # gnn_outputs shape: (batch_size * num_sensors, window_size, hidden_size)

        # Reshape for LSTM
        gnn_outputs = gnn_outputs.reshape(
            batch_size, self.num_sensors, self.window_size, self.hidden_size
        )

        # Apply LSTM to each sensor's GNN outputs
        lstm_outputs = []

        for b in range(batch_size):
            sensor_lstm_outputs = []

            for s in range(self.num_sensors):
                # Get GNN outputs for this sensor
                sensor_gnn_output = gnn_outputs[
                    b, s
                ]  # Shape: (window_size, hidden_size)

                # Add batch dimension for LSTM
                sensor_gnn_output = sensor_gnn_output.unsqueeze(
                    0
                )  # Shape: (1, window_size, hidden_size)

                # Apply LSTM
                lstm_out, _ = self.lstm(sensor_gnn_output)

                # Take the last output
                sensor_lstm_outputs.append(lstm_out[0, -1])

            # Stack outputs for all sensors in this batch
            lstm_outputs.append(torch.stack(sensor_lstm_outputs))

        # Stack outputs for all batches
        lstm_outputs = torch.stack(lstm_outputs)
        # lstm_outputs shape: (batch_size, num_sensors, hidden_size)

        # Apply output layer to get final predictions
        outputs = self.output_layer(lstm_outputs).squeeze(-1)
        # outputs shape: (batch_size, num_sensors)

        return outputs

    def aggr(self, x, training):
        if self.sparsification_method == "dropout":
            return F.dropout(x, p=self.dropout_rate, training=training)
        if self.sparsification_method == "topk":
            return F.dropout(x, p=self.dropout_rate, training=training)

    def getParmeters():
        return {"lstm_layers": 1}


class MyGAT(BaseModel):
    """docstring for MyGAT."""

    def __init__(self, **kwargs):
        super(MyGAT, self).__init__(**kwargs)

        self.adj = torch.nn.Parameter(
            dense_to_sparse(torch.ones(self.node_num, self.node_num))[0],
            requires_grad=False,
        )
        self.w = torch.nn.Parameter(
            torch.rand(self.node_num * self.node_num), requires_grad=True
        )
        self.w.requires_grad = True
        self.gcn = GCNConv(self.param.window_length, self.param.window_length)
        self.att_layer = GAT(
            self.param.window_length,
            self.param.out_layer_inter_dim,
            self.param.out_layer_num,
        )
        self.lin = torch.nn.Linear(self.param.out_layer_inter_dim, 1)
        self.relu = torch.nn.ReLU()

    def weigthAdj(self):
        return self.w

    def pre_forward(self, x: torch.Tensor):
        x = self.gcn(x=x, edge_index=self.adj, edge_weight=self.w)
        const_weight = self.w.detach().narrow_copy(0, 0, self.node_num * self.node_num)
        const_weight.requires_grad = False
        if x.dim() == 3:
            xl = [Data(x=x_, edge_index=self.adj, edge_weight=const_weight) for x_ in x]
            data = Batch.from_data_list(xl)
        else:
            data = Data(x=x, edge_index=self.adj, edge_weight=const_weight)

        x = self.att_layer(
            x=data.x, edge_index=data.edge_index, edge_weight=data.edge_weight
        )
        x = self.relu(x)
        x = self.lin(x.view(-1, self.param.out_layer_inter_dim))
        # x = F.sigmoid(x)
        return x.view(-1, self.node_num, 1)


class MyGATEmbd(BaseModel):
    """docstring for MyGAT."""

    def __init__(self, **kwargs):
        super(MyGATEmbd, self).__init__(**kwargs)

        self.node_embedding = torch.nn.Parameter(
            torch.rand(
                self.node_num, self.param.embedding_dimension, requires_grad=True
            )
        )
        self.directed = False
        self.conv1 = GCNConv(
            self.param.window_length,
            self.param.out_layer_inter_dim,
            cached=True,
            normalize=False,
        )
        self.conv2 = GCNConv(
            self.param.out_layer_inter_dim,
            1,
            cached=True,
            normalize=False,
        )

    def init_parameters(self):
        nn.init.xavier_uniform_(self.node_embedding)

    def adj(self):
        # Normalize embeddings for stable cosine similarity
        # normalized_embeddings = F.normalize(self.node_embedding, p=2, dim=1)

        # Compute cosine similarity matrix
        # w = self.node_embedding(
        #     torch.arange(self.node_num, requires_grad=False).to(self.param.device)
        # )
        # sim_matrix = torch.mm(w, w.t())
        sim_matrix = self.node_embedding @ self.node_embedding.t()
        # Ensure self-similarity is not selected
        sim_matrix.fill_diagonal_(-1)  # Set diagonal to -1 to exclude self-connections

        # Get top-k neighbors for each node
        topk_values, topk_indices = torch.topk(sim_matrix, k=self.param.topk, dim=1)

        # Create binary adjacency mask
        adj_mask = torch.zeros_like(sim_matrix)
        adj_mask.scatter_(1, topk_indices, 1.0)

        # For undirected graphs, ensure symmetric connections
        if not self.directed:
            adj_mask = torch.max(adj_mask, adj_mask.t())
            adj_mask.fill_diagonal_(0)  # Remove self-loops

        return adj_mask

    def weigthAdj(self):
        with torch.no_grad():
            adj, w = dense_to_sparse(self.adj())
        return w

    def forward(self, data):
        adj, w = dense_to_sparse(self.adj())
        x = self.conv1(data, edge_index=adj, edge_weight=w)
        x = F.relu(x)
        x = self.conv2(x, edge_index=adj, edge_weight=w)
        x = F.sigmoid(x)
        return x
