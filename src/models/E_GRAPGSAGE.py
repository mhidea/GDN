from torch import nn
import torch as th

# import dgl.function as fn
import torch.nn.functional as F


class MLPPredictor(nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src["h"]
        h_v = edges.dst["h"]
        score = self.W(th.cat([h_u, h_v], 1))
        return {"score": score}

    def forward(self, graph, h):
        with graph.local_scope():
            graph.ndata["h"] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata["score"]


class SAGE(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super(SAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGELayer(ndim_in, edim, 128, activation))
        self.layers.append(SAGELayer(128, edim, ndim_out, activation))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, nfeats, efeats):
        for i, layer in enumerate(self.layers):
            if i != 0:
                nfeats = self.dropout(nfeats)
            nfeats = layer(g, nfeats, efeats)
        return nfeats.sum(1)


class SAGELayer(nn.Module):
    def __init__(self, ndim_in, edims, ndim_out, activation):
        super(SAGELayer, self).__init__()
        ### force to outut fix dimensions
        self.W_msg = nn.Linear(ndim_in + edims, ndim_out)
        ### apply weight
        self.W_apply = nn.Linear(ndim_in + ndim_out, ndim_out)
        self.activation = activation

    def message_func(self, edges):
        return {"m": self.W_msg(th.cat([edges.src["h"], edges.data["h"]], 2))}

    def forward(self, g_dgl, nfeats, efeats):
        with g_dgl.local_scope():
            g = g_dgl
            g.ndata["h"] = nfeats
            g.edata["h"] = efeats
            # Eq4
            g.update_all(self.message_func, fn.mean("m", "h_neigh"))
            # Eq5
            g.ndata["h"] = F.relu(
                self.W_apply(th.cat([g.ndata["h"], g.ndata["h_neigh"]], 2))
            )
            return g.ndata["h"]


class Model(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super().__init__()
        self.gnn = SAGE(ndim_in, ndim_out, edim, activation, dropout)
        self.pred = MLPPredictor(ndim_out, 10)

    def forward(self, g, nfeats, efeats):
        h = self.gnn(g, nfeats, efeats)
        return self.pred(g, h)


if __name__ == "__main__":
    x = th.rand(10, 8, 5)
    model = Model(8, 1, 0, F.relu, 0.3)
    model(x, th.tensor(), th.tensor())
