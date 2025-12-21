
import torch
import torch.nn as nn
import torch.nn.functional as F

def _sym_norm_sparse(A: torch.Tensor) -> torch.Tensor:
    A = A.coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1e-12)
    invsqrt = deg.pow(-0.5)
    i, j = A.indices()
    v = A.values() * invsqrt[i] * invsqrt[j]
    return torch.sparse_coo_tensor(A.indices(), v, A.size(), device=A.device).coalesce()

class RGCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, num_relations, bias=True):
        super().__init__()
        self.num_rel = num_relations
        self.rel_weights = nn.ParameterList([nn.Parameter(torch.empty(in_feats, out_feats)) for _ in range(num_relations)])
        self.self_loop = nn.Parameter(torch.empty(in_feats, out_feats))
        self.bias = nn.Parameter(torch.zeros(out_feats)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.self_loop)
        for w in self.rel_weights:
            nn.init.xavier_uniform_(w)

    def forward(self, X, adjs):
        out = X @ self.self_loop
        for r, A in enumerate(adjs):
            A = _sym_norm_sparse(A)
            AX = torch.spmm(A, X)
            out = out + AX @ self.rel_weights[r]
        if self.bias is not None:
            out = out + self.bias
        return out

class RGCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, num_relations, dropout=0.5):
        super().__init__()
        self.rgcn1 = RGCNLayer(in_feats, h_feats, num_relations)
        self.rgcn2 = RGCNLayer(h_feats, num_classes, num_relations)
        self.dropout = dropout

    def forward(self, X, adjs):
        h = self.rgcn1(X, adjs)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.rgcn2(h, adjs)
        return h
