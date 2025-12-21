import torch
import torch.nn as nn
import torch.nn.functional as F


def _sym_norm_sparse(A: torch.Tensor) -> torch.Tensor:
    """Symmetric normalization D^{-1/2} A D^{-1/2} for sparse COO."""
    A = A.coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1e-12)
    invsqrt = deg.pow(-0.5)
    i, j = A.indices()
    v = A.values() * invsqrt[i] * invsqrt[j]
    return torch.sparse_coo_tensor(A.indices(), v, A.size(), device=A.device).coalesce()


class MedGCNLayer(nn.Module):
    """MedGCN-style multi-relation graph convolution (relation-specific weights)."""

    def __init__(self, in_feats: int, out_feats: int, num_relations: int, bias: bool = True):
        super().__init__()
        self.num_rel = num_relations
        self.rel_weights = nn.ParameterList(
            [nn.Parameter(torch.empty(in_feats, out_feats)) for _ in range(num_relations)]
        )
        self.self_loop = nn.Parameter(torch.empty(in_feats, out_feats))
        self.bias = nn.Parameter(torch.zeros(out_feats)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.self_loop)
        for w in self.rel_weights:
            nn.init.xavier_uniform_(w)

    def forward(self, X: torch.Tensor, adjs: list[torch.Tensor]) -> torch.Tensor:
        out = X @ self.self_loop
        for r, A in enumerate(adjs):
            A = _sym_norm_sparse(A)
            AX = torch.spmm(A, X)
            out = out + (AX @ self.rel_weights[r])
        if self.bias is not None:
            out = out + self.bias
        return out


class MedGCN(nn.Module):
    """2-layer MedGCN baseline (no relation attention)."""

    def __init__(self, in_feats: int, h_feats: int, num_classes: int, num_relations: int, dropout: float = 0.5):
        super().__init__()
        self.g1 = MedGCNLayer(in_feats, h_feats, num_relations)
        self.g2 = MedGCNLayer(h_feats, num_classes, num_relations)
        self.dropout = dropout

    def forward(self, X: torch.Tensor, adjs: list[torch.Tensor]) -> torch.Tensor:
        h = self.g1(X, adjs)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.g2(h, adjs)
        return h
