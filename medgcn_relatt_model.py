from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from medgcn_model import _sym_norm_sparse


class MedGCNRelationAttention(nn.Module):
    """
    2-layer MedGCN + relation-level attention (layer 2).

    Layer 1 (stable):
      h1_r = ReLU(A_r X W1_r)
      h1 = mean_r(h1_r)

    Layer 2 (attention):
      h2_r = ReLU(A_r h1 W2_r)
      alpha = softmax_r(<h2_r, q>/tau)
      h2_att = sum_r alpha_r * h2_r
      h2_base = mean_r(h2_r)
      h2 = h2_base + h2_att (residual)

    Return:
      logits, alpha  (alpha: [N, R])
    """

    def __init__(
        self,
        in_feats: int,
        h_feats: int,
        num_classes: int,
        num_relations: int,
        dropout: float = 0.5,
        rel_drop: float = 0.2,
    ):
        super().__init__()
        self.num_rel = num_relations
        self.dropout = dropout
        self.rel_drop = rel_drop

        # -------- Layer 1: relation-specific linears --------
        self.rel_linears1 = nn.ModuleList(
            [nn.Linear(in_feats, h_feats, bias=False) for _ in range(num_relations)]
        )

        # -------- Layer 2: relation-specific linears --------
        self.rel_linears2 = nn.ModuleList(
            [nn.Linear(h_feats, h_feats, bias=False) for _ in range(num_relations)]
        )

        # attention query (layer 2)
        self.att_q = nn.Parameter(torch.empty(h_feats))
        nn.init.xavier_uniform_(self.att_q.unsqueeze(0))

        # temperature (learnable)
        self.tau = nn.Parameter(torch.tensor(2.0))

        self.out = nn.Linear(h_feats, num_classes)

    def _relation_dropout(self, alpha: Tensor) -> Tensor:
        # alpha: [N, R, 1]
        if not self.training or self.rel_drop <= 0:
            return alpha
        mask = (torch.rand_like(alpha) > self.rel_drop).float()
        alpha = alpha * mask
        alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)
        return alpha

    def forward(self, X: Tensor, adjs: list[Tensor]) -> Tuple[Tensor, Tensor]:
        # ---------------- Layer 1 ----------------
        rel_h1 = []
        for r in range(self.num_rel):
            A = _sym_norm_sparse(adjs[r])
            h = torch.spmm(A, X)
            h = self.rel_linears1[r](h)
            h = F.relu(h)
            rel_h1.append(h.unsqueeze(1))  # [N,1,H]

        H1 = torch.cat(rel_h1, dim=1)  # [N,R,H]
        h1 = H1.mean(dim=1)            # [N,H]
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        # ---------------- Layer 2 (attention) ----------------
        rel_h2 = []
        for r in range(self.num_rel):
            A = _sym_norm_sparse(adjs[r])
            h = torch.spmm(A, h1)
            h = self.rel_linears2[r](h)
            h = F.relu(h)
            rel_h2.append(h.unsqueeze(1))  # [N,1,H]

        H2 = torch.cat(rel_h2, dim=1)  # [N,R,H]

        scores = torch.einsum("nrh,h->nr", H2, self.att_q)  # [N,R]
        tau = torch.clamp(self.tau, min=0.5, max=5.0)

        alpha = torch.softmax(scores / tau, dim=1).unsqueeze(-1)  # [N,R,1]
        alpha = self._relation_dropout(alpha)

        h2_att = (H2 * alpha).sum(dim=1)  # [N,H]
        h2_base = H2.mean(dim=1)          # [N,H]
        h2 = h2_base + h2_att             # residual fusion

        h2 = F.dropout(h2, p=self.dropout, training=self.training)
        logits = self.out(h2)

        return logits, alpha.squeeze(-1)  # alpha: [N,R]
