
import torch
import torch.nn as nn
import torch.nn.functional as F

def scatter_add(src, index, dim_size):
    out = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    return out.index_add(0, index, src)

class SparseGATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, negative_slope=0.2, dropout=0.5):
        super().__init__()
        self.lin = nn.Linear(in_feats, out_feats, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(out_feats))
        self.att_dst = nn.Parameter(torch.Tensor(out_feats))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.att_dst.unsqueeze(0))

    def forward(self, X, adj):
        if not adj.is_coalesced():
            adj = adj.coalesce()
        i, j = adj.indices()    # edges: i <- j
        N = X.size(0)
        h = self.lin(X)         # [N, Fout]

        # attention logits e_ij = a^T [h_i || h_j] ~= <a_src,h_i> + <a_dst,h_j>
        e_src = (h * self.att_src).sum(dim=1)   # [N]
        e_dst = (h * self.att_dst).sum(dim=1)   # [N]
        e = self.leaky_relu(e_src[i] + e_dst[j])  # [E]

        # ==== Ổn định số: clamp để tránh overflow exp ====
        e = torch.clamp(e, min=-10.0, max=10.0)

        # softmax over incoming edges of each node i
        exp_e = torch.exp(e)
        # nếu node nào không có in-edge -> mẫu số = 0; thêm epsilon
        denom = scatter_add(exp_e, i, N) + 1e-12
        alpha = exp_e / denom[i]              # [E]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # message passing: sum_j alpha_ij * h_j
        msg = alpha.unsqueeze(1) * h[j]       # [E, Fout]
        out = torch.zeros_like(h)
        out.index_add_(0, i, msg)             # aggregate to target i
        return out

class GAT(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, dropout=0.5):
        super().__init__()
        self.gat1 = SparseGATLayer(in_feats, h_feats, dropout=dropout)
        self.gat2 = SparseGATLayer(h_feats, num_classes, dropout=dropout)
        self.dropout = dropout

    def forward(self, X, adj):
        h = self.gat1(X, adj)
        h = F.elu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.gat2(h, adj)
        return h
