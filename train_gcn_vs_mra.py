import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pandas as pd

# --- Added models ---
from gat_model import GAT
from rgcn_model import RGCN

from preparedata import load_nhanes_patient_graph


# ============================================================
# 1. MODEL ĐƠN GIẢN: GCN
# ============================================================
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_feats, h_feats)
        self.fc2 = nn.Linear(h_feats, num_classes)

    def forward(self, X, adj):
        h = torch.spmm(adj, X)
        h = F.relu(self.fc1(h))
        h = F.dropout(h, p=0.5, training=self.training)
        h = torch.spmm(adj, h)
        h = self.fc2(h)
        return h


# ============================================================
# 2. MODEL MULTI-RELATION + ATTENTION
#    (attention giữa CÁC QUAN HỆ, không phải giữa neighbors)
# ============================================================
class MultiRelationalAttentionNet(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, num_relations: int):
        super().__init__()
        self.num_rel = num_relations
        # 1 linear riêng cho mỗi quan hệ
        self.rel_linears = nn.ModuleList(
            [nn.Linear(in_feats, h_feats) for _ in range(num_relations)]
        )
        # vector attention chung cho mọi relation
        self.att_q = nn.Parameter(torch.Tensor(h_feats))
        nn.init.xavier_uniform_(self.att_q.unsqueeze(0))
        self.fc_out = nn.Linear(h_feats, num_classes)

    def forward(self, X, adjs):
        # adjs: list[torch.sparse_coo_tensor] length = num_rel
        rel_hidden = []
        for r in range(self.num_rel):
            A_r = adjs[r]
            h_r = torch.spmm(A_r, X)           # [N, D]
            h_r = torch.relu(self.rel_linears[r](h_r))  # [N, H]
            rel_hidden.append(h_r.unsqueeze(1))         # [N,1,H]

        # [N, R, H]
        H = torch.cat(rel_hidden, dim=1)

        # attention theo chiều relation:
        # score_r(i) = <H[i,r], att_q>
        # H: [N,R,H], att_q: [H]
        scores = torch.einsum("nrh,h->nr", H, self.att_q)  # [N,R]
        alpha = torch.softmax(scores, dim=1).unsqueeze(-1)  # [N,R,1]

        # H_final[i] = sum_r alpha[i,r] * H[i,r]
        H_comb = (H * alpha).sum(dim=1)  # [N,H]
        H_comb = F.dropout(H_comb, p=0.5, training=self.training)

        out = self.fc_out(H_comb)  # [N,C]
        return out


# ============================================================
# 3. BUILD k-NN GRAPH (dùng thêm cho quan hệ thứ 2)
# ============================================================
def build_knn_adj_from_X(X: torch.Tensor, k: int):
    """
    X: [N,D] tensor (CPU)
    k: số neighbors (không tính chính nó)
    """
    X_np = X.cpu().numpy()
    N = X_np.shape[0]

    k_eff = min(k, max(1, N - 1))

    nbrs = NearestNeighbors(
        n_neighbors=k_eff + 1, metric="euclidean", n_jobs=-1
    ).fit(X_np)
    distances, knn_indices = nbrs.kneighbors(X_np)

    # bỏ self (cột 0)
    knn_indices = knn_indices[:, 1:]  # [N, k_eff]

    rows = np.repeat(np.arange(N), k_eff)
    cols = knn_indices.reshape(-1)
    values = np.ones(len(rows), dtype=np.float32)

    coords = np.vstack([rows, cols])
    coords_t = torch.tensor(coords, dtype=torch.long)
    values_t = torch.tensor(values, dtype=torch.float32)

    adj = torch.sparse_coo_tensor(
        coords_t, values_t, size=(N, N)
    ).coalesce()

    # symmetrize
    adj_T = torch.sparse_coo_tensor(
        torch.stack([coords_t[1], coords_t[0]]),
        values_t,
        size=(N, N),
    )
    adj = (adj + adj_T).coalesce()
    vals = torch.where(adj.values() > 0, torch.ones_like(adj.values()), adj.values())
    adj = torch.sparse_coo_tensor(adj.indices(), vals, adj.size()).coalesce()
    return adj


# ============================================================
# 4. HÀM TRAIN 1 MODEL
# ============================================================
def train_one_model(
    model,
    X,
    y,
    train_idx,
    val_idx,
    adjs,   # list of adj (GCN: [adj], MRA: [adj1, adj2,...])
    num_epochs=200,
    lr=0.01,
    weight_decay=5e-4,
    model_name="GCN",
    save_path="best_model.pth",
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {
        "epoch": [],
        "loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    best_epoch = -1

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        if len(adjs) == 1:
            out = model(X, adjs[0])
        else:
            out = model(X, adjs)

        loss = F.cross_entropy(out[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

        # train acc
        with torch.no_grad():
            pred_train = out[train_idx].argmax(dim=1)
            train_acc = (pred_train == y[train_idx]).float().mean().item()

            pred_val = out[val_idx].argmax(dim=1)
            val_acc = (pred_val == y[val_idx]).float().mean().item()

        history["epoch"].append(epoch)
        history["loss"].append(loss.item())
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)

        if epoch % 20 == 0 or epoch == num_epochs:
            print(
                f"[{model_name}] Epoch {epoch:3d} | "
                f"Loss={loss.item():.4f} | Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f}"
            )

    print(f"[{model_name}] BEST Val Acc={best_val_acc:.4f} at epoch {best_epoch}")
    return history, best_val_acc, best_epoch


# ============================================================
# 5. EVALUATE MODEL
# ============================================================
def evaluate_model(model, X, y, test_idx, adjs, model_name="GCN"):
    model.eval()
    with torch.no_grad():
        if len(adjs) == 1:
            out = model(X, adjs[0])
        else:
            out = model(X, adjs)

    logits_test = out[test_idx]
    preds_test = logits_test.argmax(dim=1)
    y_true = y[test_idx]

    test_acc = (preds_test == y_true).float().mean().item()

    print(f"\n===== EVALUATION: {model_name} =====")
    print("Test Accuracy:", test_acc)
    print("\nClassification Report:")
    print(classification_report(y_true.cpu(), preds_test.cpu()))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true.cpu(), preds_test.cpu()))

    probs = F.softmax(logits_test, dim=1)[:, 1].cpu().numpy()
    auc = roc_auc_score(y_true.cpu().numpy(), probs)
    print("ROC-AUC:", auc)

    # trả thêm để vẽ ROC
    fpr, tpr, _ = roc_curve(y_true.cpu().numpy(), probs)

    return test_acc, auc, (fpr, tpr)


# ============================================================
# 6. MAIN: TRAIN 2 MODEL + VẼ BIỂU ĐỒ + BẢNG KẾT QUẢ
# ============================================================
def main():
    os.makedirs("results", exist_ok=True)

    # 1) Load data
    data_dir = "./data"
    X, y, train_idx, val_idx, test_idx, adj_knn = load_nhanes_patient_graph(data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    y = y.to(device)
    adj_knn = adj_knn.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)

    # 2) Build thêm 1 graph khác (multi-relation) - kNN với k lớn hơn
    #    quan hệ 1: k=10 (adj_knn)  – đã có
    #    quan hệ 2: k=25 (adj_knn_25)
    print("Building second relation graph (k=25)...")
    X_cpu = X.detach().cpu()
    adj_knn_25 = build_knn_adj_from_X(X_cpu, k=25).to(device)

    # 3) Init models
    in_feats = X.shape[1]
    num_classes = 2

    gcn = GCN(in_feats=in_feats, h_feats=64, num_classes=num_classes).to(device)
    mra = MultiRelationalAttentionNet(
        in_feats=in_feats,
        h_feats=64,
        num_classes=num_classes,
        num_relations=2,
    ).to(device)
    gat = GAT(in_feats=in_feats, h_feats=64, num_classes=num_classes).to(device)
    rgcn = RGCN(in_feats=in_feats, h_feats=64, num_classes=num_classes, num_relations=2).to(device)

    # 4) Train GCN
    print("\n===== TRAINING GCN =====")
    gcn_hist, gcn_best_val, gcn_best_epoch = train_one_model(
        gcn,
        X,
        y,
        train_idx,
        val_idx,
        adjs=[adj_knn],
        num_epochs=200,
        lr=0.01,
        weight_decay=5e-4,
        model_name="GCN",
        save_path="results/best_gcn_model.pth",
    )

    # 5) Train Multi-Relational Attention
    print("\n===== TRAINING Multi-Relational Attention (MRA) =====")
    mra_hist, mra_best_val, mra_best_epoch = train_one_model(
        mra,
        X,
        y,
        train_idx,
        val_idx,
        adjs=[adj_knn, adj_knn_25],
        num_epochs=200,
        lr=0.01,
        weight_decay=5e-4,
        model_name="MRA",
        save_path="results/best_mra_model.pth",
    )

    # 5) Train GAT
    print("\n===== TRAINING GAT =====")
    gat_hist, gat_best_val, gat_best_epoch = train_one_model(
        gat,
        X,
        y,
        train_idx,
        val_idx,
        adjs=[adj_knn],
        num_epochs=200,
        lr=0.01,
        weight_decay=5e-4,
        model_name="GAT",
        save_path="results/best_gat_model.pth",
    )

    # 6) Train R-GCN
    print("\n===== TRAINING R-GCN =====")
    rgcn_hist, rgcn_best_val, rgcn_best_epoch = train_one_model(
        rgcn,
        X,
        y,
        train_idx,
        val_idx,
        adjs=[adj_knn, adj_knn_25],
        num_epochs=200,
        lr=0.01,
        weight_decay=5e-4,
        model_name="R-GCN",
        save_path="results/best_rgcn_model.pth",
    )

    # 7) Load best weights + Evaluate on test
    gcn.load_state_dict(torch.load("results/best_gcn_model.pth", map_location=device))
    mra.load_state_dict(torch.load("results/best_mra_model.pth", map_location=device))
    gat.load_state_dict(torch.load("results/best_gat_model.pth", map_location=device))
    rgcn.load_state_dict(torch.load("results/best_rgcn_model.pth", map_location=device))

    gcn_test_acc, gcn_auc, gcn_roc = evaluate_model(gcn,  X, y, test_idx, [adj_knn],            model_name="GCN")
    mra_test_acc, mra_auc, mra_roc = evaluate_model(mra,  X, y, test_idx, [adj_knn, adj_knn_25], model_name="MRA")
    gat_test_acc, gat_auc, gat_roc = evaluate_model(gat,  X, y, test_idx, [adj_knn],            model_name="GAT")
    rgcn_test_acc, rgcn_auc, rgcn_roc = evaluate_model(rgcn, X, y, test_idx, [adj_knn, adj_knn_25], model_name="R-GCN")

    # 7) BẢNG KẾT QUẢ
    results_df = pd.DataFrame([
        {"Model": "GCN",  "Best_Val_Acc": gcn_best_val,  "Best_Val_Epoch": gcn_best_epoch,  "Test_Acc": gcn_test_acc,  "ROC_AUC": gcn_auc},
        {"Model": "MRA",  "Best_Val_Acc": mra_best_val,  "Best_Val_Epoch": mra_best_epoch,  "Test_Acc": mra_test_acc,  "ROC_AUC": mra_auc},
        {"Model": "GAT",  "Best_Val_Acc": gat_best_val,  "Best_Val_Epoch": gat_best_epoch,  "Test_Acc": gat_test_acc,  "ROC_AUC": gat_auc},
        {"Model": "R-GCN","Best_Val_Acc": rgcn_best_val, "Best_Val_Epoch": rgcn_best_epoch, "Test_Acc": rgcn_test_acc, "ROC_AUC": rgcn_auc},
    ])
    print("\n===== SUMMARY TABLE =====")
    print(results_df)

    results_csv_path = os.path.join("results", "results_gcn_mra_gat_rgcn.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nSaved results to {results_csv_path}")

    # 8) VẼ BIỂU ĐỒ TRAIN/VAL ACC
    def plot_history(hist, title, save_name):
        epochs = hist["epoch"]
        plt.figure()
        plt.plot(epochs, hist["train_acc"], label="Train Acc")
        plt.plot(epochs, hist["val_acc"], label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("results", save_name))
        plt.close()

    plot_history(gcn_hist,  "GCN Train/Val Accuracy",  "gcn_train_val_acc.png")
    plot_history(mra_hist,  "MRA Train/Val Accuracy",  "mra_train_val_acc.png")
    plot_history(gat_hist,  "GAT Train/Val Accuracy",  "gat_train_val_acc.png")
    plot_history(rgcn_hist, "R-GCN Train/Val Accuracy","rgcn_train_val_acc.png")

    # 9) VẼ ROC CURVE (4 mô hình)
    gcn_fpr,  gcn_tpr  = gcn_roc
    mra_fpr,  mra_tpr  = mra_roc
    gat_fpr,  gat_tpr  = gat_roc
    rgcn_fpr, rgcn_tpr = rgcn_roc

    plt.figure()
    plt.plot(gcn_fpr,  gcn_tpr,  label=f"GCN (AUC={gcn_auc:.3f})")
    plt.plot(mra_fpr,  mra_tpr,  label=f"MRA (AUC={mra_auc:.3f})")
    plt.plot(gat_fpr,  gat_tpr,  label=f"GAT (AUC={gat_auc:.3f})")
    plt.plot(rgcn_fpr, rgcn_tpr, label=f"R-GCN (AUC={rgcn_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Diabetes Classification (All Models)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("results", "roc_curves_all.png"))
    plt.close()


# =========================
# Added: Train/Eval helpers for GCN/GAT/R-GCN alongside existing GCN/MRA
# =========================
def _train_one_generic(model, X, y, idx_train, idx_val, adjs, name, epochs=200, lr=1e-2, weight_decay=5e-4, ckpt=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_ep = 0.0, -1
    hist = {"epoch": [], "loss": [], "train_acc": [], "val_acc": []}
    for ep in range(1, epochs+1):
        model.train(); opt.zero_grad()
        out = model(X, adjs if isinstance(adjs, (list,tuple)) else adjs)
        loss = F.cross_entropy(out[idx_train], y[idx_train])
        loss.backward(); opt.step()
        with torch.no_grad():
            tr = (out[idx_train].argmax(1) == y[idx_train]).float().mean().item()
            va = (out[idx_val].argmax(1) == y[idx_val]).float().mean().item()
        hist["epoch"].append(ep); hist["loss"].append(loss.item()); hist["train_acc"].append(tr); hist["val_acc"].append(va)
        if va > best:
            best, best_ep = va, ep
            if ckpt: torch.save(model.state_dict(), ckpt)
        if ep % 20 == 0 or ep == epochs:
            print(f"[{name}] ep={ep:3d} loss={loss.item():.4f} train={tr:.4f} val={va:.4f}")
    print(f"[{name}] BEST Val Acc={best:.4f} @ {best_ep}")
    return hist, best, best_ep

def _eval_generic(model, X, y, idx_test, adjs, name):
    model.eval()
    with torch.no_grad():
        out = model(X, adjs if isinstance(adjs,(list,tuple)) else adjs)
    logits = out[idx_test]
    pred = logits.argmax(1)
    y_true = y[idx_test]
    acc = (pred == y_true).float().mean().item()
    probs = torch.softmax(logits, dim=1)[:,1].cpu().numpy()
    auc = roc_auc_score(y_true.cpu().numpy(), probs)
    fpr, tpr, _ = roc_curve(y_true.cpu().numpy(), probs)
    print(f"\\n===== {name} =====\\nAcc={acc:.4f} AUC={auc:.4f}")
    print(classification_report(y_true.cpu(), pred.cpu()))
    print(confusion_matrix(y_true.cpu(), pred.cpu()))
    return acc, auc, (fpr, tpr)

def main_compare_all():
    from preparedata import load_nhanes_patient_graph
    from gcn_model import GCN
    # MultiRelationalAttentionNet must already be defined in this file
    try:
        MultiRelationalAttentionNet
    except NameError:
        raise RuntimeError("MultiRelationalAttentionNet not found in this script; cannot run MRA baseline.")

    os.makedirs("results", exist_ok=True)
    data_dir = "./data"
    X, y, idx_train, idx_val, idx_test, adj_knn = load_nhanes_patient_graph(data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = X.to(device), y.to(device)
    idx_train, idx_val, idx_test = idx_train.to(device), idx_val.to(device), idx_test.to(device)
    adj_knn = adj_knn.coalesce().to(device)

    print("Building second relation graph (k=25) for multi-relation models...")
    adj_knn_25 = build_knn_adj_from_X(X.detach().cpu(), k=25).coalesce().to(device)

    in_feats = X.shape[1]
    num_classes = 2

    setups = [
        ("GCN", GCN(in_feats, 64, num_classes).to(device), adj_knn, "results/best_gcn.pth"),
        ("MRA", MultiRelationalAttentionNet(in_feats, 64, num_classes, num_relations=2).to(device), [adj_knn, adj_knn_25], "results/best_mra.pth"),
        ("GAT", GAT(in_feats, 64, num_classes).to(device), adj_knn, "results/best_gat.pth"),
        ("RGCN", RGCN(in_feats, 64, num_classes, num_relations=2).to(device), [adj_knn, adj_knn_25], "results/best_rgcn.pth"),
    ]

    results = []
    rocs = {}
    histories = {}

    for name, model, adjs, ckpt in setups:
        print(f"\\n===== TRAINING {name} =====")
        hist, best, ep = _train_one_generic(model, X, y, idx_train, idx_val, adjs, name, epochs=200, lr=1e-2, weight_decay=5e-4, ckpt=ckpt)
        histories[name] = hist
        model.load_state_dict(torch.load(ckpt, map_location=device))
        acc, auc, roc = _eval_generic(model, X, y, idx_test, adjs, name)
        results.append({"Model": name, "Best_Val_Acc": best, "Best_Val_Epoch": ep, "Test_Acc": acc, "ROC_AUC": auc})
        rocs[name] = roc

    df = pd.DataFrame(results)
    df.to_csv("results/results_gcn_mra_gat_rgcn.csv", index=False)
    print("\\n===== SUMMARY TABLE ====="); print(df)

    def _plot_hist(hist, title, path):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(hist["epoch"], hist["train_acc"], label="Train Acc")
        plt.plot(hist["epoch"], hist["val_acc"], label="Val Acc")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title(title)
        plt.legend(); plt.tight_layout(); plt.savefig(path); plt.close()

    for name, hist in histories.items():
        _plot_hist(hist, f"{name} Train/Val Accuracy", f"results/{name.lower()}_train_val_acc.png")

    import matplotlib.pyplot as plt
    plt.figure()
    for name,(fpr,tpr) in rocs.items():
        auc = [r["ROC_AUC"] for r in results if r["Model"]==name][0]
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve - GCN vs MRA vs GAT vs R-GCN")
    plt.legend(); plt.tight_layout(); plt.savefig("results/roc_gcn_mra_gat_rgcn.png"); plt.close()


if __name__ == "__main__":
    import sys
    if "--compare-all" in sys.argv:
        main_compare_all()
    else:
        main()
