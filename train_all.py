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
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt
import pandas as pd

from gat_model import GAT
from rgcn_model import RGCN
from medgcn_model import MedGCN
from medgcn_relatt_model import MedGCNRelationAttention

from preparedata import load_nhanes_multirel_patient_graph


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


### Note:
### In this updated version, "multi-relation" means per-modality relations
### (labs/diet/exam/meds/questionnaire), not kNN(k=10) vs kNN(k=25).


# ============================================================
# 4. HÀM TRAIN 1 MODEL
# ============================================================
def train_one_model(
    model,
    X,
    y,
    train_idx,
    val_idx,
    adjs,
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
        "train_auc": [],
        "val_auc": [],
        "train_f1": [],
        "val_f1": [],
    }

    best_val_score = -1.0
    best_epoch = -1

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()

        # forward
        if len(adjs) == 1:
            out = model(X, adjs[0])
        else:
            out = model(X, adjs)

        # unwrap
        if isinstance(out, tuple):
            logits, alpha = out
        else:
            logits, alpha = out, None

        # loss on logits
        loss = F.cross_entropy(logits[train_idx], y[train_idx])

        # entropy regularization (only if alpha exists + only for MedGCN_RelAtt)
        if alpha is not None and model_name == "MedGCN_RelAtt":
            eps = 1e-8
            ent = -(alpha[train_idx] * (alpha[train_idx] + eps).log()).sum(dim=1).mean()
            loss = loss + 0.02 * (-ent)  # lambda: 0.01–0.05, khuyên 0.02

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # accuracy
            pred_train = logits[train_idx].argmax(dim=1)
            train_acc = (pred_train == y[train_idx]).float().mean().item()

            pred_val = logits[val_idx].argmax(dim=1)
            val_acc = (pred_val == y[val_idx]).float().mean().item()

            # AUC / F1
            train_probs = F.softmax(logits[train_idx], dim=1)[:, 1].detach().cpu().numpy()
            val_probs = F.softmax(logits[val_idx], dim=1)[:, 1].detach().cpu().numpy()

            y_train_np = y[train_idx].detach().cpu().numpy()
            y_val_np = y[val_idx].detach().cpu().numpy()

            train_auc = roc_auc_score(y_train_np, train_probs) if len(np.unique(y_train_np)) > 1 else float("nan")
            val_auc = roc_auc_score(y_val_np, val_probs) if len(np.unique(y_val_np)) > 1 else float("nan")

            train_f1 = precision_recall_fscore_support(
                y_train_np, pred_train.detach().cpu().numpy(), average="binary", zero_division=0
            )[2]
            val_f1 = precision_recall_fscore_support(
                y_val_np, pred_val.detach().cpu().numpy(), average="binary", zero_division=0
            )[2]

        history["epoch"].append(epoch)
        history["loss"].append(loss.item())
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_auc"].append(float(train_auc))
        history["val_auc"].append(float(val_auc))
        history["train_f1"].append(float(train_f1))
        history["val_f1"].append(float(val_f1))

        # choose best by val AUC first; fallback to val acc if AUC NaN
        score = val_auc if val_auc == val_auc else val_acc
        if score > best_val_score:
            best_val_score = score
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)

        if epoch % 20 == 0 or epoch == num_epochs:
            print(
                f"[{model_name}] Epoch {epoch:3d} | "
                f"Loss={loss.item():.4f} | "
                f"Train Acc={train_acc:.4f} AUC={train_auc:.4f} F1={train_f1:.4f} | "
                f"Val Acc={val_acc:.4f} AUC={val_auc:.4f} F1={val_f1:.4f}"
            )

    print(f"[{model_name}] BEST Val Score={best_val_score:.4f} at epoch {best_epoch}")
    return history, best_val_score, best_epoch


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

        # unwrap
        if isinstance(out, tuple):
            logits, _ = out
        else:
            logits = out

    logits_test = logits[test_idx]
    preds_test = logits_test.argmax(dim=1)
    y_true = y[test_idx]

    test_acc = (preds_test == y_true).float().mean().item()

    y_np = y_true.cpu().numpy()
    pred_np = preds_test.cpu().numpy()
    prec, rec, f1, _ = precision_recall_fscore_support(y_np, pred_np, average="binary", zero_division=0)

    print(f"\n===== EVALUATION: {model_name} =====")
    print("Test Accuracy:", test_acc)
    print("\nClassification Report:")
    print(classification_report(y_true.cpu(), preds_test.cpu()))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true.cpu(), preds_test.cpu()))

    probs = F.softmax(logits_test, dim=1)[:, 1].cpu().numpy()
    auc = roc_auc_score(y_true.cpu().numpy(), probs)
    print("ROC-AUC:", auc)
    print(f"Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f}")

    fpr, tpr, _ = roc_curve(y_true.cpu().numpy(), probs)
    return test_acc, auc, f1, prec, rec, (fpr, tpr)


# ============================================================
# 6. MAIN: TRAIN 2 MODEL + VẼ BIỂU ĐỒ + BẢNG KẾT QUẢ
# ============================================================
def main():
    os.makedirs("results", exist_ok=True)

    # 1) Load data (multi-relation by modality)
    data_dir = "./data"
    X, y, train_idx, val_idx, test_idx, adjs, agg_adj, rel_names = load_nhanes_multirel_patient_graph(
        data_dir,
        relations=("labs", "diet", "exam", "meds", "ques"),
        k_neighbors=10,
        knn_metric="cosine",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X.to(device)
    y = y.to(device)
    adjs = [a.to(device) for a in adjs]
    agg_adj = agg_adj.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)

    print(f"Loaded relations: {rel_names} (R={len(rel_names)})")

    # 3) Init models
    in_feats = X.shape[1]
    num_classes = 2

    gcn = GCN(in_feats=in_feats, h_feats=64, num_classes=num_classes).to(device)
    gat = GAT(in_feats=in_feats, h_feats=64, num_classes=num_classes).to(device)

    # MedGCN baseline (multi-relation, no attention)
    medgcn = MedGCN(in_feats=in_feats, h_feats=64, num_classes=num_classes, num_relations=len(adjs)).to(device)

    # MedGCN + relation-level attention
    medgcn_relatt = MedGCNRelationAttention(
        in_feats=in_feats,
        h_feats=64,
        num_classes=num_classes,
        num_relations=len(adjs),
    ).to(device)

    # R-GCN baseline (multi-relation)
    rgcn = RGCN(in_feats=in_feats, h_feats=64, num_classes=num_classes, num_relations=len(adjs)).to(device)

    # 4) Train GCN (single aggregated graph)
    # print("\n===== TRAINING GCN (single graph) =====")
    # gcn_hist, gcn_best_val, gcn_best_epoch = train_one_model(
    #     gcn,
    #     X,
    #     y,
    #     train_idx,
    #     val_idx,
    #     adjs=[agg_adj],
    #     num_epochs=200,
    #     lr=0.01,
    #     weight_decay=5e-4,
    #     model_name="GCN",
    #     save_path="results/best_gcn_model.pth",
    # )

    # 5) Train MedGCN baseline
    print("\n===== TRAINING MedGCN =====")
    medgcn_hist, medgcn_best_val, medgcn_best_epoch = train_one_model(
        medgcn,
        X,
        y,
        train_idx,
        val_idx,
        adjs=adjs,
        num_epochs=200,
        lr=0.01,
        weight_decay=5e-4,
        model_name="MedGCN",
        save_path="results/best_medgcn_model.pth",
    )

    # 6) Train MedGCN + relation attention
    print("\n===== TRAINING MedGCN + Relation-Attention =====")
    relatt_hist, relatt_best_val, relatt_best_epoch = train_one_model(
        medgcn_relatt,
        X,
        y,
        train_idx,
        val_idx,
        adjs=adjs,
        num_epochs=200,
        lr=0.01,
        weight_decay=5e-4,
        model_name="MedGCN_RelAtt",
        save_path="results/best_medgcn_relatt_model.pth",
    )

    # 7) Train GAT (single aggregated graph)
    print("\n===== TRAINING GAT (single graph) =====")
    gat_hist, gat_best_val, gat_best_epoch = train_one_model(
        gat,
        X,
        y,
        train_idx,
        val_idx,
        adjs=[agg_adj],
        num_epochs=200,
        lr=0.01,
        weight_decay=5e-4,
        model_name="GAT",
        save_path="results/best_gat_model.pth",
    )

    # 8) Train R-GCN (multi-relation)
    print("\n===== TRAINING R-GCN (multi-relation) =====")
    rgcn_hist, rgcn_best_val, rgcn_best_epoch = train_one_model(
        rgcn,
        X,
        y,
        train_idx,
        val_idx,
        adjs=adjs,
        num_epochs=200,
        lr=0.01,
        weight_decay=5e-4,
        model_name="R-GCN",
        save_path="results/best_rgcn_model.pth",
    )

    # 7) Load best weights + Evaluate on test
    # gcn.load_state_dict(torch.load("results/best_gcn_model.pth", map_location=device))
    medgcn.load_state_dict(torch.load("results/best_medgcn_model.pth", map_location=device))
    medgcn_relatt.load_state_dict(torch.load("results/best_medgcn_relatt_model.pth", map_location=device))
    gat.load_state_dict(torch.load("results/best_gat_model.pth", map_location=device))
    rgcn.load_state_dict(torch.load("results/best_rgcn_model.pth", map_location=device))

    # gcn_test_acc, gcn_auc, gcn_f1, gcn_prec, gcn_rec, gcn_roc = evaluate_model(gcn,  X, y, test_idx, [agg_adj], model_name="GCN")
    med_test_acc, med_auc, med_f1, med_prec, med_rec, med_roc = evaluate_model(medgcn,  X, y, test_idx, adjs, model_name="MedGCN")
    rel_test_acc, rel_auc, rel_f1, rel_prec, rel_rec, rel_roc = evaluate_model(medgcn_relatt,  X, y, test_idx, adjs, model_name="MedGCN_RelAtt")
    gat_test_acc, gat_auc, gat_f1, gat_prec, gat_rec, gat_roc = evaluate_model(gat,  X, y, test_idx, [agg_adj], model_name="GAT")
    rgcn_test_acc, rgcn_auc, rgcn_f1, rgcn_prec, rgcn_rec, rgcn_roc = evaluate_model(rgcn, X, y, test_idx, adjs, model_name="R-GCN")

    # 7) BẢNG KẾT QUẢ
    results_df = pd.DataFrame([
        # {"Model": "GCN",            "Best_Val_Score": gcn_best_val,     "Best_Epoch": gcn_best_epoch,     "Test_Acc": gcn_test_acc,   "ROC_AUC": gcn_auc,   "F1": gcn_f1,   "Precision": gcn_prec,   "Recall": gcn_rec},
        {"Model": "MedGCN",         "Best_Val_Score": medgcn_best_val,  "Best_Epoch": medgcn_best_epoch,  "Test_Acc": med_test_acc,   "ROC_AUC": med_auc,   "F1": med_f1,   "Precision": med_prec,   "Recall": med_rec},
        {"Model": "MedGCN_RelAtt",  "Best_Val_Score": relatt_best_val,  "Best_Epoch": relatt_best_epoch,  "Test_Acc": rel_test_acc,   "ROC_AUC": rel_auc,   "F1": rel_f1,   "Precision": rel_prec,   "Recall": rel_rec},
        {"Model": "GAT",            "Best_Val_Score": gat_best_val,     "Best_Epoch": gat_best_epoch,     "Test_Acc": gat_test_acc,   "ROC_AUC": gat_auc,   "F1": gat_f1,   "Precision": gat_prec,   "Recall": gat_rec},
        {"Model": "R-GCN",          "Best_Val_Score": rgcn_best_val,    "Best_Epoch": rgcn_best_epoch,    "Test_Acc": rgcn_test_acc,  "ROC_AUC": rgcn_auc,  "F1": rgcn_f1,  "Precision": rgcn_prec,  "Recall": rgcn_rec},
    ])
    print("\n===== SUMMARY TABLE =====")
    print(results_df)

    results_csv_path = os.path.join("results", "results_all_models.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nSaved results to {results_csv_path}")

    # 8) PLOTS: Train/Val curves (Acc/AUC/F1) + Loss
    def plot_history(hist, title_prefix, save_prefix):
        epochs = hist["epoch"]
        # Acc
        plt.figure()
        plt.plot(epochs, hist["train_acc"], label="Train")
        plt.plot(epochs, hist["val_acc"], label="Val")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title(f"{title_prefix} Accuracy")
        plt.legend(); plt.tight_layout(); plt.savefig(os.path.join("results", f"{save_prefix}_acc.png")); plt.close()

        # AUC
        plt.figure()
        plt.plot(epochs, hist["train_auc"], label="Train")
        plt.plot(epochs, hist["val_auc"], label="Val")
        plt.xlabel("Epoch"); plt.ylabel("ROC-AUC"); plt.title(f"{title_prefix} ROC-AUC")
        plt.legend(); plt.tight_layout(); plt.savefig(os.path.join("results", f"{save_prefix}_auc.png")); plt.close()

        # F1
        plt.figure()
        plt.plot(epochs, hist["train_f1"], label="Train")
        plt.plot(epochs, hist["val_f1"], label="Val")
        plt.xlabel("Epoch"); plt.ylabel("F1"); plt.title(f"{title_prefix} F1")
        plt.legend(); plt.tight_layout(); plt.savefig(os.path.join("results", f"{save_prefix}_f1.png")); plt.close()

        # Loss
        plt.figure()
        plt.plot(epochs, hist["loss"], label="Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(f"{title_prefix} Loss")
        plt.legend(); plt.tight_layout(); plt.savefig(os.path.join("results", f"{save_prefix}_loss.png")); plt.close()

    # plot_history(gcn_hist,        "GCN",           "gcn")
    plot_history(medgcn_hist,     "MedGCN",        "medgcn")
    plot_history(relatt_hist,     "MedGCN_RelAtt", "medgcn_relatt")
    plot_history(gat_hist,        "GAT",           "gat")
    plot_history(rgcn_hist,       "R-GCN",         "rgcn")

    # 9) ROC CURVES
    # gcn_fpr,  gcn_tpr  = gcn_roc
    med_fpr,  med_tpr  = med_roc
    rel_fpr,  rel_tpr  = rel_roc
    gat_fpr,  gat_tpr  = gat_roc
    rgcn_fpr, rgcn_tpr = rgcn_roc

    plt.figure()
    # plt.plot(gcn_fpr,  gcn_tpr,  label=f"GCN (AUC={gcn_auc:.3f})")
    plt.plot(med_fpr,  med_tpr,  label=f"MedGCN (AUC={med_auc:.3f})")
    plt.plot(rel_fpr,  rel_tpr,  label=f"MedGCN_RelAtt (AUC={rel_auc:.3f})")
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

    # 10) BAR CHART: test metrics
    plt.figure()
    results_df.set_index("Model")[["Test_Acc", "ROC_AUC", "F1"]].plot(kind="bar")
    plt.title("Test metrics comparison")
    plt.tight_layout()
    plt.savefig(os.path.join("results", "test_metrics_bar.png"))
    plt.close()


if __name__ == "__main__":
    main()
