import os
import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

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
# Utility
# ============================================================

def set_seed(seed: int) -> None:
    """Set random seed for python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_seeds(seeds_str: str) -> List[int]:
    parts = [p.strip() for p in seeds_str.split(",") if p.strip()]
    if not parts:
        return [42]
    return [int(p) for p in parts]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ============================================================
# 1. Simple GCN baseline (single aggregated graph)
# ============================================================


class GCN(nn.Module):
    def __init__(self, in_feats: int, h_feats: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(in_feats, h_feats)
        self.fc2 = nn.Linear(h_feats, num_classes)

    def forward(self, X: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = torch.spmm(adj, X)
        h = F.relu(self.fc1(h))
        h = F.dropout(h, p=0.5, training=self.training)
        h = torch.spmm(adj, h)
        h = self.fc2(h)
        return h


# ============================================================
# 2. Train one model
# ============================================================


def train_one_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    adjs: List[torch.Tensor],
    num_epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    model_name: str = "GCN",
    save_path: str = "best_model.pth",
) -> Tuple[Dict[str, List[float]], float, int]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: Dict[str, List[float]] = {
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

        loss = F.cross_entropy(logits[train_idx], y[train_idx])

        # entropy regularization (only if alpha exists + only for MedGCN_RelAtt)
        if alpha is not None and model_name == "MedGCN_RelAtt":
            eps = 1e-8
            ent = -(alpha[train_idx] * (alpha[train_idx] + eps).log()).sum(dim=1).mean()
            loss = loss + 0.02 * (-ent)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred_train = logits[train_idx].argmax(dim=1)
            train_acc = (pred_train == y[train_idx]).float().mean().item()

            pred_val = logits[val_idx].argmax(dim=1)
            val_acc = (pred_val == y[val_idx]).float().mean().item()

            train_probs = F.softmax(logits[train_idx], dim=1)[:, 1].detach().cpu().numpy()
            val_probs = F.softmax(logits[val_idx], dim=1)[:, 1].detach().cpu().numpy()

            y_train_np = y[train_idx].detach().cpu().numpy()
            y_val_np = y[val_idx].detach().cpu().numpy()

            train_auc = roc_auc_score(y_train_np, train_probs) if len(np.unique(y_train_np)) > 1 else float("nan")
            val_auc = roc_auc_score(y_val_np, val_probs) if len(np.unique(y_val_np)) > 1 else float("nan")

            train_f1 = precision_recall_fscore_support(
                y_train_np,
                pred_train.detach().cpu().numpy(),
                average="binary",
                zero_division=0,
            )[2]
            val_f1 = precision_recall_fscore_support(
                y_val_np,
                pred_val.detach().cpu().numpy(),
                average="binary",
                zero_division=0,
            )[2]

        history["epoch"].append(epoch)
        history["loss"].append(float(loss.item()))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))
        history["train_auc"].append(float(train_auc))
        history["val_auc"].append(float(val_auc))
        history["train_f1"].append(float(train_f1))
        history["val_f1"].append(float(val_f1))

        score = val_auc if val_auc == val_auc else val_acc
        if score > best_val_score:
            best_val_score = float(score)
            best_epoch = int(epoch)
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
# 3. Evaluate model (adds confusion matrix + minority recall)
# ============================================================


def _minority_label_and_recall(y_true_np: np.ndarray, y_pred_np: np.ndarray) -> Tuple[int, float, np.ndarray]:
    """Return (minority_label, minority_recall, confusion_matrix_2x2)."""
    labels = [0, 1]
    cm = confusion_matrix(y_true_np, y_pred_np, labels=labels)

    counts = np.bincount(y_true_np.astype(int), minlength=2)
    minority_label = int(np.argmin(counts))

    # recall for a class c = TP_c / (TP_c + FN_c)
    # For 2x2 confusion_matrix with rows=true, cols=pred:
    # class 0: TP = cm[0,0], FN = cm[0,1]
    # class 1: TP = cm[1,1], FN = cm[1,0]
    if minority_label == 0:
        denom = cm[0, 0] + cm[0, 1]
        minority_recall = (cm[0, 0] / denom) if denom > 0 else 0.0
    else:
        denom = cm[1, 1] + cm[1, 0]
        minority_recall = (cm[1, 1] / denom) if denom > 0 else 0.0

    return minority_label, float(minority_recall), cm


def evaluate_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    test_idx: torch.Tensor,
    adjs: List[torch.Tensor],
    model_name: str = "GCN",
    verbose: bool = True,
) -> Dict[str, object]:
    model.eval()
    with torch.no_grad():
        if len(adjs) == 1:
            out = model(X, adjs[0])
        else:
            out = model(X, adjs)

        if isinstance(out, tuple):
            logits, _ = out
        else:
            logits = out

    logits_test = logits[test_idx]
    preds_test = logits_test.argmax(dim=1)
    y_true = y[test_idx]

    y_np = y_true.detach().cpu().numpy()
    pred_np = preds_test.detach().cpu().numpy()

    test_acc = float((preds_test == y_true).float().mean().item())

    prec, rec, f1, _ = precision_recall_fscore_support(y_np, pred_np, average="binary", zero_division=0)

    probs = F.softmax(logits_test, dim=1)[:, 1].detach().cpu().numpy()
    auc = float(roc_auc_score(y_np, probs)) if len(np.unique(y_np)) > 1 else float("nan")

    minority_label, minority_recall, cm = _minority_label_and_recall(y_np, pred_np)

    fpr, tpr, _ = roc_curve(y_np, probs) if len(np.unique(y_np)) > 1 else (np.array([]), np.array([]), np.array([]))

    if verbose:
        print(f"\n===== EVALUATION: {model_name} =====")
        print(f"Test Accuracy: {test_acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true.cpu(), preds_test.cpu(), zero_division=0))
        print("Confusion Matrix (rows=true, cols=pred):")
        print(cm)
        print(f"ROC-AUC: {auc:.4f}")
        print(f"Precision={prec:.4f} Recall={rec:.4f} F1={f1:.4f}")
        print(f"Minority label={minority_label} | Minority recall={minority_recall:.4f}")

    return {
        "Test_Acc": test_acc,
        "ROC_AUC": auc,
        "F1": float(f1),
        "Precision": float(prec),
        "Recall": float(rec),
        "Minority_Label": minority_label,
        "Minority_Recall": minority_recall,
        "Confusion_Matrix": cm,
        "ROC_Curve": (fpr, tpr),
    }


def save_confusion_matrix_plot(cm: np.ndarray, path: str, title: str) -> None:
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_history(hist: Dict[str, List[float]], title_prefix: str, out_dir: str, save_prefix: str) -> None:
    epochs = hist["epoch"]

    # Accuracy
    plt.figure()
    plt.plot(epochs, hist["train_acc"], label="Train")
    plt.plot(epochs, hist["val_acc"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title_prefix} Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{save_prefix}_acc.png"))
    plt.close()

    # AUC
    plt.figure()
    plt.plot(epochs, hist["train_auc"], label="Train")
    plt.plot(epochs, hist["val_auc"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("ROC-AUC")
    plt.title(f"{title_prefix} ROC-AUC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{save_prefix}_auc.png"))
    plt.close()

    # F1
    plt.figure()
    plt.plot(epochs, hist["train_f1"], label="Train")
    plt.plot(epochs, hist["val_f1"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title(f"{title_prefix} F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{save_prefix}_f1.png"))
    plt.close()

    # Loss
    plt.figure()
    plt.plot(epochs, hist["loss"], label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{save_prefix}_loss.png"))
    plt.close()


@dataclass
class ModelSpec:
    name: str
    uses_multirel: bool  # True -> pass adjs (list); False -> pass [agg_adj]


def build_model(name: str, in_feats: int, num_classes: int, num_relations: int, device: torch.device) -> nn.Module:
    if name == "GCN":
        return GCN(in_feats=in_feats, h_feats=64, num_classes=num_classes).to(device)
    if name == "GAT":
        return GAT(in_feats=in_feats, h_feats=64, num_classes=num_classes).to(device)
    if name == "MedGCN":
        return MedGCN(in_feats=in_feats, h_feats=64, num_classes=num_classes, num_relations=num_relations).to(device)
    if name == "MedGCN_RelAtt":
        return MedGCNRelationAttention(
            in_feats=in_feats,
            h_feats=64,
            num_classes=num_classes,
            num_relations=num_relations,
        ).to(device)
    if name == "R-GCN":
        return RGCN(in_feats=in_feats, h_feats=64, num_classes=num_classes, num_relations=num_relations).to(device)
    raise ValueError(f"Unknown model name: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate MedGCN variants on NHANES (supports multi-run seeds).")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--seeds", type=str, default="42", help='Comma-separated seeds, e.g. "0,1,2,3,4"')
    parser.add_argument("--split_seed", type=int, default=42, help="Seed for train/val/test split.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument(
        "--models",
        type=str,
        default="MedGCN,MedGCN_RelAtt,GAT,R-GCN",
        help='Comma-separated model names from: GCN,GAT,MedGCN,MedGCN_RelAtt,R-GCN',
    )
    parser.add_argument("--no_plots", action="store_true", help="Skip plotting training curves and ROC curves.")
    parser.add_argument("--k_neighbors", type=int, default=10)
    parser.add_argument("--knn_metric", type=str, default="cosine")

    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    ensure_dir(args.results_dir)

    # Load data once (split controlled by split_seed)
    X, y, train_idx, val_idx, test_idx, adjs, agg_adj, rel_names = load_nhanes_multirel_patient_graph(
        args.data_dir,
        relations=("labs", "diet", "exam", "meds", "ques"),
        k_neighbors=args.k_neighbors,
        knn_metric=args.knn_metric,
        random_state=args.split_seed,
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
    print(f"Train/Val/Test sizes: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    in_feats = int(X.shape[1])
    num_classes = 2
    num_relations = len(adjs)

    specs = {
        "GCN": ModelSpec(name="GCN", uses_multirel=False),
        "GAT": ModelSpec(name="GAT", uses_multirel=False),
        "MedGCN": ModelSpec(name="MedGCN", uses_multirel=True),
        "MedGCN_RelAtt": ModelSpec(name="MedGCN_RelAtt", uses_multirel=True),
        "R-GCN": ModelSpec(name="R-GCN", uses_multirel=True),
    }

    for m in model_names:
        if m not in specs:
            raise ValueError(f"Unknown model '{m}'. Allowed: {', '.join(specs.keys())}")

    all_runs: List[Dict[str, object]] = []

    for seed in seeds:
        print("\n" + "=" * 70)
        print(f"RUN seed={seed} (split_seed={args.split_seed})")
        print("=" * 70)

        set_seed(seed)
        run_dir = os.path.join(args.results_dir, f"seed_{seed}")
        ensure_dir(run_dir)

        for model_name in model_names:
            spec = specs[model_name]

            # Build a fresh model per run
            model = build_model(model_name, in_feats, num_classes, num_relations, device)

            model_adjs = adjs if spec.uses_multirel else [agg_adj]

            save_path = os.path.join(run_dir, f"best_{model_name.replace('-', '').replace(' ', '')}.pth")

            print(f"\n----- TRAINING {model_name} -----")
            hist, best_val, best_epoch = train_one_model(
                model,
                X,
                y,
                train_idx,
                val_idx,
                adjs=model_adjs,
                num_epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                model_name=model_name,
                save_path=save_path,
            )

            # Reload best weights then evaluate
            model.load_state_dict(torch.load(save_path, map_location=device))
            eval_dict = evaluate_model(model, X, y, test_idx, model_adjs, model_name=model_name, verbose=True)

            # Save confusion matrix plot
            cm = eval_dict["Confusion_Matrix"]
            cm_path = os.path.join(run_dir, f"confusion_matrix_{model_name.replace('-', '').replace(' ', '')}.png")
            save_confusion_matrix_plot(cm, cm_path, title=f"{model_name} Confusion Matrix (seed={seed})")

            # Optional training plots
            if not args.no_plots:
                plot_history(hist, title_prefix=f"{model_name} (seed={seed})", out_dir=run_dir, save_prefix=model_name)

            all_runs.append(
                {
                    "Seed": seed,
                    "Model": model_name,
                    "Best_Val_Score": float(best_val),
                    "Best_Epoch": int(best_epoch),
                    **{k: v for k, v in eval_dict.items() if k not in ["ROC_Curve", "Confusion_Matrix"]},
                    "CM_00": int(cm[0, 0]),
                    "CM_01": int(cm[0, 1]),
                    "CM_10": int(cm[1, 0]),
                    "CM_11": int(cm[1, 1]),
                }
            )

        # Save per-seed table
        df_seed = pd.DataFrame([r for r in all_runs if r["Seed"] == seed])
        df_seed.to_csv(os.path.join(run_dir, "results_seed.csv"), index=False)

    # Save all run-level results
    runs_df = pd.DataFrame(all_runs)
    runs_csv_path = os.path.join(args.results_dir, "results_all_models_runs.csv")
    runs_df.to_csv(runs_csv_path, index=False)
    print(f"\nSaved run-level results to {runs_csv_path}")

    # Aggregate mean/std across seeds
    metric_cols = [
        "Test_Acc",
        "ROC_AUC",
        "F1",
        "Precision",
        "Recall",
        "Minority_Recall",
    ]

    agg_rows = []
    for model_name in model_names:
        sub = runs_df[runs_df["Model"] == model_name]
        row: Dict[str, object] = {"Model": model_name, "Runs": int(len(sub))}
        for c in metric_cols:
            vals = sub[c].astype(float).to_numpy()
            row[f"{c}_Mean"] = float(np.nanmean(vals))
            row[f"{c}_Std"] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        agg_rows.append(row)

    summary_df = pd.DataFrame(agg_rows)
    summary_csv_path = os.path.join(args.results_dir, "results_all_models_mean_std.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    print("\n===== MEAN ± STD OVER SEEDS =====")
    for _, r in summary_df.iterrows():
        m = r["Model"]
        print(
            f"{m:14s} | "
            f"Acc {r['Test_Acc_Mean']:.4f}±{r['Test_Acc_Std']:.4f} | "
            f"AUC {r['ROC_AUC_Mean']:.4f}±{r['ROC_AUC_Std']:.4f} | "
            f"F1 {r['F1_Mean']:.4f}±{r['F1_Std']:.4f} | "
            f"MinorRec {r['Minority_Recall_Mean']:.4f}±{r['Minority_Recall_Std']:.4f}"
        )

    print(f"\nSaved summary results to {summary_csv_path}")


if __name__ == "__main__":
    main()
