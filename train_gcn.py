import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from preparedata import load_nhanes_patient_graph
from gcn_model import GCN

# ===========================================================
# LOAD DATA
# ===========================================================
data_dir = "./data"
X, y, train_idx, val_idx, test_idx, adj = load_nhanes_patient_graph(data_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X = X.to(device)
y = y.to(device)
adj = adj.to(device)
train_idx = train_idx.to(device)
val_idx = val_idx.to(device)
test_idx = test_idx.to(device)

# ===========================================================
# INIT MODEL
# ===========================================================
model = GCN(
    in_feats=X.shape[1],
    h_feats=64,
    num_classes=2
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# ===========================================================
# TRAIN LOOP
# ===========================================================
epochs = 200
train_losses, val_losses = [], []
train_accs, val_accs = [], []

best_val_acc = 0
best_epoch = 0

print("===== TRAINING GCN =====")

for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()

    out = model(X, adj)
    loss = F.cross_entropy(out[train_idx], y[train_idx])
    loss.backward()
    optimizer.step()

    # Train accuracy
    train_pred = out[train_idx].argmax(dim=1)
    train_acc = (train_pred == y[train_idx]).float().mean().item()

    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = out[val_idx].argmax(dim=1)
        val_acc = (val_pred == y[val_idx]).float().mean().item()
        val_loss = F.cross_entropy(out[val_idx], y[val_idx]).item()

    # Save logs
    train_losses.append(loss.item())
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    # Track best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch
        torch.save(model.state_dict(), "best_gcn_model.pth")

    # Print log
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Loss={loss.item():.4f} | Train Acc={train_acc:.4f} | Val Acc={val_acc:.4f}")

print(f"\nBest Val Accuracy: {best_val_acc:.5f} at epoch {best_epoch}")
print("Saved best model to best_gcn_model.pth")

# ===========================================================
# PLOT TRAIN vs VAL ACCURACY
# ===========================================================
plt.figure(figsize=(8,5))
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.title("Train vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.savefig("acc_curve.png", dpi=200)
plt.close()

# ===========================================================
# PLOT TRAIN vs VAL LOSS
# ===========================================================
plt.figure(figsize=(8,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Train vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig("loss_curve.png", dpi=200)
plt.close()

# ===========================================================
# PLOT ROC CURVE
# ===========================================================
model.load_state_dict(torch.load("best_gcn_model.pth"))
model.eval()

with torch.no_grad():
    out = model(X, adj)
    probs = F.softmax(out, dim=1)[:,1].cpu().numpy()

y_true = y.cpu().numpy()
fpr, tpr, _ = roc_curve(y_true, probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1], [0,1], 'k--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid()
plt.savefig("roc_curve.png", dpi=200)
plt.close()

print("Saved acc_curve.png, loss_curve.png, roc_curve.png")
