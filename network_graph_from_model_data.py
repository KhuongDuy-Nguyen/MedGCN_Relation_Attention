import os
import torch
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.decomposition import PCA
from preparedata import load_nhanes_patient_graph

DATA_DIR = "./data"

# Load processed NHANES patient graph
X, y, train_idx, val_idx, test_idx, adj = load_nhanes_patient_graph(DATA_DIR)

# Convert sparse adjacency to edge list
indices = adj.indices().cpu().numpy()
rows, cols = indices[0], indices[1]

# ---- SMART SUBSET USING PCA (to avoid isolated nodes) ----
X_np = X.numpy()
X_2d = PCA(n_components=2).fit_transform(X_np)

subset_size = 150
subset_ids = X_2d[:,0].argsort()[:subset_size]
subset_ids = set(subset_ids.tolist())

# Filter edges to subset
filtered_edges = [
    (int(r), int(c))
    for r, c in zip(rows, cols)
    if int(r) in subset_ids and int(c) in subset_ids
]

# Create graph
G = nx.Graph()
G.add_nodes_from(list(subset_ids))
G.add_edges_from(filtered_edges)

# Color nodes by diabetes label
colors = ["red" if int(y[n].item()) == 1 else "blue" for n in G.nodes]

# Draw network
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(G, seed=42, k=0.25)

nx.draw(
    G, pos,
    node_color=colors,
    node_size=70,
    edge_color="lightgray",
    linewidths=0.3
)

plt.title("NHANES Patient Graph (PCA Subset) — kNN Edges From Model")
plt.tight_layout()

os.makedirs("images", exist_ok=True)
plt.savefig("images/network_graph_final.png", dpi=200)
plt.close()

print("Saved: images/network_graph_final.png")
