import os
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


###############################################################
# SAFE CSV READER
###############################################################
def read_csv_safe(path):
    try:
        return pd.read_csv(path, encoding="utf-8", engine="python")
    except:
        return pd.read_csv(path, encoding="latin-1", engine="python")


###############################################################
# MAIN FUNCTION
###############################################################
def load_nhanes_patient_graph(
    data_dir: str,
    label_col: str = "DIQ010",
    k_neighbors: int = 10,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
):
    """
    Build a patient-centric graph from NHANES:
    - Node: patients (SEQN)
    - Features: merged numeric columns from demographic, diet, exam, labs, meds, questionnaire
    - Label: DIQ010 (1=diabetes, 2=non-diabetes) -> binary
    - Graph: k-NN (patient-patient)

    Return:
        X, y, train_idx, val_idx, test_idx, adj
    """

    ###########################################################
    # LOAD 6 NHANES TABLES
    ###########################################################
    demo = read_csv_safe(os.path.join(data_dir, "demographic.csv"))
    diet = read_csv_safe(os.path.join(data_dir, "diet.csv"))
    exam = read_csv_safe(os.path.join(data_dir, "examination.csv"))
    labs = read_csv_safe(os.path.join(data_dir, "labs.csv"))
    meds = read_csv_safe(os.path.join(data_dir, "medications.csv"))
    ques = read_csv_safe(os.path.join(data_dir, "questionnaire.csv"))


    ###########################################################
    # LABEL: DIABETES
    ###########################################################
    if label_col not in ques.columns:
        raise ValueError(f"Label column '{label_col}' not found in questionnaire.csv")

    ques = ques[["SEQN", label_col]]
    ques = ques[ques[label_col].isin([1, 2])]  # keep only yes/no
    ques["label"] = (ques[label_col] == 1).astype(int)

    df = ques[["SEQN", "label"]].copy()


    ###########################################################
    # MERGE FEATURES
    ###########################################################
    def merge(df, other, suffix):
        other = other.rename(
            columns={c: f"{c}_{suffix}" for c in other.columns if c != "SEQN"}
        )
        return df.merge(other, on="SEQN", how="left")

    df = merge(df, demo, "demo")
    df = merge(df, diet, "diet")
    df = merge(df, exam, "exam")

    ###########################################################
    # LAB FEATURES (wide numeric)
    ###########################################################
    lab_numeric_cols = [
        c for c in labs.columns if c != "SEQN" and pd.api.types.is_numeric_dtype(labs[c])
    ]
    labs_num = labs[["SEQN"] + lab_numeric_cols]
    df = df.merge(labs_num, on="SEQN", how="left")

    ###########################################################
    # MEDICATION FEATURES (long → wide)
    ###########################################################
    if "RXDDRUG" in meds.columns and "RXDUSE" in meds.columns:
        meds_wide = meds.pivot_table(
            index="SEQN",
            columns="RXDDRUG",
            values="RXDUSE",
            aggfunc="max"
        ).fillna(0)
        meds_wide = meds_wide.reset_index()
    else:
        meds_wide = pd.DataFrame({"SEQN": df["SEQN"], "NO_MED": 0})

    df = df.merge(meds_wide, on="SEQN", how="left")


    ###########################################################
    # BUILD FEATURES
    ###########################################################
    feature_cols = [
        c for c in df.columns
        if c not in ["SEQN", "label"] and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(feature_cols) == 0:
        raise RuntimeError("No numeric features found in merged NHANES dataset!")

    features = df[feature_cols].fillna(0.0).astype(np.float32)

    # simple normalization (z-score)
    mean = features.mean()
    std = features.std().replace(0, 1.0)
    features = (features - mean) / std

    X_np = features.to_numpy(dtype=np.float32)
    y_np = df["label"].to_numpy(dtype=np.int64)

    N = X_np.shape[0]
    indices = np.arange(N)


    ###########################################################
    # SPLIT DATA
    ###########################################################
    idx_train_val, idx_test = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y_np,
    )

    val_relative = val_size / (1 - test_size)

    idx_train, idx_val = train_test_split(
        idx_train_val,
        test_size=val_relative,
        random_state=random_state,
        stratify=y_np[idx_train_val],
    )


    ###########################################################
    # BUILD k-NN GRAPH
    ###########################################################
    k = min(k_neighbors, max(1, N - 1))

    nbrs = NearestNeighbors(
        n_neighbors=k + 1,
        metric="euclidean",
        n_jobs=-1
    ).fit(X_np)

    distances, knn_indices = nbrs.kneighbors(X_np)

    # remove self-loop at column 0
    knn_indices = knn_indices[:, 1:]  # shape = [N, k]

    rows = np.repeat(np.arange(N), k)
    cols = knn_indices.reshape(-1)
    values = np.ones(len(rows), dtype=np.float32)

    coords = np.vstack([rows, cols])
    coords_t = torch.tensor(coords, dtype=torch.long)
    values_t = torch.tensor(values, dtype=torch.float32)

    adj = torch.sparse_coo_tensor(
        coords_t,
        values_t,
        size=(N, N)
    ).coalesce()

    # symmetrize
    adj_T = torch.sparse_coo_tensor(
        torch.stack([coords_t[1], coords_t[0]]),
        values_t,
        size=(N, N)
    )

    adj = (adj + adj_T).coalesce()
    mask_vals = torch.where(adj.values() > 0, torch.ones_like(adj.values()), adj.values())
    adj = torch.sparse_coo_tensor(adj.indices(), mask_vals, adj.size()).coalesce()


    ###########################################################
    # RETURN
    ###########################################################
    X = torch.from_numpy(X_np)
    y = torch.from_numpy(y_np)
    train_idx = torch.from_numpy(idx_train)
    val_idx = torch.from_numpy(idx_val)
    test_idx = torch.from_numpy(idx_test)

    return X, y, train_idx, val_idx, test_idx, adj


###############################################################
# MULTI-RELATION PATIENT GRAPH (MODALITY-SPECIFIC)
###############################################################
def _zscore_np(df: pd.DataFrame) -> np.ndarray:
    x = df.fillna(0.0).astype(np.float32)
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0
    return ((x - mu) / sd).to_numpy(np.float32)


def _sym_norm_sparse(A: torch.Tensor) -> torch.Tensor:
    """Symmetric normalization for sparse adjacency."""
    A = A.coalesce()
    deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1e-12)
    invsqrt = deg.pow(-0.5)
    i, j = A.indices()
    v = A.values() * invsqrt[i] * invsqrt[j]
    return torch.sparse_coo_tensor(A.indices(), v, A.size()).coalesce()


def _build_knn_adj(X_np: np.ndarray, k: int, metric: str = "cosine") -> torch.Tensor:
    """Build undirected kNN sparse adjacency with self-loops (un-normalized)."""
    N = X_np.shape[0]
    k = min(k, max(1, N - 1))

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric, n_jobs=-1).fit(X_np)
    _, knn_indices = nbrs.kneighbors(X_np)
    knn_indices = knn_indices[:, 1:]

    rows = np.repeat(np.arange(N), k)
    cols = knn_indices.reshape(-1)
    vals = np.ones(len(rows), dtype=np.float32)
    coords = np.vstack([rows, cols])
    coords_t = torch.tensor(coords, dtype=torch.long)
    values_t = torch.tensor(vals, dtype=torch.float32)
    A = torch.sparse_coo_tensor(coords_t, values_t, size=(N, N)).coalesce()

    # symmetrize + binarize
    AT = torch.sparse_coo_tensor(torch.stack([coords_t[1], coords_t[0]]), values_t, size=(N, N)).coalesce()
    A = (A + AT).coalesce()
    v = torch.where(A.values() > 0, torch.ones_like(A.values()), A.values())
    A = torch.sparse_coo_tensor(A.indices(), v, A.size()).coalesce()

    # self-loop
    d = torch.arange(N, dtype=torch.long)
    I = torch.sparse_coo_tensor(torch.stack([d, d]), torch.ones(N), size=(N, N)).coalesce()
    return (A + I).coalesce()


def _meds_long_to_wide(meds: pd.DataFrame, top_n_drugs: int = 200) -> pd.DataFrame:
    """Convert long medication table (SEQN,RXDDRUG,...) to wide binary features for top drugs."""
    if "RXDDRUG" not in meds.columns:
        return pd.DataFrame({"SEQN": meds["SEQN"].unique()})

    top = meds["RXDDRUG"].dropna().value_counts().head(top_n_drugs).index
    m = meds[meds["RXDDRUG"].isin(top)][["SEQN", "RXDDRUG"]].copy()
    m["val"] = 1.0
    wide = m.pivot_table(index="SEQN", columns="RXDDRUG", values="val", aggfunc="max").fillna(0.0)
    wide.reset_index(inplace=True)
    wide.columns = ["SEQN"] + [f"DRUG_{c}" for c in wide.columns[1:]]
    return wide


def load_nhanes_multirel_patient_graph(
    data_dir: str,
    label_col: str = "DIQ010",
    relations=("labs", "diet", "exam", "meds", "ques"),
    k_neighbors: int = 10,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
    knn_metric: str = "cosine",
):
    """
    Multi-relation patient graph for diabetes prediction (NHANES):
      - Nodes: patients with label DIQ010 in {1,2}
      - X: patient features from demographic numeric columns (stable across models)
      - adjs: list of modality-specific patient-patient graphs (kNN over that modality)
      - agg_adj: aggregated adjacency for single-graph baselines (GCN/GAT)

    Return:
      X, y, train_idx, val_idx, test_idx, adjs, agg_adj, rel_names
    """

    demo = read_csv_safe(os.path.join(data_dir, "demographic.csv"))
    diet = read_csv_safe(os.path.join(data_dir, "diet.csv"))
    exam = read_csv_safe(os.path.join(data_dir, "examination.csv"))
    labs = read_csv_safe(os.path.join(data_dir, "labs.csv"))
    meds = read_csv_safe(os.path.join(data_dir, "medications.csv"))
    ques = read_csv_safe(os.path.join(data_dir, "questionnaire.csv"))

    if label_col not in ques.columns:
        raise ValueError(f"Label column '{label_col}' not found in questionnaire.csv")

    ydf = ques[["SEQN", label_col]].drop_duplicates("SEQN", keep="last")
    ydf = ydf[ydf[label_col].isin([1, 2])].copy()
    ydf["label"] = (ydf[label_col] == 1).astype(int)

    seqn = ydf["SEQN"].to_numpy()
    y_np = ydf["label"].to_numpy(np.int64)
    N = len(seqn)

    # X for all models: demographic numeric columns (stable)
    demo_u = demo.drop_duplicates("SEQN", keep="last").set_index("SEQN").reindex(seqn)
    demo_num = demo_u[[c for c in demo_u.columns if c != "SEQN" and pd.api.types.is_numeric_dtype(demo_u[c])]]
    if demo_num.shape[1] == 0:
        raise RuntimeError("No numeric columns found in demographic.csv for patient features")
    X_np = _zscore_np(demo_num)

    # Build per-modality matrices for kNN graph construction
    mats = {}
    rel_names = []

    if "labs" in relations:
        labs_u = labs.drop_duplicates("SEQN", keep="last").set_index("SEQN").reindex(seqn)
        labs_num = labs_u[[c for c in labs_u.columns if c != "SEQN" and pd.api.types.is_numeric_dtype(labs_u[c])]]
        if labs_num.shape[1] > 0:
            mats["labs"] = _zscore_np(labs_num)
            rel_names.append("labs")

    if "diet" in relations:
        diet_u = diet.drop_duplicates("SEQN", keep="last").set_index("SEQN").reindex(seqn)
        diet_num = diet_u[[c for c in diet_u.columns if c != "SEQN" and pd.api.types.is_numeric_dtype(diet_u[c])]]
        if diet_num.shape[1] > 0:
            mats["diet"] = _zscore_np(diet_num)
            rel_names.append("diet")

    if "exam" in relations:
        exam_u = exam.drop_duplicates("SEQN", keep="last").set_index("SEQN").reindex(seqn)
        exam_num = exam_u[[c for c in exam_u.columns if c != "SEQN" and pd.api.types.is_numeric_dtype(exam_u[c])]]
        if exam_num.shape[1] > 0:
            mats["exam"] = _zscore_np(exam_num)
            rel_names.append("exam")

    if "ques" in relations:
        ques_u = ques.drop_duplicates("SEQN", keep="last").set_index("SEQN").reindex(seqn)
        if label_col in ques_u.columns:
            ques_u = ques_u.drop(columns=[label_col])
        ques_num = ques_u[[c for c in ques_u.columns if c != "SEQN" and pd.api.types.is_numeric_dtype(ques_u[c])]]
        if ques_num.shape[1] > 0:
            mats["ques"] = _zscore_np(ques_num)
            rel_names.append("ques")

    if "meds" in relations:
        meds_wide = _meds_long_to_wide(meds, top_n_drugs=200)
        meds_u = meds_wide.drop_duplicates("SEQN", keep="last").set_index("SEQN").reindex(seqn).fillna(0.0)
        if meds_u.shape[1] > 0:
            mats["meds"] = meds_u.to_numpy(np.float32)
            rel_names.append("meds")

    if len(rel_names) == 0:
        raise RuntimeError("No relations could be built (no numeric columns found in modality tables).")

    # Split indices (stratified)
    indices = np.arange(N)
    idx_train_val, idx_test = train_test_split(indices, test_size=test_size, random_state=random_state, stratify=y_np)
    val_relative = val_size / (1 - test_size)
    idx_train, idx_val = train_test_split(idx_train_val, test_size=val_relative, random_state=random_state, stratify=y_np[idx_train_val])

    # Build relation-specific adjs
    adjs = []
    for r in rel_names:
        A = _build_knn_adj(mats[r], k_neighbors, metric=knn_metric)
        adjs.append(_sym_norm_sparse(A))

    # Aggregate for single-graph baselines
    A_sum = adjs[0]
    for A in adjs[1:]:
        A_sum = (A_sum + A).coalesce()
    v = torch.where(A_sum.values() > 0, torch.ones_like(A_sum.values()), A_sum.values())
    agg_adj = torch.sparse_coo_tensor(A_sum.indices(), v, A_sum.size()).coalesce()
    agg_adj = _sym_norm_sparse(agg_adj)

    X = torch.from_numpy(X_np)
    y = torch.from_numpy(y_np)
    train_idx = torch.from_numpy(idx_train)
    val_idx = torch.from_numpy(idx_val)
    test_idx = torch.from_numpy(idx_test)

    return X, y, train_idx, val_idx, test_idx, adjs, agg_adj, rel_names
