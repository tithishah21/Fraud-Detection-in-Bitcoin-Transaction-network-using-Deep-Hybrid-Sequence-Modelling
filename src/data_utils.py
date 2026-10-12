"""
Data loading + preprocessing for the Elliptic Bitcoin Dataset.

Expected raw files (as distributed on Kaggle: "Elliptic Data Set"):
    elliptic_txs_features.csv   -> 203,769 rows x 167 cols, NO header.
                                    col 0        = txId
                                    col 1        = time step (1-49)
                                    col 2..167   = 165 node features
                                                   (93 local + 72 aggregated)
    elliptic_txs_classes.csv    -> header: txId,class
                                    class in {'1' (illicit), '2' (licit), 'unknown'}
    elliptic_txs_edgelist.csv   -> header: txId1,txId2

If your downloaded CSVs have a header row on features.csv, set
FEATURES_HAS_HEADER = True below.
"""

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

FEATURES_HAS_HEADER = False


def load_raw(data_dir):
    feat_kwargs = {} if FEATURES_HAS_HEADER else {"header": None}
    features = pd.read_csv(f"{data_dir}/elliptic_txs_features.csv", **feat_kwargs)
    classes = pd.read_csv(f"{data_dir}/elliptic_txs_classes.csv")
    edges = pd.read_csv(f"{data_dir}/elliptic_txs_edgelist.csv")

    # name columns explicitly
    n_feat_cols = features.shape[1] - 2
    features.columns = ["txId", "time_step"] + [f"feat_{i}" for i in range(n_feat_cols)]

    classes.columns = ["txId", "class"]
    # map: illicit -> 1, licit -> 0, unknown -> -1 (excluded from loss)
    class_map = {"1": 1, "2": 0, "unknown": -1, 1: 1, 2: 0}
    classes["label"] = classes["class"].map(class_map)

    edges.columns = ["txId1", "txId2"]

    df = features.merge(classes[["txId", "label"]], on="txId", how="left")
    df["label"] = df["label"].fillna(-1).astype(int)
    return df, edges


def build_snapshots(df, edges, feature_cols, train_max_step=34, test_min_step=35):
    """
    Builds one PyG Data object per time step (1..49).
    Returns: list_of_Data (ordered by time step), split info.
    Time-based split follows the standard Elliptic convention:
        steps 1-34  -> train
        steps 35-49 -> test
    """
    snapshots = []
    time_steps = sorted(df["time_step"].unique())

    for t in time_steps:
        sub = df[df["time_step"] == t].reset_index(drop=True)
        txid_to_idx = {tx: i for i, tx in enumerate(sub["txId"].values)}

        x = torch.tensor(sub[feature_cols].values, dtype=torch.float)
        y = torch.tensor(sub["label"].values, dtype=torch.long)

        # keep only edges where both endpoints exist in this snapshot
        sub_edges = edges[edges["txId1"].isin(txid_to_idx) & edges["txId2"].isin(txid_to_idx)]
        if len(sub_edges) > 0:
            src = sub_edges["txId1"].map(txid_to_idx).values
            dst = sub_edges["txId2"].map(txid_to_idx).values
            edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
            # make undirected (helps message passing for this task)
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        data.time_step = int(t)
        data.is_train = int(t) <= train_max_step
        snapshots.append(data)

    return snapshots


def get_feature_cols(df):
    return [c for c in df.columns if c.startswith("feat_")]
