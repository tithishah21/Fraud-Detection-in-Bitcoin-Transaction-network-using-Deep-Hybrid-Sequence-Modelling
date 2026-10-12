"""
Creates a TINY synthetic dataset with the same file structure/schema as the
real Elliptic Bitcoin Dataset, purely so train.py can be smoke-tested before
you plug in the real ~200k-node CSVs. Do NOT use this for your actual
results -- swap in the real Kaggle files under data/ before training for real.
"""
import numpy as np
import pandas as pd

np.random.seed(0)

N_TIME_STEPS = 8
NODES_PER_STEP = 40
N_FEATURES = 165

rows_feat, rows_class, rows_edge = [], [], []
tx_counter = 0

for t in range(1, N_TIME_STEPS + 1):
    ids_this_step = []
    for _ in range(NODES_PER_STEP):
        tx_counter += 1
        tx_id = tx_counter
        ids_this_step.append(tx_id)
        feats = np.random.randn(N_FEATURES).round(4)
        rows_feat.append([tx_id, t] + feats.tolist())

        r = np.random.rand()
        cls = "1" if r < 0.12 else ("2" if r < 0.9 else "unknown")
        rows_class.append([tx_id, cls])

    # random edges within this time step's nodes
    for _ in range(NODES_PER_STEP):
        a, b = np.random.choice(ids_this_step, size=2, replace=False)
        rows_edge.append([a, b])

features_df = pd.DataFrame(rows_feat)
classes_df = pd.DataFrame(rows_class, columns=["txId", "class"])
edges_df = pd.DataFrame(rows_edge, columns=["txId1", "txId2"])

features_df.to_csv("data/elliptic_txs_features.csv", index=False, header=False)
classes_df.to_csv("data/elliptic_txs_classes.csv", index=False)
edges_df.to_csv("data/elliptic_txs_edgelist.csv", index=False)

print("Synthetic dataset written to data/:")
print(f"  {len(features_df)} nodes across {N_TIME_STEPS} time steps")
print(f"  {len(edges_df)} edges")
print(f"  class distribution:\n{classes_df['class'].value_counts()}")
