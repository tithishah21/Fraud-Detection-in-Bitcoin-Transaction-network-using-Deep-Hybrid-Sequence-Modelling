# Bitcoin Fraud Detection — Hybrid GNN + LSTM (Review 2 Implementation Starter)

This is a **working, tested pipeline** (verified end-to-end on synthetic data of
the same schema) for your Deep Hybrid Sequence Modelling project. Swap in the
real Elliptic dataset and it should run as-is.

## 1. Tech stack

| Component | Choice | Why |
|---|---|---|
| Language | Python 3.10+ | standard for ML |
| Deep learning framework | **PyTorch** | flexibility, most GNN libraries build on it |
| Graph learning | **PyTorch Geometric (PyG)** — `GCNConv`, `global_mean_pool` | industry-standard GNN library, used by nearly all your cited papers |
| Sequence modelling | `torch.nn.LSTM` | as specified in your proposal |
| Data handling | `pandas`, `numpy` | CSV loading, merging features/labels/edges |
| Evaluation | `scikit-learn` (`precision_recall_fscore_support`, `roc_auc_score`, `confusion_matrix`) | matches the metrics your Review 1 doc committed to |
| Environment | **Google Colab (free GPU)** recommended, or local with `venv` | Colab has enough RAM/GPU for the full 200k-node dataset and one-click Kaggle download |
| Dataset | **Elliptic Bitcoin Dataset** (Kaggle: `ellipticco/elliptic-data-set`) | as named in your Review 1 report |

Install (Colab cell or local terminal):
```bash
pip install torch torch_geometric pandas numpy scikit-learn
```

## 2. Important design decision — read this before you code

Your Review 1 report describes the LSTM as modelling *"the transaction
history of specific wallets."* The **raw Elliptic dataset's nodes are
individual transactions**, each appearing at exactly **one** time step —
there is no multi-transaction wallet history to feed an LSTM directly
(that level of detail exists only in the newer, heavier **Elliptic++**
dataset, which adds wallet/actor-level features).

To hit your deadline without abandoning the hybrid design, this
implementation instead treats **the 49 time-step graph snapshots
themselves as the temporal sequence** — a GCN encodes each snapshot,
and an LSTM learns how the overall transaction network evolves
step-to-step. This is the same style of architecture used in the
Wan & Li (2024) and Lin et al. (2026) papers you already cited, so it
is defensible and consistent with your literature review.

**Say this explicitly in your report/PPT** ("Note on Temporal
Modelling") — it reads as an informed design decision, not a shortcut,
and pre-empts the obvious question a reviewer would ask. If you have
time after Review 2, upgrading to Elliptic++ for true per-wallet
sequences is a natural "future work" line.

## 3. File structure

```
bitcoin_fraud/
├── data/                      # put the 3 real Elliptic CSVs here
├── checkpoints/                # trained model weights land here
├── generate_synthetic_data.py  # smoke-test data (already verified working)
└── src/
    ├── data_utils.py    # CSV loading, snapshot/graph construction, splitting
    ├── model.py         # GCNEncoder + HybridGCNLSTM
    ├── train.py         # training loop (run this)
    └── evaluate.py      # precision/recall/F1/ROC-AUC + confusion matrix
```

## 4. Step-by-step: what to actually do today

**Step 1 — Get the real data (10 min)**
Download the Elliptic Data Set from Kaggle and place these 3 files in `data/`:
- `elliptic_txs_features.csv`
- `elliptic_txs_classes.csv`
- `elliptic_txs_edgelist.csv`

In Colab:
```python
import kagglehub
path = kagglehub.dataset_download("ellipticco/elliptic-data-set")
print(path)  # copy the 3 CSVs from here into data/
```
Open `elliptic_txs_features.csv` and check the first row — if it looks
like column names (not numbers), set `FEATURES_HAS_HEADER = True` at
the top of `src/data_utils.py`.

**Step 2 — Sanity-check the pipeline runs (5 min)**
Don't skip this — confirm the code runs before pointing it at 200k real rows:
```bash
python generate_synthetic_data.py      # already verified to work
cd src
python train.py --data_dir ../data --epochs 20 --train_max_step 6
```
You should see loss decreasing and metrics printing every few epochs
(exactly what you saw when this was tested while building it for you).

**Step 3 — Point it at the real dataset (2 min)**
Regenerate/replace the CSVs in `data/` with the real Kaggle files, then:
```bash
python train.py --data_dir ../data --epochs 150 --eval_every 10 \
                 --hidden_dim 64 --lstm_hidden 64 --illicit_weight 4.0
```
On the real ~200k-node dataset this will take a few minutes on Colab's
free GPU (CPU will be much slower — use Colab's GPU runtime).

**Step 4 — Read and record your results (15–20 min)**
`train.py` prints, every `eval_every` epochs:
- Accuracy, Precision/Recall/F1 for the **illicit** class specifically
  (this is the number that matters most — the classes are ~9:1
  imbalanced, so accuracy alone is misleading)
- ROC-AUC
- Confusion matrix

Screenshot/copy these for your report's Results section and for the
demo. If illicit recall is low, try raising `--illicit_weight` (e.g. to
6–8) — this trades some precision for catching more fraud, which is
usually the right trade-off to argue for in your report.

**Step 5 — (If time remains) Baseline comparison**
Your proposal promises "GNN/LSTM standalone vs Hybrid" comparison. Fastest
way to get this for tomorrow: comment out the LSTM branch temporarily (feed
the GCN embedding alone into the classifier, skip step 3 in the LSTM/fusion
code) and re-run — that's your "GNN-only" baseline number for the same
codebase, letting you show the hybrid model wins in a same-conditions
comparison. A from-scratch separate LSTM-only baseline (features without
graph structure) is a good stretch goal but not essential for the
60–75% checkpoint.

## 5. What "60–75% done" should look like tomorrow

✅ Data pipeline loading real Elliptic CSVs into per-time-step graphs
✅ Hybrid GCN+LSTM model training without errors, loss decreasing
✅ Precision/Recall/F1/ROC-AUC computed on a proper time-based train/test split
✅ At least one results table/screenshot ready for the PPT
🔲 Hyperparameter tuning — leave for Review 3
🔲 Full baseline comparison (GNN-only, LSTM-only, classical ML) — partial is fine, full is Review 3
🔲 Real-time / production-style deployment — Review 3 scope only

## 6. Common errors and fixes

| Error | Fix |
|---|---|
| `FileNotFoundError` on the 3 CSVs | Check `--data_dir` path and exact filenames |
| All metrics are 0 for a class | Normal in early epochs / on tiny synthetic data — real data + more epochs fixes it |
| CUDA out of memory (Colab GPU) | Lower `--hidden_dim`/`--lstm_hidden` to 32 |
| Training loss is `nan` | Lower `--lr` to 0.001 |
| `roc_auc` prints `nan` | That split has only one class present (e.g. no illicit txns at that split) — usually fine for train/test as a whole, only an issue if it happens on the full split |
