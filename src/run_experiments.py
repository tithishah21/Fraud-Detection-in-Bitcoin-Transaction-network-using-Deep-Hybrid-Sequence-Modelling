"""
run_experiments.py -- produces every number and figure the paper needs.

Put this file in  bitcoin_fraud/src/  (next to data_utils.py and model.py) and run from there:

    python run_experiments.py --data_dir ../data/elliptic_bitcoin_dataset

Protocol (matches Section 5 of the paper)
    train steps 1-24 | validation steps 25-34 | test steps 35-49
    * features standardised with TRAIN-step statistics only
    * best epoch is chosen by validation PR-AUC, threshold by validation F1
    * the test steps are used once, after selection
    * deep models: 5 seeds; LR / SVM: 1 run; random forest: 5 seeds

Models:  lr, svm, rf_lf, rf_af, mlp, gcn, temporal, hybrid
    hybrid   = your HybridGCNLSTM (GCN per snapshot -> mean pool -> LSTM -> fuse)
    gcn      = same, without the LSTM context          (ablation: removes temporal part)
    temporal = MLP encoder instead of GCN + LSTM context (ablation: removes graph part)
    mlp      = MLP encoder, no context                  (removes both)

Outputs in --out_dir (default ../results):
    <model>.json      raw per-seed results (saved after every run, so a Colab disconnect loses little)
    summary.md        table to read
    table4.tex        rows for Table 4 of the paper (main comparison)
    table5.tex        rows for Table 5 of the paper (ablation)
    per_step.csv      per-test-step F1 (for Figure 3)
    f1_per_step.png   Figure 3
    dataset_counts.json   numbers for Table 2 (labelled / illicit per split)
"""
import argparse
import copy
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.svm import SVC

from data_utils import build_snapshots, get_feature_cols, load_raw
from model import GCNEncoder

MODEL_INFO = {
    "lr": ("Logistic regression", "AF"),
    "svm": ("SVM (RBF)", "AF"),
    "rf_lf": ("Random forest", "LF"),
    "rf_af": ("Random forest", "AF"),
    "mlp": ("MLP", "AF"),
    "gcn": ("GCN only", "AF"),
    "temporal": ("Temporal only (no graph)", "AF"),
    "hybrid": ("Snapshot GCN--LSTM (proposed)", "AF"),
}
DEEP = ("mlp", "gcn", "temporal", "hybrid")
SK_MULTI_SEED = ("rf_lf", "rf_af")
KEYS = ["precision", "recall", "f1", "roc_auc", "pr_auc", "accuracy"]


# ----------------------------------------------------------------------------------------
# metrics (numpy / sklearn only)
# ----------------------------------------------------------------------------------------
def best_threshold(y, s):
    """Score threshold that maximises illicit F1 (used on VALIDATION data only)."""
    pr, rc, th = precision_recall_curve(y, s)
    f1 = 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1] + 1e-12)
    return float(th[int(np.argmax(f1))])


def evaluate_scores(y, s, thr):
    pred = (s >= thr).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", pos_label=1, zero_division=0)
    return {
        "precision": float(p), "recall": float(r), "f1": float(f),
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
        "accuracy": float((pred == y).mean()),
        "threshold": float(thr),
        "confusion": confusion_matrix(y, pred, labels=[0, 1]).tolist(),  # [[TN,FP],[FN,TP]]
    }


def per_step(y, s, t, thr):
    out = {}
    for step in np.unique(t):
        m = t == step
        pred = (s[m] >= thr).astype(int)
        _, _, f, _ = precision_recall_fscore_support(y[m], pred, average="binary", pos_label=1, zero_division=0)
        out[int(step)] = {"f1": float(f), "n_illicit": int(y[m].sum()), "n": int(m.sum())}
    return out


def bootstrap_f1(y, s, thr, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    pred = (s >= thr).astype(int)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        vals.append(precision_recall_fscore_support(y[idx], pred[idx], average="binary",
                                                    pos_label=1, zero_division=0)[2])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def finish_record(yv, sv, yt, st, tt, seed, best_epoch):
    thr = best_threshold(yv, sv)
    rec = evaluate_scores(yt, st, thr)
    rec["seed"] = seed
    rec["best_epoch"] = best_epoch
    rec["val_pr_auc"] = float(average_precision_score(yv, sv))
    rec["per_step"] = per_step(yt, st, tt, thr)
    if seed == 0:
        rec["ci_f1"] = bootstrap_f1(yt, st, thr)
    return rec


# ----------------------------------------------------------------------------------------
# classical baselines
# ----------------------------------------------------------------------------------------
def tabular(snaps, split, cols=None):
    X, y, t = [], [], []
    for d in snaps:
        if d.split != split:
            continue
        m = (d.y != -1).cpu().numpy()
        if m.sum() == 0:
            continue
        x = d.x.cpu().numpy()[m]
        X.append(x if cols is None else x[:, cols])
        y.append(d.y.cpu().numpy()[m])
        t.append(np.full(int(m.sum()), d.time_step))
    return np.concatenate(X), np.concatenate(y), np.concatenate(t)


def run_sklearn(name, seed, snaps, args):
    cols = list(range(93)) if name == "rf_lf" else None        # first 93 = local features
    Xtr, ytr, _ = tabular(snaps, "train", cols)
    Xva, yva, _ = tabular(snaps, "val", cols)
    Xte, yte, tte = tabular(snaps, "test", cols)
    if name == "lr":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    elif name == "svm":
        if len(Xtr) > args.svm_max_train:
            idx = np.random.default_rng(seed).choice(len(Xtr), args.svm_max_train, replace=False)
            Xtr, ytr = Xtr[idx], ytr[idx]
        clf = SVC(kernel="rbf", class_weight="balanced")
    else:
        clf = RandomForestClassifier(n_estimators=args.rf_trees, class_weight="balanced_subsample",
                                     n_jobs=-1, random_state=seed)
    clf.fit(Xtr, ytr)
    if name in ("lr", "svm"):
        sv, st = clf.decision_function(Xva), clf.decision_function(Xte)
    else:
        sv, st = clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1]
    return finish_record(yva, sv, yte, st, tte, seed, None)


# ----------------------------------------------------------------------------------------
# deep models
# ----------------------------------------------------------------------------------------
class SnapNet(nn.Module):
    """encoder in {gcn, mlp}; ctx=True adds mean-pool -> LSTM temporal context.
    SnapNet('gcn', ctx=True) is the same model as HybridGCNLSTM in model.py."""

    def __init__(self, in_dim, hid, lstm_hid, enc="gcn", ctx=True, dropout=0.3):
        super().__init__()
        self.enc, self.ctx = enc, ctx
        if enc == "gcn":
            self.gcn = GCNEncoder(in_dim, hid, dropout=dropout)
        else:
            self.mlp = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Dropout(dropout),
                                     nn.Linear(hid, hid), nn.ReLU())
        if ctx:
            self.lstm = nn.LSTM(input_size=hid, hidden_size=lstm_hid, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hid + (lstm_hid if ctx else 0), hid), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hid, 2))

    def forward(self, snaps):
        hs = [self.gcn(d.x, d.edge_index) if self.enc == "gcn" else self.mlp(d.x) for d in snaps]
        if self.ctx:
            seq = torch.stack([h.mean(dim=0) for h in hs]).unsqueeze(0)   # [1, T, hid]
            r, _ = self.lstm(seq)
            r = r.squeeze(0)                                              # [T, lstm_hid]
        out = []
        for t, h in enumerate(hs):
            z = torch.cat([h, r[t].unsqueeze(0).expand(h.size(0), -1)], dim=1) if self.ctx else h
            out.append(self.head(z))
        return out


def collect(logits, snaps, split):
    ys, ss, ts = [], [], []
    for d, lg in zip(snaps, logits):
        if d.split != split:
            continue
        m = d.y != -1
        n = int(m.sum())
        if n == 0:
            continue
        ss.append(torch.softmax(lg[m], dim=1)[:, 1].detach().cpu().numpy())
        ys.append(d.y[m].cpu().numpy())
        ts.append(np.full(n, d.time_step))
    return np.concatenate(ys), np.concatenate(ss), np.concatenate(ts)


def run_deep(name, seed, snaps, in_dim, args, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    enc = "gcn" if name in ("gcn", "hybrid") else "mlp"
    ctx = name in ("hybrid", "temporal")
    model = SnapNet(in_dim, args.hidden, args.lstm_hidden, enc, ctx, args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    cw = torch.tensor([1.0, args.illicit_weight], device=device)
    best_score, best_state, best_ep = -1.0, None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(snaps)
        losses = []
        for d, lg in zip(snaps, logits):
            if d.split != "train":
                continue
            m = d.y != -1
            if int(m.sum()) == 0:
                continue
            losses.append(F.cross_entropy(lg[m], d.y[m], weight=cw))
        loss = torch.stack(losses).mean()
        loss.backward()
        opt.step()
        if ep % args.eval_every == 0 or ep == args.epochs:
            model.eval()
            with torch.no_grad():
                lgs = model(snaps)
            yv, sv, _ = collect(lgs, snaps, "val")
            score = float(average_precision_score(yv, sv))      # selection uses VALIDATION only
            if score > best_score:
                best_score, best_state, best_ep = score, copy.deepcopy(model.state_dict()), ep
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        lgs = model(snaps)
    yv, sv, _ = collect(lgs, snaps, "val")
    yt, st, tt = collect(lgs, snaps, "test")
    return finish_record(yv, sv, yt, st, tt, seed, best_ep)


# ----------------------------------------------------------------------------------------
# aggregation -> paper-ready outputs
# ----------------------------------------------------------------------------------------
def fmt(vals):
    if len(vals) > 1:
        return "%.3f $\\pm$ %.3f" % (np.mean(vals), np.std(vals, ddof=1))
    return "%.3f" % vals[0]


def fmt_md(vals):
    if len(vals) > 1:
        return "%.3f ± %.3f" % (np.mean(vals), np.std(vals, ddof=1))
    return "%.3f" % vals[0]


def aggregate(out_dir):
    runs = {}
    for name in MODEL_INFO:
        p = os.path.join(out_dir, name + ".json")
        if os.path.exists(p):
            runs[name] = json.load(open(p))["runs"]
    if not runs:
        print("No result files found in", out_dir)
        return
    md = ["| Model | Feat. | Precision | Recall | F1 | ROC-AUC | PR-AUC | #runs | best epoch | val PR-AUC |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    t4, steps = [], {}
    for name, rs in runs.items():
        label, feat = MODEL_INFO[name]
        cells = {k: [r[k] for r in rs] for k in KEYS}
        eps = [r["best_epoch"] for r in rs if r.get("best_epoch")]
        val = [r["val_pr_auc"] for r in rs]
        md.append("| %s | %s | %s | %s | %s | %s | %s | %d | %s | %s |" % (
            label, feat, fmt_md(cells["precision"]), fmt_md(cells["recall"]), fmt_md(cells["f1"]),
            fmt_md(cells["roc_auc"]), fmt_md(cells["pr_auc"]), len(rs),
            (", ".join(map(str, eps)) if eps else "-"), fmt_md(val)))
        t4.append("%s & %s & %s & %s & %s & %s & %s \\\\" % (
            label, feat, fmt(cells["precision"]), fmt(cells["recall"]), fmt(cells["f1"]),
            fmt(cells["roc_auc"]), fmt(cells["pr_auc"])))
        for r in rs:
            for k, v in r["per_step"].items():
                steps.setdefault(name, {}).setdefault(int(k), []).append((v["f1"], v["n_illicit"]))
    ci_lines = []
    for name, rs in runs.items():
        for r in rs:
            if r["seed"] == 0 and "ci_f1" in r:
                ci_lines.append("- %s: test F1 = %.3f, 95%% bootstrap CI [%.3f, %.3f] (seed 0)" % (
                    MODEL_INFO[name][0] + " (" + MODEL_INFO[name][1] + ")", r["f1"], r["ci_f1"][0], r["ci_f1"][1]))
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n\nBootstrap confidence intervals\n" + "\n".join(ci_lines) + "\n")
    with open(os.path.join(out_dir, "table4.tex"), "w") as f:
        f.write("\n".join(t4) + "\n")
    t5 = []
    for name, graph, ctx in (("hybrid", "yes", "yes"), ("gcn", "yes", "no"),
                             ("temporal", "no", "yes"), ("mlp", "no", "no")):
        if name in runs:
            rs = runs[name]
            lab = MODEL_INFO[name][0]
            t5.append("%s & %s & %s & %s & %s \\\\" % (lab, graph, ctx, fmt([r["f1"] for r in rs]),
                                                     fmt([r["pr_auc"] for r in rs])))
    with open(os.path.join(out_dir, "table5.tex"), "w") as f:
        f.write("\n".join(t5) + "\n")
    with open(os.path.join(out_dir, "per_step.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "step", "mean_f1", "n_illicit"])
        for name, d in steps.items():
            for step in sorted(d):
                w.writerow([name, step, "%.4f" % np.mean([x[0] for x in d[step]]), d[step][0][1]])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, d in steps.items():
            xs = sorted(d)
            ax.plot(xs, [np.mean([x[0] for x in d[s]]) for s in xs], marker="o", ms=3,
                    label=MODEL_INFO[name][0] + " (" + MODEL_INFO[name][1] + ")")
        ax.axvline(43, color="red", ls="--", lw=1)
        ax.set_xlabel("Test time step")
        ax.set_ylabel("Illicit-class F1")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "f1_per_step.png"), dpi=200)
    except Exception as e:  # plotting must never kill the run
        print("plot skipped:", e)
    print("\n".join(md))
    print("\n".join(ci_lines))
    print("\nWrote summary.md, table4.tex, table5.tex, per_step.csv, f1_per_step.png in", out_dir)


# ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="../data/elliptic_bitcoin_dataset")
    ap.add_argument("--out_dir", default="../results")
    ap.add_argument("--models", default="lr,svm,rf_lf,rf_af,mlp,gcn,temporal,hybrid")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--train_end", type=int, default=24)
    ap.add_argument("--val_end", type=int, default=34)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lstm_hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--illicit_weight", type=float, default=4.0)
    ap.add_argument("--rf_trees", type=int, default=300)
    ap.add_argument("--svm_max_train", type=int, default=20000)
    ap.add_argument("--allow_small", action="store_true", help="smoke test on synthetic data")
    ap.add_argument("--aggregate_only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.aggregate_only:
        aggregate(args.out_dir)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    df, edges = load_raw(args.data_dir)
    feat_cols = get_feature_cols(df)
    n_steps = df["time_step"].nunique()
    print("nodes=%d  edges=%d  features=%d  time steps=%d" % (len(df), len(edges), len(feat_cols), n_steps))
    if len(df) < 100000 and not args.allow_small:
        raise SystemExit("STOP: this is not the real Elliptic data (expected 203,769 nodes). "
                         "Check --data_dir. Use --allow_small only for the synthetic smoke test.")

    # standardise with TRAIN-step statistics only
    tr = df["time_step"] <= args.train_end
    mu = df.loc[tr, feat_cols].mean()
    sd = df.loc[tr, feat_cols].std().replace(0, 1.0) + 1e-6
    df[feat_cols] = (df[feat_cols] - mu) / sd

    snaps = build_snapshots(df, edges, feat_cols)
    for d in snaps:
        d.split = "train" if d.time_step <= args.train_end else ("val" if d.time_step <= args.val_end else "test")
        d.x, d.edge_index, d.y = d.x.to(device), d.edge_index.to(device), d.y.to(device)

    counts = {}
    for sp in ("train", "val", "test"):
        lab = sum(int((d.y != -1).sum()) for d in snaps if d.split == sp)
        ill = sum(int((d.y == 1).sum()) for d in snaps if d.split == sp)
        counts[sp] = {"labelled": lab, "illicit": ill}
        print("%-5s labelled=%d illicit=%d" % (sp, lab, ill))
    json.dump(counts, open(os.path.join(args.out_dir, "dataset_counts.json"), "w"), indent=2)

    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        seeds = range(args.seeds) if (name in DEEP or name in SK_MULTI_SEED) else [0]
        recs = []
        for sd_ in seeds:
            t0 = time.time()
            if name in DEEP:
                rec = run_deep(name, sd_, snaps, len(feat_cols), args, device)
            else:
                rec = run_sklearn(name, sd_, snaps, args)
            rec["seconds"] = round(time.time() - t0, 1)
            recs.append(rec)
            print("[%s seed %d] test F1=%.3f P=%.3f R=%.3f ROC=%.3f PR=%.3f | val PR-AUC=%.3f best_epoch=%s (%.0fs)" % (
                name, sd_, rec["f1"], rec["precision"], rec["recall"], rec["roc_auc"], rec["pr_auc"],
                rec["val_pr_auc"], rec["best_epoch"], rec["seconds"]))
            json.dump({"model": name, "args": vars(args), "runs": recs},
                      open(os.path.join(args.out_dir, name + ".json"), "w"))
    aggregate(args.out_dir)


if __name__ == "__main__":
    main()