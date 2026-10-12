import torch
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix


@torch.no_grad()
def evaluate(model, snapshots, which="test"):
    model.eval()
    logits_per_snapshot = model(snapshots)

    all_probs, all_preds, all_true = [], [], []
    for data, logits in zip(snapshots, logits_per_snapshot):
        is_split = data.is_train if which == "train" else (not data.is_train)
        if not is_split:
            continue
        mask = data.y != -1
        if mask.sum() == 0:
            continue
        probs = torch.softmax(logits[mask], dim=1)[:, 1]  # P(illicit)
        preds = probs.argmax(dim=0)  # placeholder, replaced below
        preds = (probs >= 0.5).long()
        all_probs.append(probs)
        all_preds.append(preds)
        all_true.append(data.y[mask])

    if not all_true:
        return {}

    y_true = torch.cat(all_true).cpu().numpy()
    y_pred = torch.cat(all_preds).cpu().numpy()
    y_prob = torch.cat(all_probs).cpu().numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")  # only one class present in this split

    acc = (y_true == y_pred).mean()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "accuracy": acc,
        "precision_illicit": precision,
        "recall_illicit": recall,
        "f1_illicit": f1,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
        "n_samples": len(y_true),
    }


def print_metrics(tag, metrics):
    if not metrics:
        print(f"[{tag}] no labeled samples in this split")
        return
    print(f"[{tag}] "
          f"acc={metrics['accuracy']:.4f}  "
          f"precision(illicit)={metrics['precision_illicit']:.4f}  "
          f"recall(illicit)={metrics['recall_illicit']:.4f}  "
          f"f1(illicit)={metrics['f1_illicit']:.4f}  "
          f"roc_auc={metrics['roc_auc']:.4f}  "
          f"(n={metrics['n_samples']})")
    print(f"        confusion_matrix [[TN,FP],[FN,TP]] = {metrics['confusion_matrix']}")
