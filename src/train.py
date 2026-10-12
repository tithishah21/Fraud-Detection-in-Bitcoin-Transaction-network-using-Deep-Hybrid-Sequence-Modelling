import argparse
import torch
import torch.nn.functional as F

from data_utils import load_raw, build_snapshots, get_feature_cols
from model import HybridGCNLSTM
from evaluate import evaluate, print_metrics


def move_to_device(snapshots, device):
    for d in snapshots:
        d.x = d.x.to(device)
        d.edge_index = d.edge_index.to(device)
        d.y = d.y.to(device)
    return snapshots


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Loading raw CSVs ...")
    df, edges = load_raw(args.data_dir)
    feature_cols = get_feature_cols(df)
    print(f"Loaded {len(df)} nodes, {len(feature_cols)} features, {len(edges)} edges.")

    print("Building per-time-step graph snapshots ...")
    snapshots = build_snapshots(df, edges, feature_cols,
                                 train_max_step=args.train_max_step)
    snapshots = move_to_device(snapshots, device)

    n_train = sum(s.is_train for s in snapshots)
    print(f"{len(snapshots)} snapshots total -> {n_train} train / {len(snapshots) - n_train} test")

    model = HybridGCNLSTM(
        in_dim=len(feature_cols),
        hidden_dim=args.hidden_dim,
        lstm_hidden=args.lstm_hidden,
        num_classes=2,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # class weights to counter illicit << licit imbalance (typical ratio ~9:1 in Elliptic)
    class_weights = torch.tensor([1.0, args.illicit_weight], device=device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits_per_snapshot = model(snapshots)

        losses = []
        for data, logits in zip(snapshots, logits_per_snapshot):
            if not data.is_train:
                continue
            mask = data.y != -1
            if mask.sum() == 0:
                continue
            loss = F.cross_entropy(logits[mask], data.y[mask], weight=class_weights)
            losses.append(loss)

        total_loss = torch.stack(losses).mean()
        total_loss.backward()
        optimizer.step()

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            train_metrics = evaluate(model, snapshots, which="train")
            test_metrics = evaluate(model, snapshots, which="test")
            print(f"\nEpoch {epoch:03d}  loss={total_loss.item():.4f}")
            print_metrics("train", train_metrics)
            print_metrics("test ", test_metrics)

    torch.save(model.state_dict(), args.out)
    print(f"\nSaved model weights to {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="../data",
                    help="folder containing the three elliptic_txs_*.csv files")
    p.add_argument("--train_max_step", type=int, default=34,
                    help="time steps 1..N used for training, rest for test")
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--lstm_hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--illicit_weight", type=float, default=4.0,
                    help="class weight for the minority (illicit) class")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--out", type=str, default="../checkpoints/hybrid_gcn_lstm.pt")
    args = p.parse_args()
    main(args)
