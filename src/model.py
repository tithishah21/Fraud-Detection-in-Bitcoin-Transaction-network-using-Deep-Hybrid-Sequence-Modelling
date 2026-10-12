"""
Hybrid Deep Sequence Model: GCN (spatial/structural) + LSTM (temporal).

Design rationale (matches the "dynamic GCN + LSTM" approach used in
Wan & Li 2024 and similar cited work):

  Elliptic transaction-nodes exist at exactly ONE time step each (a
  transaction is not observed again later), so there is no literal
  per-wallet history to feed an LSTM. Instead we treat the sequence of
  49 GRAPH SNAPSHOTS itself as the temporal axis:

    1. A GCN encodes each snapshot G_t independently  -> node embeddings.
    2. Snapshot embeddings are mean-pooled into one "network state"
       vector per time step  -> a sequence of 49 vectors.
    3. An LSTM consumes that sequence and produces a temporal-context
       vector h_t for every time step (summarising how the network has
       evolved up to and including t).
    4. Each node's final representation = [its own GCN embedding] concat
       [temporal context h_t of its snapshot]  -> classifier head.

  This gives every node's classification both a structural signal (its
  local graph neighbourhood) and a temporal signal (how the overall
  transaction network was behaving at that point in time) -- i.e. a
  genuine spatio-temporal fusion, not two separate models bolted
  together.

  NOTE for the report: this is a deliberate, documented modelling choice
  necessitated by the Elliptic dataset's structure. If you later switch
  to Elliptic++ (which has wallet-level, multi-transaction actor data),
  the same LSTM module can instead consume true per-wallet sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        return h  # [num_nodes, hidden_dim]


class HybridGCNLSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, lstm_hidden=64, num_classes=2, dropout=0.3):
        super().__init__()
        self.gcn = GCNEncoder(in_dim, hidden_dim, dropout=dropout)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=lstm_hidden,
                             num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + lstm_hidden, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, snapshots):
        """
        snapshots: ordered list of torch_geometric.data.Data, one per time step.
        Returns: list of per-node logits, same order as snapshots.
        """
        node_embeddings = []
        snapshot_summaries = []

        for data in snapshots:
            h = self.gcn(data.x, data.edge_index)          # [N_t, hidden_dim]
            node_embeddings.append(h)
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
            summary = global_mean_pool(h, batch)            # [1, hidden_dim]
            snapshot_summaries.append(summary)

        # sequence of network-state vectors: [1, T, hidden_dim]
        seq = torch.cat(snapshot_summaries, dim=0).unsqueeze(0)
        temporal_context, _ = self.lstm(seq)                # [1, T, lstm_hidden]
        temporal_context = temporal_context.squeeze(0)       # [T, lstm_hidden]

        logits_per_snapshot = []
        for t, h in enumerate(node_embeddings):
            ctx = temporal_context[t].unsqueeze(0).expand(h.size(0), -1)  # broadcast
            fused = torch.cat([h, ctx], dim=1)
            logits = self.classifier(fused)
            logits_per_snapshot.append(logits)

        return logits_per_snapshot
