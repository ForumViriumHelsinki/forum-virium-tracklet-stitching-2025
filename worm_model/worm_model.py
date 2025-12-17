import torch
import torch.nn as nn
import torch.nn.functional as F


class TrackletEncoder(nn.Module):
    """
    Encodes a fixed-length tracklet sequence (T=5) + static metadata (e.g., start/end/duration)
    into an L2-normalized embedding vector.

    Inputs:
      loc_seq: (B, T=5, loc_dim)   e.g., loc_dim=1 for 1D, loc_dim=2 for (x,y)
      meta:    (B, meta_dim)       e.g., meta_dim=3 for [t_start, t_end, duration]

    Output:
      z: (B, emb_dim) L2-normalized embedding
    """

    def __init__(
        self,
        loc_dim: int = 1,
        meta_dim: int = 3,
        lstm_hidden: int = 64,
        emb_dim: int = 64,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.0,
        pool: str = "mean",  # "mean" or "last"
    ):
        super().__init__()
        self.pool = pool

        self.lstm = nn.LSTM(
            input_size=loc_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)

        # Fuse pooled LSTM output + metadata -> embedding
        self.fuse = nn.Sequential(
            nn.Linear(lstm_out_dim + meta_dim, 128),
            nn.ReLU(),
            nn.Linear(128, emb_dim),
        )

    def forward(self, loc_seq: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        if loc_seq.ndim != 3:
            raise ValueError(
                f"loc_seq must be (B,T,F). Got shape {tuple(loc_seq.shape)}"
            )
        if meta.ndim != 2:
            raise ValueError(f"meta must be (B,M). Got shape {tuple(meta.shape)}")
        if loc_seq.shape[0] != meta.shape[0]:
            raise ValueError("Batch size mismatch between loc_seq and meta")

        out, (h_n, c_n) = self.lstm(loc_seq)  # out: (B,T,H*)

        if self.pool == "mean":
            pooled = out.mean(dim=1)  # (B,H*)
        elif self.pool == "last":
            # last timestep hidden from out (since T fixed=5)
            pooled = out[:, -1, :]  # (B,H*)
        else:
            raise ValueError("pool must be 'mean' or 'last'")

        fused = torch.cat([pooled, meta], dim=-1)
        z = self.fuse(fused)
        z = F.normalize(z, dim=-1)  # common for metric-style embeddings
        return z


class PairClassifierWithLink(nn.Module):
    """
    Stitch-aware classifier: uses (zA, zB) plus explicit link features computed from endpoints.

    link_feats suggested:
      - log1p_dt
      - dp_x, dp_y  (B_start - A_end) in the SAME normalized coordinate frame
      - implied_speed = ||dp|| / (dt+eps)
      - optional: v_x = dp_x/(dt+eps), v_y = dp_y/(dt+eps)
    """

    def __init__(
        self,
        emb_dim: int = 64,
        link_dim: int = 6,
        hidden: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        in_dim = emb_dim * 4 + link_dim  # [zA, zB, |zA-zB|, zA*zB] + link
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(
        self, z1: torch.Tensor, z2: torch.Tensor, link: torch.Tensor
    ) -> torch.Tensor:
        feats = torch.cat([z1, z2, (z1 - z2).abs(), z1 * z2, link], dim=-1)
        return self.mlp(feats).squeeze(-1)


class SiameseSameTrackModel(nn.Module):
    """
    Convenience wrapper: encodes both tracklets using the same encoder, then classifies.
    """

    def __init__(self, encoder: TrackletEncoder, classifier: PairClassifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(
        self,
        loc_a: torch.Tensor,
        meta_a: torch.Tensor,
        loc_b: torch.Tensor,
        meta_b: torch.Tensor,
    ) -> torch.Tensor:
        z_a = self.encoder(loc_a, meta_a)
        z_b = self.encoder(loc_b, meta_b)
        logits = self.classifier(z_a, z_b)
        return logits
