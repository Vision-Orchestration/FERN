"""
FERN v2 — Foot gesture recognition from skeleton sequences.

Architecture: SpatialCNN -> BiLSTM -> Attention -> Classifier

Input shape:  (batch, seq_len, num_features)
              where num_features = num_joints * 3
Output shape: (batch, num_classes)

The design mirrors what worked in the SensiFoot paper (Enfield Report):
  - 1D CNN captures spatial relationships between joints within each frame.
  - BiLSTM captures how those relationships change over time.
  - Additive attention learns which frames in the window matter most.
  - The classifier head reads the attended context vector.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Spatial CNN block
# ---------------------------------------------------------------------------

class SpatialCNNBlock(nn.Module):
    """
    Applies 1D convolutions across the feature dimension of each frame.

    Conceptually, each frame is a vector of joint coordinates.  We treat
    each joint as a "channel" and let Conv1d discover patterns like
    "left ankle is far from right heel", etc.

    The block stacks two conv layers with BatchNorm and ReLU so the
    network has enough non-linearity without becoming too deep.
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.3):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=out_features // 2,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=out_features // 2,
            out_channels=out_features,
            kernel_size=3,
            padding=1
        )
        self.bn1  = nn.BatchNorm1d(out_features // 2)
        self.bn2  = nn.BatchNorm1d(out_features)
        self.proj = nn.Linear(in_features * out_features, out_features)
        self.drop = nn.Dropout(dropout)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch * seq_len, in_features)
        # Treat the feature vector as a 1-channel 1D signal.
        b = x.size(0)
        x = x.unsqueeze(1)                       # (B*T, 1, in_features)
        x = F.relu(self.bn1(self.conv1(x)))      # (B*T, out//2, in_features)
        x = F.relu(self.bn2(self.conv2(x)))      # (B*T, out, in_features)
        x = x.view(b, -1)                        # flatten
        x = self.drop(F.relu(self.proj(x)))      # (B*T, out_features)
        return x


# ---------------------------------------------------------------------------
# Additive (Bahdanau-style) Attention
# ---------------------------------------------------------------------------

class AdditiveAttention(nn.Module):
    """
    Additive attention over the time axis of a BiLSTM output.

    Given hidden states h_1 ... h_T, computes a single context vector as a
    weighted sum: context = sum_t( alpha_t * h_t ).

    The weights alpha_t are learned via a small feed-forward network that
    scores each hidden state against a trainable query vector.  This lets
    the model learn "which moments in the gesture window are most important."
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Project each hidden state to a scalar score.
        self.score_fc = nn.Linear(hidden_dim, hidden_dim)
        self.query    = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor):
        # lstm_out: (batch, seq_len, hidden_dim)
        scores = torch.tanh(self.score_fc(lstm_out))   # (B, T, H)
        scores = self.query(scores).squeeze(-1)        # (B, T)
        weights = F.softmax(scores, dim=1)             # (B, T)  — sum to 1
        # Weighted sum over time.
        context = torch.bmm(
            weights.unsqueeze(1),   # (B, 1, T)
            lstm_out                # (B, T, H)
        ).squeeze(1)                # (B, H)
        return context, weights


# ---------------------------------------------------------------------------
# Full FERN v2 model
# ---------------------------------------------------------------------------

class FERNv2(nn.Module):
    """
    Full skeleton-based foot gesture recognition model.

    Parameters
    ----------
    num_joints : int
        Number of skeleton joints used as input (default 10 = lower body).
    num_classes : int
        Number of gesture classes (default 8).
    cnn_out : int
        Output channels of the spatial CNN block.
    lstm_hidden : int
        Hidden size of each LSTM direction (total is 2 * lstm_hidden
        because it is bidirectional).
    lstm_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout probability applied throughout.
    """

    def __init__(
        self,
        num_joints:  int = 10,
        num_classes: int = 7,
        cnn_out:     int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout:     float = 0.4,
    ):
        super().__init__()

        in_features = num_joints * 3   # x, y, z per joint

        # --- Spatial CNN ---
        self.spatial_cnn = SpatialCNNBlock(
            in_features=in_features,
            out_features=cnn_out,
            dropout=dropout,
        )

        # --- BiLSTM ---
        # Bidirectional doubles the output size, so lstm_hidden * 2.
        self.lstm = nn.LSTM(
            input_size=cnn_out,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out_dim = lstm_hidden * 2   # bidirectional

        # --- Attention ---
        self.attention = AdditiveAttention(lstm_out_dim)

        # --- Classifier head ---
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier init for linear layers, orthogonal for LSTM."""
        for name, p in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(p)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(p)
            elif 'bias' in name:
                nn.init.zeros_(p)
            elif isinstance(p, nn.Linear):
                nn.init.xavier_uniform_(p.weight)

    def forward(self, x: torch.Tensor):
        """
        x : (batch, seq_len, num_joints * 3)
        returns logits : (batch, num_classes)
        """
        B, T, F = x.shape

        # --- Spatial CNN: process every frame independently ---
        x_flat = x.view(B * T, F)              # merge batch and time
        x_cnn  = self.spatial_cnn(x_flat)      # (B*T, cnn_out)
        x_seq  = x_cnn.view(B, T, -1)          # restore time axis

        # --- BiLSTM over the sequence ---
        lstm_out, _ = self.lstm(x_seq)         # (B, T, lstm_hidden*2)

        # --- Attention pooling ---
        context, attn_weights = self.attention(lstm_out)   # (B, lstm_hidden*2)

        # --- Classify ---
        logits = self.classifier(context)      # (B, num_classes)

        return logits

    def predict_with_attention(self, x: torch.Tensor):
        """
        Same as forward but also returns attention weights for visualization.
        Useful for debugging: which frames in the window drove the prediction?
        """
        B, T, F = x.shape
        x_flat  = x.view(B * T, F)
        x_cnn   = self.spatial_cnn(x_flat)
        x_seq   = x_cnn.view(B, T, -1)
        lstm_out, _ = self.lstm(x_seq)
        context, attn_weights = self.attention(lstm_out)
        logits = self.classifier(context)
        return logits, attn_weights


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    model = FERNv2(num_joints=10, num_classes=8)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")

    dummy = torch.randn(4, 60, 30)   # batch=4, seq=60 frames, 10 joints * 3
    out   = model(dummy)
    print(f"Output shape: {out.shape}")  # expected: (4, 8)
