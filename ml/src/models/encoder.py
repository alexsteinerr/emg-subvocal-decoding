# encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Sequential(
            nn.Linear(channels, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, channels), nn.Sigmoid()
        )

    def forward(self, x):          # [B, C, T]
        b, c, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1)
        return x * w


class Encoder(nn.Module):
    """
    1-D convolutional encoder.

    forward() returns both:
      seq_features  – [B, 512, T']  (sequential, used for CTC)
      pooled        – [B, hidden_dim]  (global embedding, kept for compatibility)
    """

    def __init__(self, input_channels: int, hidden_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input T=6000 → after 3 stride-4 layers → T≈94 frames
        # CTC target phoneme seqs are ~10-20 tokens,
        # so we need T in the 50-200 range to avoid blank collapse.
        # stride=4 per layer: 6000 → 1500 → 375 → 94
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_channels, 128, kernel_size=15, stride=4, padding=7),
                nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
                ChannelAttention(128)
            ),
            nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=11, stride=4, padding=5),
                nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
                ChannelAttention(256)
            ),
            nn.Sequential(
                nn.Conv1d(256, 512, kernel_size=7, stride=4, padding=3),
                nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.2),
                ChannelAttention(512)
            ),
        ])

        self.residuals = nn.ModuleList([
            nn.Conv1d(input_channels, 128, kernel_size=1, stride=4) if input_channels != 128
                else nn.Conv1d(input_channels, 128, kernel_size=1, stride=4),
            nn.Conv1d(128, 256, kernel_size=1, stride=4),
            nn.Conv1d(256, 512, kernel_size=1, stride=4),
        ])

        # Total downsampling factor: 4 * 4 * 4 = 64
        self.downsample_factor = 64

        self.self_attention = nn.MultiheadAttention(512, num_heads=8,
                                                    batch_first=True, dropout=0.1)
        self.layer_norm = nn.LayerNorm(512)

        self.pool_proj = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        ) if hidden_dim != 512 else nn.Identity()

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(x, list):
            device = next(self.parameters()).device
            x = torch.stack(x).to(device) if isinstance(x[0], torch.Tensor) \
                else torch.tensor(__import__('numpy').array(x), dtype=torch.float32, device=device)

        # per-channel z-score
        x = (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6)

        for conv, res in zip(self.conv_layers, self.residuals):
            identity = res(x)
            x = conv(x)
            if identity.shape[-1] != x.shape[-1]:
                identity = F.adaptive_avg_pool1d(identity, x.shape[-1])
            x = F.gelu(x + identity)

        # self-attention over time dimension
        x_t = x.transpose(1, 2)                          # [B, T_out, 512]
        attn_out, _ = self.self_attention(x_t, x_t, x_t)
        x_t = self.layer_norm(x_t + attn_out)

        seq_features = x_t.transpose(1, 2)               # [B, 512, T_out]

        pooled = x_t.mean(dim=1)                          # [B, 512]
        pooled = self.pool_proj(pooled)                   # [B, hidden_dim]

        return seq_features, pooled