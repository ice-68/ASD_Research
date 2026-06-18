"""
ImprovedBrainTransformer — Transformer with [CLS] token
原: 01_Classification_Vision_2 / transformer 文件夹
"""
import torch
import torch.nn as nn


class ImprovedBrainTransformer(nn.Module):
    """输出 2 个 logits → CrossEntropyLoss"""

    def __init__(self, num_regions=116, input_dim=116, d_model=64,
                 nhead=4, num_layers=2, num_classes=2):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(d_model, d_model),
        )
        self.region_embedding = nn.Parameter(torch.randn(1, num_regions, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 2, dropout=0.2,
            activation="gelu", batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x):
        B = x.size(0)
        x = self.input_projection(x) + self.region_embedding
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = self.transformer_encoder(x)
        cls_out = self.layer_norm(x[:, 0, :])
        return self.fc(cls_out)
