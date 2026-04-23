import torch
import torch.nn as nn

class GRUModel(nn.Module):

    def __init__(self, num_concepts, embed_dim=128, hidden_dim=128):

        super().__init__()

        # -----------------------------
        # EMBEDDING (CRITICAL FIX)
        # -----------------------------
        self.embed = nn.Embedding(
            num_concepts,
            embed_dim,
            padding_idx=0
        )

        # -----------------------------
        # GRU
        # -----------------------------
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # -----------------------------
        # OUTPUT HEAD
        # -----------------------------
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_domain, x_concept, x_time):

        # x_concept: (B, T)

        mask = (x_concept != 0).float()  # padding mask

        emb = self.embed(x_concept)  # (B, T, D)

        # GRU
        out, _ = self.gru(emb)

        # -----------------------------
        # MASKED POOLING (IMPORTANT)
        # -----------------------------
        mask = mask.unsqueeze(-1)
        out = out * mask

        summed = out.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)

        pooled = summed / lengths

        return self.fc(pooled).squeeze(-1)
    
    def get_patient_embedding(self, x_concept):

        mask = (x_concept != 0).float()

        emb = self.embed(x_concept)

        out, _ = self.gru(emb)

        mask = mask.unsqueeze(-1)

        out = out * mask

        pooled = out.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return pooled