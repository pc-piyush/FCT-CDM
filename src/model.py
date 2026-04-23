import torch
import torch.nn as nn

class GRUPatientModel(nn.Module):
    def __init__(self, vocab_size=10000, embed_dim=128, hidden=128):
        super().__init__()

        print("[MODEL] Initializing GRU model")

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.gru(x)

        # take last hidden state
        out = out[:, -1, :]

        return self.fc(out).squeeze()