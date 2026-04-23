import torch

def collate_fn(batch):
    tensors = [b["tensor"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.float)

    # flatten events per patient
    sequences = []

    for t in tensors:
        seq = []

        for domain in t:
            for event in t[domain]:
                if len(event) == 2:
                    seq.append(event[0] * 1000 + event[1])
                elif len(event) == 3:
                    seq.append(event[0] * 1000 + event[1])

        sequences.append(seq[:200])  # truncate

    # padding
    max_len = max(len(s) for s in sequences)

    padded = []
    for s in sequences:
        s = s + [0] * (max_len - len(s))
        padded.append(s)

    return torch.tensor(padded, dtype=torch.long), labels