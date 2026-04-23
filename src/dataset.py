import torch
from torch.utils.data import Dataset

class PatientDataset(Dataset):
    def __init__(self, tensors, labels):
        self.pids = list(tensors.keys())
        self.tensors = tensors
        self.labels = labels

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        pid = self.pids[idx]

        return {
            "tensor": self.tensors[pid],
            "label": self.labels.get(pid, 0)
        }