import torch
from torch.utils.data import Dataset


class TextClassificationDataset(Dataset):
    """Custom Dataset for text classification with soft and hard labels."""

    def __init__(self, encodings, soft_labels=None, hard_labels=None):
        self.encodings = encodings
        self.soft_labels = soft_labels if soft_labels is not None else [None] * len(hard_labels)
        self.hard_labels = hard_labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        if self.hard_labels is not None:
            item["hard_labels"] = torch.tensor(self.hard_labels[idx], dtype=torch.long) # hard_labels
        if self.soft_labels[idx] is not None:
            item["soft_labels"] = torch.tensor(self.soft_labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.hard_labels) if self.hard_labels is not None else len(self.soft_labels)