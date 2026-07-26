"""Utilities for frozen-encoder linear classification evaluation."""
import torch
import torch.nn.functional as F
from torch import nn


@torch.no_grad()
def extract_mean_features(encoder: nn.Module, loader, device: str):
    """Pool frozen token features into one normalized vector per video."""
    features, labels = [], []
    encoder.eval()
    for batch in loader:
        tokens = encoder(batch["video"].to(device))  # [B, tokens, dim]
        features.append(F.normalize(tokens.mean(dim=1), dim=-1).cpu())
        labels.append(batch["label"].cpu())
    return torch.cat(features), torch.cat(labels)


def fit_linear_probe(train_features, train_labels, val_features, val_labels,
                     num_classes: int, epochs: int = 100, lr: float = 0.1):
    """Fit a single linear layer; the encoder features remain fully frozen."""
    classifier = nn.Linear(train_features.size(1), num_classes)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(classifier(train_features), train_labels).backward()
        optimizer.step()
    with torch.no_grad():
        predictions = classifier(val_features).argmax(dim=1)
        accuracy = (predictions == val_labels).float().mean().item()
        per_class = {}
        for label in range(num_classes):
            subset = val_labels == label
            per_class[str(label)] = (predictions[subset] == label).float().mean().item() if subset.any() else None
    return accuracy, per_class
