import torch
import torch.nn.functional as F


@torch.no_grad()
def knn_accuracy(train_features, train_labels, test_features, test_labels, k=20):
    sim = F.normalize(test_features, dim=1) @ F.normalize(train_features, dim=1).T
    labels = train_labels[sim.topk(k, dim=1).indices]
    votes = torch.stack([torch.bincount(row, minlength=int(train_labels.max()) + 1).argmax() for row in labels])
    return (votes == test_labels).float().mean().item()

