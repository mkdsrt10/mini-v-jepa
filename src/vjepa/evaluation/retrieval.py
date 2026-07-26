import torch
import torch.nn.functional as F


@torch.no_grad()
def nearest_neighbors(query, gallery, k=5):
    return (F.normalize(query, dim=-1) @ F.normalize(gallery, dim=-1).T).topk(k, dim=-1).indices

