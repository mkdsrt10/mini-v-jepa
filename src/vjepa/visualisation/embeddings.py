import torch


def pairwise_cosine(embeddings: torch.Tensor) -> torch.Tensor:
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    return embeddings @ embeddings.T

