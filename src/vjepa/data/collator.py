import torch


def collate_videos(batch):
    collated = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if torch.is_tensor(values[0]):
            collated[key] = torch.stack(values)
        elif key == "label":
            collated[key] = torch.tensor(values, dtype=torch.long)
        else:
            collated[key] = values
    return collated
