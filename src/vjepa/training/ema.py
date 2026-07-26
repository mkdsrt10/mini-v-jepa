import torch


@torch.no_grad()
def update_ema(online: torch.nn.Module, target: torch.nn.Module, momentum: float) -> None:
    for online_param, target_param in zip(online.parameters(), target.parameters()):
        target_param.mul_(momentum).add_(online_param, alpha=1 - momentum)


@torch.no_grad()
def parameter_rms_distance(online: torch.nn.Module, target: torch.nn.Module) -> float:
    """Return the RMS distance between corresponding encoder parameters.

    This is ``sqrt(mean((theta_online - theta_ema)**2))`` over every trainable
    encoder weight.  Unlike a raw L2 norm, RMS distance remains comparable if
    the encoder size changes.  It is measured *after* the EMA update, so it
    reports the residual lag of the target encoder behind the online encoder.
    """
    squared_difference, parameter_count = 0.0, 0
    for online_param, target_param in zip(online.parameters(), target.parameters()):
        difference = (online_param.detach().float() - target_param.detach().float())
        squared_difference += difference.square().sum().item()
        parameter_count += difference.numel()
    if parameter_count == 0:
        return 0.0
    return (squared_difference / parameter_count) ** 0.5
