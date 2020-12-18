import torch
import torch.nn.functional as F


def _build_reference(logits: torch.Tensor, is_real: bool) -> torch.Tensor:
    batch_size = logits.size()[0]
    value = 1.0 if is_real else 0.0

    return torch.tensor(
        [value],
        dtype=torch.float,
        device=logits.device,
    ).expand(batch_size)


def wasserstein_loss(prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
    r"""
    Wasserstein loss.
    https://arxiv.org/pdf/1704.00028.pdf
    """
    if is_real:
        return -torch.mean(prediction)
    return torch.mean(prediction)


def gan_loss(logits: torch.Tensor, is_real: bool) -> torch.Tensor:
    reference = _build_reference(logits, is_real)

    preds = torch.sigmoid(logits).view(-1)
    return F.binary_cross_entropy(preds, reference)


def mse_loss(logits: torch.Tensor, is_real: bool) -> torch.Tensor:
    reference = _build_reference(logits, is_real)
    return F.mse_loss(logits.view(-1), reference)
