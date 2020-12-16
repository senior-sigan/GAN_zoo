import torch
import torch.nn.functional as F


def wasserstein_loss(prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
    r"""
    Wasserstein loss.
    https://arxiv.org/pdf/1704.00028.pdf
    """
    if is_real:
        return -torch.mean(prediction)
    return torch.mean(prediction)


def gan_loss(logits: torch.Tensor, is_real: bool) -> torch.Tensor:
    batch_size = logits.size()[0]
    value = int(is_real)

    reference = torch.tensor(
        [value],
        dtype=torch.float,
        device=logits.device,
    ).expand(batch_size)

    preds = torch.sigmoid(logits).view(-1)
    return F.binary_cross_entropy(preds, reference)
