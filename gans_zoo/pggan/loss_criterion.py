import torch


def wgangp(prediction: torch.Tensor, is_real: bool) -> torch.Tensor:
    r"""
    Paper WGANGP loss : linear activation for the generator.
    https://arxiv.org/pdf/1704.00028.pdf
    """

    if is_real:
        return -prediction[:, 0].sum()
    return prediction[:, 0].sum()
