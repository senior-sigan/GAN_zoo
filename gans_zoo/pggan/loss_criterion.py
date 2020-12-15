def wgangp(prediction, is_real: bool):
    r"""
    Paper WGANGP loss : linear activation for the generator.
    https://arxiv.org/pdf/1704.00028.pdf
    """

    if is_real:
        return -prediction[:, 0].sum()
    return prediction[:, 0].sum()
