from torch.optim.lr_scheduler import LambdaLR


class LinearLRCallback:
    def __init__(
        self,
        n_epochs: int,
        decay_start_epoch: int = 100,
        offset: int = 0,
    ) -> None:
        """
        Lambda wrapper for torch LambdaLR to implement Linear LR.

        Keep the same learning rate for the first <opt.n_epochs> epochs
        and linearly decay the rate to zero over the next
        <opt.n_epochs_decay> epochs.

        :param n_epochs: total number of epochs
        :param offset: if start training from scratch it's 0.
            If continue training it should be the starting epoch.
        :param decay_start_epoch: epoch from which to start lr decay
        """
        if (n_epochs - decay_start_epoch) <= 0:
            msg = 'Decay must start before the training session ends' + \
                  ': n_epochs={0} decay_start_epoch={1}!'
            raise AssertionError(msg.format(n_epochs, decay_start_epoch))

        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def __call__(self, epoch: int) -> float:
        """

        :param epoch: current epoch of training
        :return:
        """
        a = max(0, epoch + self.offset - self.decay_start_epoch)
        b = self.n_epochs - self.decay_start_epoch
        return 1.0 - a / float(b)


def LinearLR(
    optimizer,
    n_epochs: int,
    decay_start_epoch: int,
    offset: int,
) -> LambdaLR:
    """
    Builder for LinearLR scheduler.

    :param optimizer:
    :param n_epochs:
    :param offset:
    :param decay_start_epoch:
    :return:
    """
    lr_lambda = LinearLRCallback(n_epochs, decay_start_epoch, offset)
    return LambdaLR(optimizer, lr_lambda=lr_lambda)
