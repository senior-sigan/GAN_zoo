from torch import nn


def build_trans(transform_cls, func, with_input=False):
    def _builder(*args, **kwargs) -> ParallelTransform:
        return ParallelTransform(transform_cls, func, with_input, *args, **kwargs)

    return _builder


class ParallelTransform(nn.Module):
    def __init__(self, transform_cls, func, with_input, *args, **kwargs):
        super().__init__()
        self.transform_cls = transform_cls
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.with_input = with_input

    def forward(self, x):
        if hasattr(self.transform_cls, 'get_params'):
            print(self.transform_cls.__name__)
            if self.with_input:
                self.args.append(x)
            params = self.transform_cls.get_params(*self.args)
            return {k: self.func(x[k], params, **self.kwargs) for k in x}
        else:
            return {k: self.func(x[k], **self.kwargs) for k in x}
