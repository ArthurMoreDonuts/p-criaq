from torch import nn

class ConvNextWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # your modifications...
        for name, param in self.model.named_parameters():
            if "head" not in name:
                 param.requires_grad = False
    def forward(self, x, **kwargs):  # kwargs absorbs sample_weight
        return self.model(x)
