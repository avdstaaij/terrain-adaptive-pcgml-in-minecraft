import torch
import torch.nn as nn


class SafeInstanceNorm3d(nn.Module):
    """Acts like InstanceNorm3d, but does nothing (rather than crash) when the input has only one
    spatial element"""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.instanceNorm = nn.InstanceNorm3d(*args, **kwargs)

    def forward(self, x):
        if list(x.shape)[-3:] == [1, 1, 1]:
            return x
        return self.instanceNorm(x)
