import torch.nn as nn
from pytorch_symbolic import Input, SymbolicModel
from pytorch_symbolic.useful_layers import ConcatLayer

from lib.ml.models.safe_instance_norm import SafeInstanceNorm3d
from lib.util import suppressWarnings


def Vox2voxGenerator(channelsIn: int, channelsOut: int):
    """
    Generator model as used in [vox2vox](https://arxiv.org/abs/2003.13653),
    without a final activation layer.
    """

    def ConvDown(channelsIn: int, channelsOut: int, doInstanceNorm=True, dropout=0.0):
        y = x = Input((channelsIn, 2, 2, 2))
        y = nn.Conv3d(channelsIn, channelsOut, kernel_size=4, stride=2, padding=1, padding_mode="zeros", bias=False)(y, custom_name="conv")
        if doInstanceNorm:
            y = SafeInstanceNorm3d(channelsOut)(y)
        y = nn.LeakyReLU(0.2)(y)
        if dropout > 0.0:
            y = nn.Dropout(dropout)(y)
        return SymbolicModel(x, y)

    def ConvSame(channelsIn: int, channelsOut: int, doInstanceNorm=True, dropout=0.0):
        y = x = Input((channelsIn, 1, 1, 1))
        with suppressWarnings(): # Suppress warning about requiring an additional copy here
            y = nn.Conv3d(channelsIn, channelsOut, kernel_size=4, stride=1, padding="same", padding_mode="zeros", bias=False)(y, custom_name="conv")
        if doInstanceNorm:
            y = SafeInstanceNorm3d(channelsOut)(y)
        y = nn.LeakyReLU(0.2)(y)
        if dropout > 0.0:
            y = nn.Dropout(dropout)(y)
        return SymbolicModel(x, y)

    def ConvUp(channelsIn: int, channelsOut: int, doInstanceNorm=True, dropout=0.0):
        y = x = Input((channelsIn, 1, 1, 1))
        y = nn.ConvTranspose3d(channelsIn, channelsOut, kernel_size=4, stride=2, padding=1, padding_mode="zeros", bias=False)(y, custom_name="convTranspose")
        if doInstanceNorm:
            y = SafeInstanceNorm3d(channelsOut)(y)
        y = nn.ReLU(inplace=True)(y)
        if dropout > 0.0:
            y = nn.Dropout(dropout)(y)
        return SymbolicModel(x, y)

    x = Input((channelsIn, 16, 16, 16))

    d1 = ConvDown(channelsIn, 64, doInstanceNorm=False)(x, custom_name="down1")
    d2 = ConvDown(64,  128)(d1, custom_name="down2")
    d3 = ConvDown(128, 256)(d2, custom_name="down3")
    d4 = ConvDown(256, 512)(d3, custom_name="down4")

    s1 = ConvSame(512,  512, dropout=0.2)(d4, custom_name="same1")
    s2 = ConvSame(1024, 512, dropout=0.2)(ConcatLayer(-4)(s1, d4), custom_name="same2")
    s3 = ConvSame(1024, 512, dropout=0.2)(ConcatLayer(-4)(s2, s1), custom_name="same3")
    s4 = ConvSame(1024, 512, dropout=0.2)(ConcatLayer(-4)(s3, s2), custom_name="same4")

    u1 = ConvUp  (1024, 256)(ConcatLayer(-4)(s4, s3), custom_name="up1")
    u2 = ConvUp  (512,  128)(ConcatLayer(-4)(u1, d3), custom_name="up2")
    u3 = ConvUp  (256,  64 )(ConcatLayer(-4)(u2, d2), custom_name="up3")

    y = nn.ConvTranspose3d(128, channelsOut, kernel_size=4, stride=2, padding=1, padding_mode="zeros", bias=False)(ConcatLayer(-4)(u3, d1), custom_name="output_convTranspose")

    return SymbolicModel(x, y)


# downConvCount | receptive field
# -------------------------------
# 0             | 4
# 1             | 10
# 2             | 22
# 3             | 46
# 4             | 94
def Vox2voxDiscriminator(channelsIn: int, downConvCount: int = 4, initialConvChannels: int = 64):
    """
    Patch discriminator model as used in [vox2vox](https://arxiv.org/abs/2003.13653).
    """

    def ConvDown(channelsIn: int, channelsOut: int, doInstanceNorm=True):
        y = x = Input((channelsIn, 2, 2, 2))
        y = nn.Conv3d(channelsIn, channelsOut, kernel_size=4, stride=2, padding=1, padding_mode="zeros")(y, custom_name="conv")
        if doInstanceNorm:
            y = SafeInstanceNorm3d(channelsOut)(y)
        y = nn.LeakyReLU(0.2, inplace=True)(y)
        return SymbolicModel(x, y)

    minInputSize = 2 ** downConvCount
    input1 = Input((channelsIn, minInputSize, minInputSize, minInputSize))
    input2 = Input((channelsIn, minInputSize, minInputSize, minInputSize))
    x = ConcatLayer(-4)(input1, input2)

    channelsIn = 2 * channelsIn
    for i in range(downConvCount):
        channelsOut = initialConvChannels * (2 ** i)
        x = ConvDown(channelsIn, channelsOut, doInstanceNorm=(i != 0))(x, custom_name=f"down{i+1}")
        channelsIn = channelsOut

    y = nn.Conv3d(channelsIn, 1, kernel_size=4, stride=1, padding="same", padding_mode="zeros", bias=False)(x, custom_name="output_conv")

    return SymbolicModel((input1, input2), y)
