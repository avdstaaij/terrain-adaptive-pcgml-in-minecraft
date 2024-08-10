import torch


# Based on the GANLoss class from pytorch-CycleGAN-and-pix2pix repository:
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class TargetTensorGanLoss(torch.nn.Module):
    """
    A base class for GAN losses that require target tensors with the same shape as the input.

    Instances of this class store the required target tensors internally, which means that they
    can easily be moved to the correct device by moving the entire loss instance to that device.
    """
    def __init__(self, targetValueReal=1.0, targetValueFake=0.0):
        super().__init__()
        self.register_buffer("_targetReal", torch.tensor(targetValueReal))
        self.register_buffer("_targetFake", torch.tensor(targetValueFake))

    def getTargetTensor(self, size: torch.Size, targetIsReal: bool):
        """Returns the specified target tensor with size <size>."""
        targetTensor = self._targetReal if targetIsReal else self._targetFake
        return targetTensor.expand(size)

    def forward(self):
        pass


class SigmoidBceGanLoss(TargetTensorGanLoss):
    """
    Vanilla Sigmoid+BCE GAN loss.

    Notes:
    - Instances of this class need to be moved to the correct device before use.
    - The discriminator should not have a final sigmoid activation.
    """
    def __call__(self, prediction: torch.Tensor, targetIsReal: bool):
        """Returns sigmoid+BCE GAN loss."""
        target = self.getTargetTensor(prediction.size(), targetIsReal)
        return torch.nn.functional.binary_cross_entropy_with_logits(prediction, target)


class LsganLoss(TargetTensorGanLoss):
    """
    Least Squares GAN loss.

    Notes:
    - Instances of this need to be moved to the correct device before use.
    - The discriminator should not have a final sigmoid activation.
    """
    def __call__(self, prediction: torch.Tensor, targetIsReal: bool):
        """Returns LSGAN loss."""
        target = self.getTargetTensor(prediction.size(), targetIsReal)
        return torch.nn.functional.mse_loss(prediction, target)
