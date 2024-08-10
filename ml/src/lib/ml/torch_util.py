from typing import Any, Iterable, Callable, Optional, Sequence
from math import prod
import sys
import random

import torch
from torch.utils.data import Dataset, Sampler, DataLoader

from lib.util import tqdm


# ==================================================================================================


def dataStructureToDevice(dataStructure: Any, device: torch.device):
    """Moves all tensors in a (nested) Python data structure to the specified device"""

    if isinstance(dataStructure, torch.Tensor):
        return dataStructure.to(device)

    if isinstance(dataStructure, (list, tuple)):
        return type(dataStructure)(dataStructureToDevice(item, device) for item in dataStructure)

    if isinstance(dataStructure, dict):
        return type(dataStructure)((key, dataStructureToDevice(value, device)) for key, value in dataStructure.items())

    raise ValueError(f"Unsupported type: {type(dataStructure)}")


def computeReceptiveField(kernelSizes: Sequence[int], strides: Sequence[int]):
    """Computes the receptive field size of a single-path convolutional neural network"""
    # Based on https://distill.pub/2019/computing-receptive-fields/
    L = len(kernelSizes)
    assert len(strides) == L or len(strides) == L - 1
    return sum((kernelSizes[l] - 1)*prod(strides[:l]) for l in range(L)) + 1


# ==================================================================================================


class Learner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._device = torch.device("cpu")
        self._epochIndex = 0
        self._batchIndex = 0
        self._globalBatchIndex = 0

    def forward(self, *args, **kwargs):
        pass

    def trainingStep(self, batch):
        """Should perform a single training step"""

    def onEpochEnd(self):
        """Called at the end of each epoch"""

    def logConsole(self, text: str):
        """Log <text> to the console without disturbing the progress bars"""
        tqdm.write(text, file=sys.stderr)

    @property
    def device(self):
        """The device on which the learner is currently stored"""
        return self._device

    @property
    def epochIndex(self):
        """The current epoch index"""
        return self._epochIndex

    @property
    def batchIndex(self):
        """The current batch index"""
        return self._batchIndex

    @property
    def globalBatchIndex(self):
        """The current global batch index"""
        return self._globalBatchIndex

    def toDevice(self, device: torch.device):
        """Moves the learner to the specified device"""
        super().to(device)
        self._device = device
        return self

    def fit(self, trainingData: Iterable, epochCount: int, initialEpoch: int = 0, device: Optional[torch.device] = None):
        """Trains the learner"""

        ### Setup

        if device is not None:
            self.toDevice(device)

        if isinstance(trainingData, DataLoader):
            oldPinMemory = trainingData.pin_memory
            if self.device.type == "cuda":
                trainingData.pin_memory=True

        ### Training loop

        for self._epochIndex in tqdm(range(initialEpoch, epochCount), desc="Epoch", total=epochCount, initial=initialEpoch):
            for self._batchIndex, batch in tqdm(enumerate(trainingData), total=len(trainingData), desc="Batch", leave=False):
                self._globalBatchIndex += 1
                batch = dataStructureToDevice(batch, self.device)
                self.trainingStep(batch)

            self.onEpochEnd()

        ### Cleanup

        if isinstance(trainingData, DataLoader):
            trainingData.pin_memory = oldPinMemory


# ==================================================================================================


class DiscardRandomSampler(Sampler):
    """
    Samples randomly with replacement, but discards (re-rolls) some samples.

    <indexToDiscardProb> should map indices to the probability of discarding the sample at that
    index.

    Unlike WeightedRandomSampler, this sampler does not need additional memory that scales with the
    number of samples, so it can be used with arbitrarily large datasets.
    It does however incur some overhead while sampling: it needs to evaluate <indexToDiscardProb>
    and possibly perform some re-rolls.

    Sampling without replacement is not supported because it would require additional memory.
    """
    def __init__(self, dataset: Dataset, indexToDiscardProb: Callable[[int], float], sampleCount: Optional[int] = None):
        self._dataset = dataset
        self._indexToDiscardProb = indexToDiscardProb
        self._sampleCount = sampleCount if sampleCount is not None else len(dataset)

    def __len__(self):
        return self._sampleCount

    def __iter__(self):
        def sample():
            index = random.randrange(len(self))
            if random.random() < self._indexToDiscardProb(index):
                return sample()
            return index

        for _ in range(len(self)):
            yield sample()
