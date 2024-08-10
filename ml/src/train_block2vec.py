#!/usr/bin/env python3

# pylint: disable=not-callable
# pylint: disable=arguments-differ

from typing import Optional
import os
import sys
import random

import cloup
from termcolor import colored
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset, IterableDataset, DataLoader, get_worker_info

from lib.util import eprint, tqdmProgressCallback
from lib.constants import FILENAME_STYLE, CLI_OPTION_STYLE, ERROR_PREFIX, WARNING_PREFIX
from lib.palette_tools import loadPalette, countBlockTypeCounts, CountBlockTypeCountsError
from lib.ml.torch_util import Learner
from lib.ml.models.skip_gram import SkipGram


# block2vec cannot embed types that never occur in the non-edge part of the block array.
#
# There are two reasons why a block array may have missing non-edge block types:
# 1. The array has a block type that is only used in the edge parts of the dataset.
# 2. The array simply has missing block types.
#
# There are three ways to handle this problem:
# A. Simply embed these types to their random initializations (and possibly log a warning).
# B. Create a mapping (block index)->(embedding index) that skips the missing block types.
# C. Disallow arrays with missing block types.
#
# For cause 1:
# - Solution A is what would happen "by default".
# - Solution B is not appropriate because the edge blocks still need to be embeddable.
# - Solution C is not appropriate because the detection/prevention of cause 1 is very
#   annoying and depends on the chosen neighbor radius.
#
# For cause 2:
# - Solution A is viable, but may result in unexpected behavior when the embedding is used.
#   The effect of this solution cannot easily be replicated by solutions B and C.
# - Solution B is viable, but it is more complex and it essentially defers the problem to
#   the code that will use the embedding.
# - Solution C is viable, but enforces data preprocessing (it "pre-fers" the problem).
#
# Another issue with cause 2 is that we have to read the block palette to detect if there
# are missing block types whose indices are higher than highest non-missing one.
#
# We have chosen to go with solution A for both causes.
#
# To summarize: We will "randomly" embed block types that do not occur in the non-edge
# parts of the dataset.


DEFAULT_DEVICE_NAME = "cuda" if torch.cuda.is_available() else "cpu"


class Block2VecDataset(Dataset):
    def __init__(self, blockArray: np.ndarray, neighborRadius: int = 1):
        self._blockArray = blockArray.view()
        self._neighborRadius = neighborRadius

        shape = self._blockArray.shape
        assert len(shape) == 4
        assert shape[1] >= 2*neighborRadius + 1 and shape[2] >= 2*neighborRadius + 1 and shape[3] >= 2*neighborRadius + 1
        assert np.issubdtype(self._blockArray.dtype, np.integer)

    @property
    def blockArray(self):
        return self._blockArray

    def indexToPosition(self, index: int):
        shape = self._blockArray.shape
        nr    = self._neighborRadius
        return tuple(np.unravel_index(index, (shape[0], shape[1] - 2*nr, shape[2] - 2*nr, shape[3] - 2*nr)) + np.array([0, nr, nr, nr]))

    def getContextAtPosition(self, position):
        contextPositions = [
           (position[0], position[1] + dy, position[2] + dx, position[3] + dz)
            for dy in range(-self._neighborRadius, self._neighborRadius + 1)
            for dx in range(-self._neighborRadius, self._neighborRadius + 1)
            for dz in range(-self._neighborRadius, self._neighborRadius + 1)
            if (dy, dx, dz) != (0, 0, 0)
        ]
        return np.array([self._blockArray[p] for p in contextPositions])

    def __len__(self):
        shape = self._blockArray.shape
        return shape[0] * (shape[1] - 2*self._neighborRadius) * (shape[2] - 2*self._neighborRadius) * (shape[3] - 2*self._neighborRadius)

    def __getitem__(self, idx):
        position = self.indexToPosition(idx)
        target = self._blockArray[position]
        context = self.getContextAtPosition(position)
        return torch.tensor(target, dtype=torch.int32), torch.from_numpy(context.astype(np.int32))


class Block2VecDatasetDiscardRandomWrapper(IterableDataset): # pylint: disable=abstract-method
    """
    Wrapper around Block2VecDataset that samples randomly with replacement and discards
    (re-rolls) some samples.

    <discardProbs> should map target block indices to their discarding probability.

    Implementing this as a dataset wrapper is more efficient than using a sampler for two reasons:
    - All dataloader workers can simultaneously perform re-rolls.
    - Non-rerolled samples do not have to be fetched twice (once by the sampler and once by the
      dataloader).

    It's essential that all dataset workers use a different random seed (for Python's standard
    random module).
    """
    def __init__(self, block2vecDataset: Block2VecDataset, discardProbs: np.ndarray, sampleCount: Optional[int] = None):
        self._block2vecDataset = block2vecDataset
        self._discardProbs = discardProbs
        self.sampleCount = len(block2vecDataset) if sampleCount is None else sampleCount

    def __len__(self):
        return self.sampleCount

    def __iter__(self):
        for _ in range(len(self)):
            while True:
                index = random.randrange(len(self._block2vecDataset))
                position = self._block2vecDataset.indexToPosition(index)
                target = self._block2vecDataset.blockArray[position]
                if random.random() < self._discardProbs[target]:
                    continue # re-roll
                context = self._block2vecDataset.getContextAtPosition(position)
                yield torch.tensor(target, dtype=torch.int32), torch.from_numpy(context.astype(np.int32))
                break


def neighborRadiusToContextSize(neighborRadius: int):
    return (2*neighborRadius + 1)**3 - 1


@cloup.command(context_settings={"show_default": True})
@cloup.option_group(
    "Task settings",
    cloup.option("--dataset-dir",            type=cloup.Path(exists=True, file_okay=False), required=True, help="Dataset directory."),
    cloup.option("--output-dir",             type=cloup.Path(),                             required=True, help="Output directory."),
    cloup.option("--half1/--no-half1",       default=False,                                                help="Use first half of input paired block array.", show_default=False),
    cloup.option("--half2/--no-half2",       default=False,                                                help="Use second half of input paired block array.", show_default=False),
    cloup.option("--embedding-size",         type=int, required=True,                                      help="Size of the embeddings"),
)
@cloup.constraint(cloup.constraints.require_any, ["half1", "half2"])
@cloup.option_group(
    "Initialization",
    cloup.option("--initial-ckpt-dir",       type=cloup.Path(exists=True, file_okay=False), help="Directory containing initial checkpoint."),
    cloup.option("--initial-epoch",          type=int,                                      help="Epoch to start training from."),
    cloup.option("--continue", "continue_",  is_flag=True,                                  help="Continue from last checkpoint in output dir."),
)
@cloup.constraint(cloup.constraints.mutually_exclusive, ["continue_", "initial_ckpt_dir"])
@cloup.constraint(cloup.constraints.mutually_exclusive, ["continue_", "initial_epoch"])
@cloup.option_group(
    "Checkpointing",
    cloup.option("--save-epoch-freq",        type=int,   default=1, help="Epochs between checkpoints."),
)
@cloup.option_group(
    "Training hyperparameters",
    cloup.option("--epoch-size",             type=int,                              help="Epoch size (defaults to dataset size)"),
    cloup.option("--epoch-count",            type=int,   default=100,               help="Number of epochs to train for."),
    cloup.option("--batch-size",             type=int,   default=32,                help="Batch size."),
    cloup.option("--neighbor-radius",        type=int,   default=1,                 help="Radius of block context"),
    cloup.option("--discard-factor",         type=float, default=0.001,             help="Subsampling/discard factor."),
    cloup.option("--lr",                     type=float, default=0.0002,            help="Adam learning rate."),
    cloup.option("--beta1",                  type=float, default=0.5,               help="Adam beta1 parameter."),
    cloup.option("--beta2",                  type=float, default=0.999,             help="Adam beta2 parameter."),
)
@cloup.option("--device", "device_name",     type=str,   default=DEFAULT_DEVICE_NAME, help=f'Device to use (e.g. "cuda:0", "cpu"). Defaults to "cuda" if available, otherwise "cpu". Current default: "{DEFAULT_DEVICE_NAME}".', show_default=False)
def cli(
    dataset_dir:      str,
    output_dir:       str,
    half1:            bool,
    half2:            bool,
    embedding_size:   int,
    initial_ckpt_dir: Optional[str],
    initial_epoch:    Optional[int],
    continue_:        bool,
    save_epoch_freq:  int,
    epoch_size:       Optional[int],
    epoch_count:      int,
    batch_size:       int,
    neighbor_radius:  int,
    discard_factor:   float,
    lr:               float,
    beta1:            float,
    beta2:            float,
    device_name:      Optional[str],
):
    if os.path.exists(output_dir) and not continue_:
        eprint(
            f"Error: The output directory {colored(output_dir, **FILENAME_STYLE)} already exists.\n"
             "\n"
             "If the training process crashed or was interrupted, or if want to train for more\n"
             "epochs in the same output directory, you can continue from the last checkpoint\n"
            f"by passing the {colored('--continue', **CLI_OPTION_STYLE)} flag."
        )
        sys.exit(1)

    if continue_ and not os.path.exists(output_dir):
        eprint(
            f"Error: {colored('--continue', **CLI_OPTION_STYLE)} was passed, but the output directory\n"
            f"{colored(output_dir, **FILENAME_STYLE)} does not exist."
        )
        sys.exit(1)

    if continue_:
        lastCheckpoint = max(int(epochDir.split("-")[-1]) for epochDir in os.listdir(output_dir))
        initial_ckpt_dir = f"{output_dir}/epoch-{lastCheckpoint}"
        initial_epoch = lastCheckpoint + 1

    initial_epoch = 0 if initial_epoch is None else initial_epoch

    train(
        datasetDir     = dataset_dir,
        outputDir      = output_dir,
        half1          = half1,
        half2          = half2,
        embeddingSize  = embedding_size,
        initialCkptDir = initial_ckpt_dir,
        initialEpoch   = initial_epoch,
        saveEpochFreq  = save_epoch_freq,
        epochSize      = epoch_size,
        epochCount     = epoch_count,
        batchSize      = batch_size,
        neighborRadius = neighbor_radius,
        discardFactor  = discard_factor,
        lr             = lr,
        beta1          = beta1,
        beta2          = beta2,
        deviceName     = device_name,
    )


def train(
    datasetDir:     str,
    outputDir:      str,
    half1:          bool,
    half2:          bool,
    embeddingSize:  int,
    initialCkptDir: Optional[str],
    initialEpoch:   int,
    saveEpochFreq:  int,
    epochSize:      Optional[int],
    epochCount:     int,
    batchSize:      int,
    neighborRadius: int,
    discardFactor:  float,
    lr:             float,
    beta1:          float,
    beta2:          float,
    deviceName:     Optional[str],
):
    random.seed(0)
    np.random.seed(0)
    # This will not completely guarantee reproducibility. See:
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(0)

    device = torch.device(deviceName)

    ### Set up dataloader

    blockArray = np.load(f"{datasetDir}/blocks.npy", mmap_mode="r")
    blockArray = blockArray[:, (0 if half1 else 1):(2 if half2 else 1)]
    blockArray.shape = (blockArray.shape[0] * blockArray.shape[1], blockArray.shape[2], blockArray.shape[3], blockArray.shape[4])

    b2vDataset = Block2VecDataset(blockArray, neighborRadius=neighborRadius)

    if epochSize is None:
        epochSize = len(b2vDataset)

    palette = loadPalette(f"{datasetDir}/palette.json")

    blockTypeCount = len(palette)

    # It is unclear whether we should count blocks in the edge parts of the dataset (those that can
    # act as context, but never as a target), but the difference is probably negligible. We do count
    # them here.
    try:
        blockTypeCounts = countBlockTypeCounts(
            b2vDataset.blockArray,
            blockTypeCount,
            progressCallback=tqdmProgressCallback(desc="Counting block type freqs", file=sys.stderr, dynamic_ncols=True)
        )
    except CountBlockTypeCountsError as e:
        eprint(ERROR_PREFIX + f"The block index array contains the index {e.badIndex}, but the palette contains only {blockTypeCount} blocks. (Max index should be {blockTypeCount-1}.)")
        sys.exit(1)


    if any(blockTypeCounts == 0):
        eprint(WARNING_PREFIX + "Some block types never occur in the dataset. These block types will not get useful embeddings.")
        #eprint(f"Missing block types:\n{', '.join(str(palette[i]) for i in np.where(blockTypeCounts == 0)[0])}")

    blockTypeFreqs = blockTypeCounts / np.sum(blockTypeCounts)
    nonZeroBlockTypeFreqs = np.where(blockTypeFreqs == 0, 1, blockTypeFreqs)
    discardProbs = 1.0 - (np.sqrt(nonZeroBlockTypeFreqs / discardFactor) + 1) * (discardFactor / nonZeroBlockTypeFreqs)

    workerCount = os.cpu_count() or 1

    # Every worker gets their own copy of the dataset and will "exhaust" it fully, so we need to
    # divide the artificial epoch size through the number of workers. To avoid having to deal with
    # aligning on (num_workers * batch_size), we just remove the remainder here.
    epochSize -= epochSize % (workerCount * batchSize)

    dataset = Block2VecDatasetDiscardRandomWrapper(b2vDataset, discardProbs, sampleCount = epochSize)

    def worker_init_fn(worker_id):
        workerInfo = get_worker_info()
        random.seed((random.random(), worker_id))
        workerInfo.dataset.sampleCount //= workerInfo.num_workers

    dataloader = DataLoader(
        dataset,
        batch_size     = batchSize,
        shuffle        = False,
        num_workers    = workerCount,
        worker_init_fn = worker_init_fn
    )

    ### Train

    class Block2VecLearner(Learner):
        def __init__(self):
            super().__init__()
            self.model     = SkipGram(blockTypeCount, embeddingSize, neighborRadiusToContextSize(neighborRadius))
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, betas=(beta1, beta2))

        def trainingStep(self, batch):
            predictions = self.model(batch[0])
            loss = torch.nn.functional.cross_entropy(torch.flatten(predictions, end_dim=-2), torch.flatten(batch[1]).to(torch.int64))

            if self.globalBatchIndex % 100 == 0:
                self.logConsole(f"loss: {loss}")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        def onEpochEnd(self):
            self.logConsole(f"== Epoch {self.epochIndex} finished. ==")
            if self.epochIndex % saveEpochFreq == 0 or self.epochIndex == epochCount - 1:
                self.saveCheckpoint(f"{outputDir}/epoch-{self.epochIndex}")

        def saveCheckpoint(self, path: str):
            embeddings = self.model.state_dict()[self.model.embeddingsKey].detach().cpu().numpy() # pylint: disable=unsubscriptable-object
            trainingState = {
                "model":     {key: value for key, value in self.model.state_dict().items() if key != self.model.embeddingsKey},
                "optimizer": self.optimizer.state_dict()
            }
            os.makedirs(path, exist_ok=False)
            np.save(f"{path}/embeddings.npy", embeddings)
            torch.save(trainingState, f"{path}/training-state.pt")

        def loadCheckpoint(self, path: str):
            embeddings = np.load(f"{path}/embeddings.npy")
            trainingState = torch.load(f"{path}/training-state.pt", map_location=self.device)
            trainingState["model"].update({self.model.embeddingsKey: torch.from_numpy(embeddings).to(self.device)})
            self.model    .load_state_dict(trainingState["model"])
            self.optimizer.load_state_dict(trainingState["optimizer"])

    learner = Block2VecLearner().toDevice(device)

    if initialCkptDir is not None:
        learner.loadCheckpoint(initialCkptDir)

    eprint()
    Block2VecLearner().fit(trainingData=dataloader, epochCount=epochCount, initialEpoch=initialEpoch)


def main():
    cli() # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    main()
