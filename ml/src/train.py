#!/usr/bin/env python3

# pylint: disable=not-callable

from typing import Any, Dict, Optional, List
import os
import sys

import cloup
from termcolor import colored
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset, DataLoader
import wandb

from lib.util import eprint, promptConfirmation, tqdm
from lib.constants import ERROR_PREFIX, WARNING_PREFIX, FILENAME_STYLE, CLI_OPTION_STYLE
from lib.ml.torch_util import Learner
from lib.ml.models.vox2vox import Vox2voxGenerator, Vox2voxDiscriminator
from lib.ml.numpy_dataset import NumpyPairDataset
from lib.ml.losses import LsganLoss


# ==================================================================================================


def loadDatasetBinary(datasetDir: str):
    def transform(x: torch.Tensor):
        x = x * 2 - 1 # Map [0,1] to [-1, 1]
        return x.unsqueeze_(0).to(dtype=torch.float32)
    return NumpyPairDataset(f"{datasetDir}/blocks.npy", targetDtype=None, useMmap=True, transform=transform)


def loadDatasetEmbeddings(datasetDir: str, embeddingsPath: str):
    embeddings = torch.from_numpy(np.load(embeddingsPath))
    assert embeddings.ndim == 2
    embeddingSize = int(embeddings.shape[1])
    def transform(x: torch.Tensor):
        return torch.moveaxis(embeddings[x], -1, 0).to(dtype=torch.float32)
    return NumpyPairDataset(f"{datasetDir}/blocks.npy", targetDtype=np.int32, useMmap=True, transform=transform), embeddingSize


# ==================================================================================================


defaultDataloaderWorkers = os.cpu_count() or 1
defaultDeviceName = "cuda" if torch.cuda.is_available() else "cpu"


@cloup.group(context_settings={"show_default": True})
def cli():
    # This will not completely guarantee reproducibility. See:
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(0)


# ==================================================================================================


@cli.command()
@cloup.option_group(
    "Task settings",
    cloup.option("--binary/--no-binary",     default=False,                                                help="Train on binary dataset", show_default=False),
    cloup.option("--dataset-dir",            type=cloup.Path(exists=True, file_okay=False), required=True, help="Dataset directory."),
    cloup.option("--embeddings-path",        type=cloup.Path(exists=True, dir_okay=False),                 help="Path to embeddings file."),
    cloup.option("--output-dir",             type=cloup.Path(),                             required=True, help="Output directory."),
)
@cloup.constraint(cloup.constraints.If("binary", then=cloup.constraints.accept_none, else_=cloup.constraints.require_all), ["embeddings_path"])
@cloup.option_group(
    "Training hyperparameters",
    cloup.option("--epoch-count",            type=int,   default=100,               help="Number of epochs to train for."),
    cloup.option("--batch-size",             type=int,   default=8,                 help="Batch size."),
    cloup.option("--lr",                     type=float, default=0.0002,            help="Adam learning rate."),
    cloup.option("--beta1",                  type=float, default=0.5,               help="Adam beta1 parameter."),
    cloup.option("--beta2",                  type=float, default=0.999,             help="Adam beta2 parameter."),
    cloup.option("--d-down-conv-count",      type=int,   default=4,                 help="Number of down-convs in D."),
)
@cloup.option_group(
    "Initialization",
    cloup.option("--initial-ckpt-dir",       type=cloup.Path(exists=True, file_okay=False), help="Directory containing initial checkpoint."),
    cloup.option("--initial-g-weight-path",  type=cloup.Path(exists=True, dir_okay=False),  help="Path to initial generator weights."),
    cloup.option("--initial-d-weight-path",  type=cloup.Path(exists=True, dir_okay=False),  help="Path to initial discriminator weights."),
    cloup.option("--initial-epoch",          type=int,                                      help="Epoch to start training from."),
    cloup.option("--continue", "continue_",  is_flag=True,                                  help="Continue from last checkpoint in output dir."),
)
@cloup.constraint(cloup.constraints.mutually_exclusive, ["continue_", "initial_ckpt_dir", "initial_g_weight_path", "initial_d_weight_path"])
@cloup.constraint(cloup.constraints.mutually_exclusive, ["continue_", "initial_epoch"])
@cloup.option_group(
    "Checkpointing",
    cloup.option("--save-epoch-freq",        type=int,   default=5, help="Epochs between checkpoints."),
)
@cloup.option_group(
    "Training process",
    cloup.option("--dataloader-workers",     type=int, default=defaultDataloaderWorkers, help=f"Number of dataloader workers. Defaults to the number of CPUs. Current default: {defaultDataloaderWorkers}.", show_default=False),
    cloup.option("--device", "device_name",  type=str, default=defaultDeviceName, help=f'Device to use (e.g. "cuda:0", "cpu"). Defaults to "cuda" if available, otherwise "cpu". Current default: "{defaultDeviceName}".', show_default=False),
)
@cloup.option_group(
    "Wandb",
    cloup.option("--wandb-entity",           type=str,                 help="Wandb entity name."),
    cloup.option("--wandb-project",          type=str,                 help="Wandb project name."),
    cloup.option("--wandb-log-freq",         type=int,                 help="Batches between wandb logs.")
)
@cloup.constraint(cloup.constraints.If("wandb_project", then=cloup.constraints.require_all, else_=cloup.constraints.accept_none), ["wandb_entity", "wandb_log_freq"])
def vox2vox(
    binary:                bool,
    dataset_dir:           str,
    embeddings_path:       Optional[str],
    output_dir:            str,
    epoch_count:           int,
    batch_size:            int,
    lr:                    float,
    beta1:                 float,
    beta2:                 float,
    d_down_conv_count:     int,
    initial_ckpt_dir:      Optional[str],
    initial_g_weight_path: Optional[str],
    initial_d_weight_path: Optional[str],
    initial_epoch:         Optional[int],
    continue_:             bool,
    save_epoch_freq:       int,
    dataloader_workers:    int,
    device_name:           Optional[str],
    wandb_entity:          Optional[str],
    wandb_project:         Optional[str],
    wandb_log_freq:        int,
):
    if os.path.exists(output_dir) and not continue_:
        eprint(
            f"{ERROR_PREFIX}The output directory {colored(output_dir, **FILENAME_STYLE)} already exists.\n"
             "\n"
             "If the training process crashed or was interrupted, or if want to train for more\n"
             "epochs in the same output directory, you can continue from the last checkpoint\n"
            f"by passing the {colored('--continue', **CLI_OPTION_STYLE)} flag."
        )
        sys.exit(1)

    if continue_ and not os.path.exists(f"{output_dir}/checkpoints"):
        eprint(
            f"{ERROR_PREFIX}{colored('--continue', **CLI_OPTION_STYLE)} was passed, but the checkpoint directory\n"
            f"{colored(f'{output_dir}/checkpoints', **FILENAME_STYLE)} does not exist."
        )
        sys.exit(1)

    initial_other_state_path: Optional[str] = None

    if continue_:
        lastCheckpoint = max(int(epochDir.split("-")[-1]) for epochDir in os.listdir(f"{output_dir}/checkpoints"))
        initial_ckpt_dir = f"{output_dir}/checkpoints/epoch-{lastCheckpoint}"
        initial_epoch = lastCheckpoint + 1

    if initial_ckpt_dir is not None:
        initial_g_weight_path = f"{initial_ckpt_dir}/generator.pt"
        initial_d_weight_path = f"{initial_ckpt_dir}/discriminator.pt"
        initial_other_state_path = f"{initial_ckpt_dir}/other-state.pt"

    device = torch.device(device_name)

    initialWeightsG   = None if initial_g_weight_path    is None else torch.load(initial_g_weight_path,    map_location=device)
    initialWeightsD   = None if initial_d_weight_path    is None else torch.load(initial_d_weight_path,    map_location=device)
    initialOtherState = None if initial_other_state_path is None else torch.load(initial_other_state_path, map_location=device)
    initial_epoch = 0 if initial_epoch is None else initial_epoch

    # ----------------------------------------------------------------------------------------------

    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(f"{output_dir}/metrics.csv"):
        metricsFile = open(f"{output_dir}/metrics.csv", "a", encoding="utf-8", buffering=1)
    else:
        metricsFile = open(f"{output_dir}/metrics.csv", "w", encoding="utf-8", buffering=1)
        metricsFile.write("epoch,batch,loss-g,loss-d,loss-d-real,loss-d-fake\n")

    use_wandb = wandb_project is not None
    if use_wandb:
        wandb.login()
        modelName = "vox2vox-binary" if binary else "vox2vox-embedding"
        wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            group=modelName,
            config={
                "model":             modelName,
                "epoch-count":       epoch_count,
                "batch-size":        batch_size,
                "lr":                lr,
                "beta1":             beta1,
                "beta2":             beta2,
                "d-down-conv-count": d_down_conv_count,
                "continue":          continue_,
            },
            dir=f"{output_dir}"
        )

    if binary:
        dataset = loadDatasetBinary(dataset_dir)
    else:
        dataset, embeddingSize = loadDatasetEmbeddings(dataset_dir, embeddings_path)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=dataloader_workers)

    class Vox2VoxLearner(Learner):
        def __init__(self):
            super().__init__()

            if binary:
                self.generator     = Vox2voxGenerator(1, 1)
                self.discriminator = Vox2voxDiscriminator(1, downConvCount=d_down_conv_count)
                self.generatorActivation = torch.nn.functional.tanh
            else:
                self.generator     = Vox2voxGenerator(embeddingSize, embeddingSize)
                self.discriminator = Vox2voxDiscriminator(embeddingSize, downConvCount=d_down_conv_count)
                self.generatorActivation = lambda x:x

            self.optimizerG = torch.optim.Adam(self.generator    .parameters(), lr=lr, betas=(beta1, beta2))
            self.optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, beta2))

            self.lossGan = LsganLoss()

            if use_wandb:
                wandb.watch(self.generator,     log="all", log_freq=wandb_log_freq)
                wandb.watch(self.discriminator, log="all", log_freq=wandb_log_freq)

        def trainingStep(self, batch):
            a, bReal = batch

            # Train generator

            bFake = self.generator(a)
            bFake = self.generatorActivation(bFake)
            predFake = self.discriminator(a, bFake)
            lossG = self.lossGan(predFake, True)

            self.optimizerG.zero_grad()
            lossG.backward()
            self.optimizerG.step()

            # Train discriminator

            bFake = self.generator(a).detach() # Don't backpropagate through generator here
            if binary:
                torch.sign(bFake, out=bFake)
            predReal = self.discriminator(a, bReal)
            predFake = self.discriminator(a, bFake)
            lossReal = self.lossGan(predReal, True)
            lossFake = self.lossGan(predFake, False)
            lossD = 0.5 * (lossReal + lossFake)

            self.optimizerD.zero_grad()
            lossD.backward()
            self.optimizerD.step()

            metricsFile.write(f"{self.epochIndex},{self.batchIndex},{lossG.item()},{lossD.item()},{lossReal.item()},{lossFake.item()}\n")
            if use_wandb and self.batchIndex % wandb_log_freq == 0:
                wandb.log({
                    "loss-g": lossG.item(),
                    "loss-d": lossD.item(),
                    "loss-d-real": lossReal.item(),
                    "loss-d-fake": lossFake.item(),
                })

        def onEpochEnd(self):
            if self.epochIndex % save_epoch_freq == 0 or self.epochIndex == epoch_count - 1:
                self.saveCheckpoint()

        def saveCheckpoint(self):
            checkpointDir = f"{output_dir}/checkpoints/epoch-{self.epochIndex}"
            os.makedirs(checkpointDir, exist_ok=False)
            torch.save(self.generator.state_dict(),     f"{checkpointDir}/generator.pt")
            torch.save(self.discriminator.state_dict(), f"{checkpointDir}/discriminator.pt")
            otherState = {
                "optimizerG": self.optimizerG.state_dict(),
                "optimizerD": self.optimizerD.state_dict(),
            }
            torch.save(otherState, f"{checkpointDir}/other-state.pt")

        def loadOtherStateDict(self, otherState: Dict):
            self.optimizerG.load_state_dict(otherState["optimizerG"])
            self.optimizerD.load_state_dict(otherState["optimizerD"])

    learner = Vox2VoxLearner().toDevice(device)

    if initialWeightsG   is not None: learner.generator    .load_state_dict(initialWeightsG)
    if initialWeightsD   is not None: learner.discriminator.load_state_dict(initialWeightsD)
    if initialOtherState is not None: learner.loadOtherStateDict(initialOtherState)

    learner.fit(trainingData=dataloader, epochCount=epoch_count, initialEpoch=initial_epoch)

    metricsFile.close()


# ==================================================================================================


@cli.command()
@cloup.option_group(
    "Task settings",
    cloup.option("--dataset-dir",            type=cloup.Path(exists=True, file_okay=False), required=True, help="Dataset directory."),
    cloup.option("--embeddings-path",        type=cloup.Path(exists=True, dir_okay=False),  required=True, help="Path to embeddings file."),
    cloup.option("--output-dir",             type=cloup.Path(),                             required=True, help="Output directory."),
)
@cloup.option_group(
    "Model hyperparameters",
    cloup.option("--scales", "scale_factor", type=str,   default="0.25,0.5,0.75,1.0", help="Scales, from coarse to fine. For options marked with (*), one value can be given for each scale. For options marked with (**), giving one value is equivalent to giving that value for each scale."),
    # TODO: Configurable up/downscale interpolation algorithms / "align_corners" settings?
)
@cloup.option_group(
    "Training hyperparameters",
    cloup.option("--epoch-count",            type=str,   default="100",     help="Epochs to train for. (**)"),
    cloup.option("--batch-size",             type=str,   default="8",       help="Batch size. (**)"),
    cloup.option("--lr",                     type=str,   default="0.0002",  help="Adam learning rate. (**)"),
    cloup.option("--beta1",                  type=str,   default="0.5",     help="Adam beta1 parameter. (**)"),
    cloup.option("--beta2",                  type=str,   default="0.999",   help="Adam beta2 parameter. (**)"),
    cloup.option("--d-down-conv-count",      type=str,   default="4",       help="Number of down-convs in D. (**)"),
)
@cloup.option_group(
    "Initialization",
    cloup.option("--initial-ckpt-dir",       type=str,                      help="Directory containing initial checkpoint. (*)"),
    cloup.option("--initial-g-weight-path",  type=str,                      help="Path to initial generator weights. (*)"),
    cloup.option("--initial-d-weight-path",  type=str,                      help="Path to initial discriminator weights. (*)"),
    cloup.option("--initial-state-path",     type=str,                      help="Path to initial other state. (*)"),
    cloup.option("--initial-epoch",          type=str,                      help="Epoch to start training from. (*)"),
    cloup.option("--continue", "continue_",  is_flag=True,                  help="Continue from last checkpoint in output dir."),
)
@cloup.constraint(cloup.constraints.mutually_exclusive, ["continue_", "initial_ckpt_dir", "initial_g_weight_path", "initial_d_weight_path", "initial_state_path"])
@cloup.constraint(cloup.constraints.mutually_exclusive, ["continue_", "initial_epoch"])
@cloup.option_group(
    "Checkpointing",
    cloup.option("--save-epoch-freq",        type=str,   default=5,         help="Epochs between checkpoints. (**)"),
)
@cloup.option_group(
    "Training process",
    cloup.option("--dataloader-workers",     type=int, default=defaultDataloaderWorkers, help=f"Number of dataloader workers. Defaults to the number of CPUs. Current default: {defaultDataloaderWorkers}.", show_default=False),
    cloup.option("--device", "device_name",  type=str, default=defaultDeviceName, help=f'Device to use (e.g. "cuda:0", "cpu"). Defaults to "cuda" if available, otherwise "cpu". Current default: "{defaultDeviceName}".', show_default=False),
)
@cloup.option_group(
    "Wandb",
    cloup.option("--wandb-entity",           type=str,                      help="Wandb entity name."),
    cloup.option("--wandb-project",          type=str,                      help="Wandb project name."),
    cloup.option("--wandb-log-freq",         type=int,                      help="Batches between wandb logs.")
)
@cloup.constraint(cloup.constraints.If("wandb_project", then=cloup.constraints.require_all, else_=cloup.constraints.accept_none), ["wandb_entity", "wandb_log_freq"])
def vox2vox_multiscale(
    dataset_dir:           str,
    embeddings_path:       str,
    output_dir:            str,
    scale_factor:          str,
    epoch_count:           str,
    batch_size:            str,
    lr:                    str,
    beta1:                 str,
    beta2:                 str,
    d_down_conv_count:     str,
    initial_ckpt_dir:      Optional[str],
    initial_g_weight_path: Optional[str],
    initial_d_weight_path: Optional[str],
    initial_state_path:    Optional[str],
    initial_epoch:         Optional[str],
    continue_:             bool,
    save_epoch_freq:       str,
    dataloader_workers:    int,
    device_name:           Optional[str],
    wandb_entity:          Optional[str],
    wandb_project:         Optional[str],
    wandb_log_freq:        int,
):
    scale_factors = [float(x) for x in scale_factor.split(",")]
    scaleCount = len(scale_factors)

    epoch_counts           = [int(x)   for x in epoch_count          .split(",")]
    batch_sizes            = [int(x)   for x in batch_size           .split(",")]
    lrs                    = [float(x) for x in lr                   .split(",")]
    beta1s                 = [float(x) for x in beta1                .split(",")]
    beta2s                 = [float(x) for x in beta2                .split(",")]
    d_down_conv_counts     = [int(x)   for x in d_down_conv_count    .split(",")]
    initial_ckpt_dirs      = [x        for x in initial_ckpt_dir     .split(",")] if initial_ckpt_dir      is not None else [None] * scaleCount
    initial_g_weight_paths = [x        for x in initial_g_weight_path.split(",")] if initial_g_weight_path is not None else [None] * scaleCount
    initial_d_weight_paths = [x        for x in initial_d_weight_path.split(",")] if initial_d_weight_path is not None else [None] * scaleCount
    initial_state_paths    = [x        for x in initial_state_path   .split(",")] if initial_state_path    is not None else [None] * scaleCount
    initial_epochs         = [int(x)   for x in initial_epoch        .split(",")] if initial_epoch         is not None else [0]    * scaleCount
    save_epoch_freqs       = [int(x)   for x in save_epoch_freq      .split(",")]

    for l in [epoch_counts, batch_sizes, lrs, beta1s, beta2s, d_down_conv_counts, save_epoch_freqs]:
        if len(l) == 1:
            l *= scaleCount
        if len(l) != scaleCount:
            eprint(
                f"{ERROR_PREFIX}The number of values given for a parameter marked with (**) must either\n"
                f"be 1 or equal to the number of scales ({scaleCount}), but {len(l)} were given."
            )
            sys.exit(1)

    for l in [initial_ckpt_dirs, initial_g_weight_paths, initial_d_weight_paths, initial_state_paths, initial_epochs]:
        if len(l) > scaleCount:
            eprint(
                f"{ERROR_PREFIX}The number of values given for a parameter marked with (*) must be at\n"
                f"most equal to the number of scales ({scaleCount}), but {len(l)} were given."
            )
            sys.exit(1)

    if os.path.exists(output_dir) and not continue_:
        eprint(
            f"{ERROR_PREFIX}The output directory {colored(output_dir, **FILENAME_STYLE)} already exists.\n"
             "\n"
             "If the training process crashed or was interrupted, or if want to train for more\n"
             "epochs in the same output directory, you can continue from the last checkpoint\n"
            f"by passing the {colored('--continue', **CLI_OPTION_STYLE)} flag."
        )
        sys.exit(1)

    if continue_ and not os.path.exists(f"{output_dir}"):
        eprint(
            f"{ERROR_PREFIX}{colored('--continue', **CLI_OPTION_STYLE)} was passed, but the output directory\n"
            f"{colored(f'{output_dir}', **FILENAME_STYLE)} does not exist."
        )
        sys.exit(1)

    device = torch.device(device_name)

    if continue_:
        existingScaleCount = len([dirname for dirname in os.listdir(f"{output_dir}") if dirname.startswith("scale-")])
        for scale in range(existingScaleCount):
            lastCheckpoint = max(int(epochDir.split("-")[-1]) for epochDir in os.listdir(f"{output_dir}/scale-{scale}/checkpoints"))
            initial_ckpt_dirs[scale] = f"{output_dir}/scale-{scale}/checkpoints/epoch-{lastCheckpoint}"
            initial_epochs[scale] = lastCheckpoint + 1

    for scale in range(scaleCount):
        if initial_ckpt_dirs[scale] is not None:
            initial_g_weight_paths[scale] = f"{initial_ckpt_dirs[scale]}/generator.pt"
            initial_d_weight_paths[scale] = f"{initial_ckpt_dirs[scale]}/discriminator.pt"
            initial_state_paths[scale]    = f"{initial_ckpt_dirs[scale]}/other-state.pt"

    for initial_scale in range(scaleCount):
        if initial_epochs[initial_scale] != epoch_counts[initial_scale]:
            break

    for scale in range(initial_scale + 1, scaleCount):
        if initial_g_weight_paths[scale] or initial_d_weight_paths[scale] or initial_state_paths[scale]:
            if continue_:
                eprint(
                    f"{WARNING_PREFIX}Initial state was implicity specified for scale {scale} via the --continue\n"
                    f"flag, even though continued training will start at scale {initial_scale}."
                )
            else:
                eprint(
                    f"{WARNING_PREFIX}Initial weights or state were explicitly specified for scale {scale}, even\n"
                    f"though training will start at scale {initial_scale}."
                )
            eprint(
                f"The given state for scale {scale} may have been based on the original state of\n"
                f"scale {initial_scale}, possibly making the given state for scale {scale} an inappropriate\n"
                f"starting point if scale {initial_scale} is trained further."
            )
            if not promptConfirmation("Continue anyway?"):
                sys.exit(0)
            break

    use_wandb = wandb_project is not None
    if use_wandb:
        wandb.login()

    for scale in tqdm(range(initial_scale, scaleCount), desc="Scale", total=scaleCount, initial=initial_scale):
        scale_output_dir = f"{output_dir}/scale-{scale}"

        if use_wandb:
            os.makedirs(scale_output_dir, exist_ok=True)
            modelName = "vox2vox-multiscale-embedding"
            wandb.init(
                reinit=True,
                entity=wandb_entity,
                project=wandb_project,
                group=modelName,
                config={
                    "model":              modelName,
                    "scale":              scale,
                    "scale-factors":      scale_factors,
                    "epoch-counts":       epoch_counts,
                    "batch-sizes":        batch_sizes,
                    "lrs":                lrs,
                    "beta1s":             beta1s,
                    "beta2s":             beta2s,
                    "d-down-conv-counts": d_down_conv_counts,
                    "continue":           continue_,
                },
                dir=f"{output_dir}"
            )

        dataset, embeddingSize = loadDatasetEmbeddings(dataset_dir, embeddings_path)

        prev_gs_weights     = [torch.load(initial_g_weight_paths[prevScale], map_location=device) for prevScale in range(scale)]
        initial_g_weights   = None if initial_g_weight_paths[scale] is None else torch.load(initial_g_weight_paths[scale], map_location=device)
        initial_d_weights   = None if initial_d_weight_paths[scale] is None else torch.load(initial_d_weight_paths[scale], map_location=device)
        initial_other_state = None if initial_state_paths[scale]    is None else torch.load(initial_state_paths[scale],    map_location=device)

        vox2vox_multiscale_train_single_scale(
            dataset =             dataset,
            embeddingSize =       embeddingSize,
            output_dir =          scale_output_dir,
            scale_factors =       scale_factors[:scale+1],
            epoch_count =         epoch_counts[scale],
            batch_size =          batch_sizes[scale],
            lr =                  lrs[scale],
            beta1 =               beta1s[scale],
            beta2 =               beta2s[scale],
            d_down_conv_count =   d_down_conv_counts[scale],
            prev_gs_weights =     prev_gs_weights,
            initial_g_weights =   initial_g_weights,
            initial_d_weights =   initial_d_weights,
            initial_other_state = initial_other_state,
            initial_epoch       = initial_epochs[scale],
            save_epoch_freq     = save_epoch_freqs[scale],
            dataloader_workers =  dataloader_workers,
            device =              device,
            wandb_log_freq =      wandb_log_freq if use_wandb else None,
        )

        if use_wandb:
            wandb.finish()

        # TODO: Perhaps we should keep the generators in memory instead of reloading them for each scale.
        initial_g_weight_paths[scale] = f"{output_dir}/scale-{scale}/checkpoints/epoch-{epoch_counts[scale]-1}/generator.pt"


def vox2vox_multiscale_train_single_scale(
    dataset:             Dataset,
    embeddingSize:       int,
    output_dir:          str,
    scale_factors:       List[float],
    epoch_count:         int,
    batch_size:          int,
    lr:                  float,
    beta1:               float,
    beta2:               float,
    d_down_conv_count:   int,
    prev_gs_weights:     List[Any],
    initial_g_weights:   Optional[Any],
    initial_d_weights:   Optional[Any],
    initial_other_state: Optional[Any],
    initial_epoch:       int,
    save_epoch_freq:     int,
    dataloader_workers:  int,
    device:              torch.device,
    wandb_log_freq:      Optional[int],
):
    assert len(scale_factors) == len(prev_gs_weights) + 1
    currentScale = len(prev_gs_weights)

    use_wandb = wandb_log_freq is not None

    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(f"{output_dir}/metrics.csv"):
        metricsFile = open(f"{output_dir}/metrics.csv", "a", encoding="utf-8", buffering=1)
    else:
        metricsFile = open(f"{output_dir}/metrics.csv", "w", encoding="utf-8", buffering=1)
        metricsFile.write("epoch,batch,loss-g,loss-d,loss-d-real,loss-d-fake\n")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=dataloader_workers)

    class Vox2VoxLearner(Learner):
        def __init__(self):
            super().__init__()

            self.prevGenerators = torch.nn.ModuleList([Vox2voxGenerator(embeddingSize if scale == 0 else embeddingSize * 2, embeddingSize) for scale in range(currentScale)])
            self.generator      = Vox2voxGenerator(embeddingSize if currentScale == 0 else embeddingSize * 2, embeddingSize)
            self.discriminator  = Vox2voxDiscriminator(embeddingSize, downConvCount=d_down_conv_count)

            self.optimizerG = torch.optim.Adam(self.generator    .parameters(), lr=lr, betas=(beta1, beta2))
            self.optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, beta2))

            self.lossGan = LsganLoss()

            if use_wandb:
                wandb.watch(self.generator,     log="all", log_freq=wandb_log_freq)
                wandb.watch(self.discriminator, log="all", log_freq=wandb_log_freq)

        @staticmethod
        def _applyGenerator(generator: torch.nn.Module, downscaledA: torch.Tensor, upscaledPrevGenOut: Optional[torch.Tensor]):
            if upscaledPrevGenOut is None:    # Scale 0
                return generator(downscaledA)
            else:                             # Scale >0
                # First the downscaled original input image, then the upscaled previous output image. The order doesn't matter as long as we're consistent.
                refinement = generator(torch.concatenate([downscaledA, upscaledPrevGenOut], dim=1))
                return upscaledPrevGenOut + refinement

        def trainingStep(self, batch):
            a, bReal = batch

            # Compute size for each scale

            # Since our models are fully convolutional, the size of the input image doesn't actually
            # matter. Hence, we could in theory accept differently-sized images in each batch.
            # For this reason, we recompute the scale sizes every iteration.
            originalSize = a.shape[-3:]
            scaleSizes = [
                [int(scale_factors[scale] * originalSize[axis]) for axis in range(3)]
                for scale in range(currentScale + 1)
            ]

            # Pass input image (`a`) through previous generators

            with torch.no_grad():
                downscaledA        = torch.nn.functional.interpolate(a, size=scaleSizes[0], mode="trilinear", align_corners=False)
                upscaledPrevGenOut = None
                for scale in range(currentScale):
                    # Pass input image through generator `scale`.
                    prevGenOut = self._applyGenerator(self.prevGenerators[scale], downscaledA, upscaledPrevGenOut)
                    # Prepare input image for generator `scale + 1`.
                    downscaledA        = torch.nn.functional.interpolate(a,          size=scaleSizes[scale + 1], mode="trilinear", align_corners=False)
                    upscaledPrevGenOut = torch.nn.functional.interpolate(prevGenOut, size=scaleSizes[scale + 1], mode="trilinear", align_corners=False)

            # Downscale real image

            with torch.no_grad():
                bReal = torch.nn.functional.interpolate(bReal, size=scaleSizes[currentScale], mode="trilinear", align_corners=False)

            # Train generator

            bFake = self._applyGenerator(self.generator, downscaledA, upscaledPrevGenOut)
            predFake = self.discriminator(downscaledA, bFake)
            lossG = self.lossGan(predFake, True)

            self.optimizerG.zero_grad()
            lossG.backward()
            self.optimizerG.step()

            # Train discriminator

            bFake = self._applyGenerator(self.generator, downscaledA, upscaledPrevGenOut).detach() # Don't backpropagate through generator here
            predReal = self.discriminator(downscaledA, bReal)
            predFake = self.discriminator(downscaledA, bFake)
            lossReal = self.lossGan(predReal, True)
            lossFake = self.lossGan(predFake, False)
            lossD = 0.5 * (lossReal + lossFake)

            self.optimizerD.zero_grad()
            lossD.backward()
            self.optimizerD.step()

            metricsFile.write(f"{self.epochIndex},{self.batchIndex},{lossG.item()},{lossD.item()},{lossReal.item()},{lossFake.item()}\n")
            if use_wandb and self.batchIndex % wandb_log_freq == 0:
                wandb.log({
                    "loss-g": lossG.item(),
                    "loss-d": lossD.item(),
                    "loss-d-real": lossReal.item(),
                    "loss-d-fake": lossFake.item(),
                })

        def onEpochEnd(self):
            if self.epochIndex % save_epoch_freq == 0 or self.epochIndex == epoch_count - 1:
                self.saveCheckpoint()

        def saveCheckpoint(self):
            checkpointDir = f"{output_dir}/checkpoints/epoch-{self.epochIndex}"
            os.makedirs(checkpointDir, exist_ok=False)
            torch.save(self.generator.state_dict(),     f"{checkpointDir}/generator.pt")
            torch.save(self.discriminator.state_dict(), f"{checkpointDir}/discriminator.pt")
            otherState = {
                "optimizerG": self.optimizerG.state_dict(),
                "optimizerD": self.optimizerD.state_dict(),
            }
            torch.save(otherState, f"{checkpointDir}/other-state.pt")

        def loadOtherStateDict(self, otherState: Dict):
            self.optimizerG.load_state_dict(otherState["optimizerG"])
            self.optimizerD.load_state_dict(otherState["optimizerD"])

    learner = Vox2VoxLearner().toDevice(device)

    for scale in range(currentScale):
        learner.prevGenerators[scale].load_state_dict(prev_gs_weights[scale])

    if initial_g_weights   is not None: learner.generator    .load_state_dict(initial_g_weights)
    if initial_d_weights   is not None: learner.discriminator.load_state_dict(initial_d_weights)
    if initial_other_state is not None: learner.loadOtherStateDict(initial_other_state)

    learner.fit(trainingData=dataloader, epochCount=epoch_count, initialEpoch=initial_epoch)

    metricsFile.close()


# ==================================================================================================


def main():
    cli() # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    main()
