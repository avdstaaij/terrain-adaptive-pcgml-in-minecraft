#!/usr/bin/env python3

from typing import Callable, Optional
import sys
import os
import shutil
import tempfile

import click
import numpy as np
from termcolor import colored

from lib.util import tqdm, eprint, promptConfirmation, promptInteger, tempArray, tqdmProgressCallback
from lib.constants import ERROR_PREFIX, FILENAME_STYLE
from lib.minecraft_util import AUTOMATIC_BLOCK_STATES, LIQUID_BLOCKS
from lib.palette_tools import CountBlockTypeCountsError, BlockTuple, Palette, countBlockTypeCounts, loadPalette, savePalette, mergePalettes, prunePalette, mapPaletteBlocks, removeStatesFromPalette, sortPalette
from lib.block_embedding import embedBlocksBinary


# ==================================================================================================


COMMAND_STYLE = {"color": "yellow", "attrs": ["bold"]}

TQDM_SETTINGS = {"leave": False}
PROGRESS = tqdmProgressCallback(TQDM_SETTINGS)


def doSingleDatasetOperation(
    ctx:         click.Context,
    inputDir:    str,
    outputDir:   str,
    operation:   Callable[[Optional[np.ndarray], Optional[Palette], Optional[str]], None],
    arrayMode:   str,
    paletteMode: str,
    makeTempDir: bool = False
):
    """Performs an in-place dataset operation.

    <operation> is called with (blocks, palette, tempDir) as arguments.

    inputDir and outputDir may be the same.

    Modes can be:
    ```text
    "i":   modify in-place
    "r":   read
    "rt":  read and move result in from tempDir
    "t":   move result in from tempDir
    "rd":  read and delete/don't copy
    "d":   delete/don't copy
    "":    do not read or modify
    ```
    """

    for mode in [arrayMode, paletteMode]:
        if mode not in ["i", "r", "rt", "t", "rd", "d", ""]:
            raise ValueError(f"Invalid mode: {mode}")

    if not makeTempDir and ("t" in arrayMode or "t" in paletteMode):
        raise ValueError("Cannot use 't' mode without making a temporary directory.")

    if inputDir == "-":
        inputDir = ctx.obj["prev_output_dir"]
    if outputDir == "-":
        outputDir = inputDir

    ctx.obj["prev_output_dir"] = outputDir

    eprint(f"\nRunning {colored(ctx.info_name, **COMMAND_STYLE)} command.")

    if not os.path.isdir(inputDir):
        eprint(ERROR_PREFIX + f"The input directory {colored(inputDir, **FILENAME_STYLE)} does not exist.")
        sys.exit(1)

    if outputDir != inputDir:
        if os.path.isdir(outputDir):
            eprint(ERROR_PREFIX + f"The output directory {colored(outputDir, **FILENAME_STYLE)} already exists.")
            sys.exit(1)
        os.makedirs(outputDir, exist_ok=True)
        if "d" not in arrayMode and "t" not in arrayMode:
            shutil.copy2(f"{inputDir}/blocks.npy", f"{outputDir}/blocks.npy")
        if "d" not in paletteMode and "t" not in paletteMode and paletteMode != "i":
            shutil.copy2(f"{inputDir}/palette.json", f"{outputDir}/palette.json")

    blocks = np.load(f"{outputDir if arrayMode == 'i' else inputDir}/blocks.npy", mmap_mode=("r+" if arrayMode == "i" else "r")) if "r" in arrayMode or arrayMode == "i" else None
    palette = loadPalette(f"{inputDir}/palette.json") if "r" in paletteMode or paletteMode == "i" else None

    if makeTempDir:
        with tempfile.TemporaryDirectory() as tempDir:
            operation(blocks, palette, tempDir)
            if "t" in arrayMode:
                shutil.move(f"{tempDir}/blocks.npy", f"{outputDir}/blocks.npy")
            if "t" in paletteMode:
                shutil.move(f"{tempDir}/palette.json", f"{outputDir}/palette.json")
    else:
        operation(blocks, palette, None)

    if paletteMode == "i":
        savePalette(f"{outputDir}/palette.json", palette)

    if outputDir == inputDir:
        if "d" in arrayMode:
            os.remove(f"{outputDir}/blocks.npy")
        if "d" in paletteMode:
            os.remove(f"{outputDir}/palette.json")


# ==================================================================================================


@click.group(chain=True)
@click.pass_context
def cli(context: click.Context):
    """
    Preprocesses datasets according to the specified commands.

    \b
    Most commands require some input dataset paths and some output dataset paths.
    To easily chain multiple commands, the following features are provided:
    - If an input path is specified as "-", the output path of the previous command
      will be used.
    - If an output path is specified as "-", the input dataset will be modified
      in-place.
    """


__doc__ = cli.__doc__


# ==================================================================================================


@cli.command(short_help="Stack two datasets into a single set of paired examples.")
@click.pass_context
@click.argument("input-1-dir", type=str)
@click.argument("input-2-dir", type=str)
@click.argument("output_dir",  type=str)
def stack(ctx: click.Context, input_1_dir: str, input_2_dir: str, output_dir: str):
    """
    Combines two paired datasets (datasets that each contain one "side" of a set of
    paired examples) into a single dataset of paired examples, and flattens the X
    and Z sample axes into a single one.

    \b
    Input:
    Two datasets, each containing (at least):
    - blocks.npy:   A block array of shape
                    (sample count X, sample count Z, size Y, size X, size Z)
    - palette.json: The block palette that `blocks.npy` indexes.

    \b
    Output:
    One dataset, consisting of:
    - blocks.npy:   A block array of shape
                    (total sample count, 2, size Y, size X, size Z), where:
                    - total sample count = sample count X * sample count Z.
                    - The second axis contains first dataset's sample at index 0
                    and the second dataset's sample at index 1.
    - palette.json: The block palette that `blocks.npy` indexes.
    """

    if input_1_dir == "-":
        input_1_dir = ctx.obj["prev_output_dir"]
    if input_2_dir == "-":
        input_2_dir = ctx.obj["prev_output_dir"]

    ctx.obj["prev_output_dir"] = output_dir

    eprint(f"\nRunning {colored(ctx.info_name, **COMMAND_STYLE)} command.\n")

    if os.path.isdir(output_dir):
        eprint(ERROR_PREFIX + f"The output directory {colored(output_dir, **FILENAME_STYLE)} already exists.")
        sys.exit(1)


    # Load datasets

    eprint("Loading datasets...")

    blocks1: np.memmap = np.load(f"{input_1_dir}/blocks.npy", mmap_mode="r")
    blocks2: np.memmap = np.load(f"{input_2_dir}/blocks.npy", mmap_mode="r")
    palette1 = loadPalette(f"{input_1_dir}/palette.json")
    palette2 = loadPalette(f"{input_2_dir}/palette.json")

    eprint("Done.")


    # Validate shapes

    if blocks1.ndim != 5 or blocks2.ndim != 5:
        eprint(ERROR_PREFIX + "One of the block arrays has an unexpected number of dimensions.")
        sys.exit(1)

    if blocks1.shape[-3:] != blocks2.shape[-3:]:
        eprint(ERROR_PREFIX + "The two datasets have samples of different sizes.")
        sys.exit(1)

    sampleSize = tuple(blocks1.shape[-3:])

    sampleCountDiffX = blocks1.shape[0] - blocks2.shape[0]
    sampleCountDiffZ = blocks1.shape[1] - blocks2.shape[1]

    if (sampleCountDiffX < 0 and sampleCountDiffZ > 0) or (sampleCountDiffX > 0 and sampleCountDiffZ < 0):
        eprint(ERROR_PREFIX + "The two datasets have different sample counts, and neither of them is strictly larger than the other.")
        sys.exit(1)

    if sampleCountDiffX == 0 and sampleCountDiffZ == 0:
        sampleSizesAreEqual = True
        sampleCounts = tuple(blocks1.shape[:2])
    else:
        sampleSizesAreEqual = False
        largeDatasetIndex = 0 if sampleCountDiffX > 0 or sampleCountDiffZ > 0 else 1
        smallDatasetIndex = 1 - largeDatasetIndex
        largeBlockArray = [blocks1, blocks2][largeDatasetIndex]
        smallBlockArray = [blocks1, blocks2][smallDatasetIndex]

        eprint(
            f"\nDataset {largeDatasetIndex+1} contains more samples than dataset {smallDatasetIndex+1}: {tuple(largeBlockArray.shape[:2])} vs {smallBlockArray.shape[:2]}.\n"
            "Please specify an offset (in samples) for the smaller dataset.\n"
        )
        offsetX = promptInteger("Offset X", default=0, min=0, max=largeBlockArray.shape[0]-smallBlockArray.shape[0])
        offsetZ = promptInteger("Offset Z", default=0, min=0, max=largeBlockArray.shape[1]-smallBlockArray.shape[1])
        eprint()

        pruneLargeDatasetPalette = promptConfirmation("Prune unused blocks from the sliced larger dataset's palette?", default=False)
        eprint()

        sampleCounts = tuple(smallBlockArray.shape[:2])


    # Create target array

    eprint("Allocating memmap...")

    os.makedirs(output_dir, exist_ok=True)
    stackedBlockArray: np.memmap = np.lib.format.open_memmap(f"{output_dir}/blocks.npy", mode="w+", shape=(sampleCounts[0]*sampleCounts[1], 2, *sampleSize), dtype=blocks1.dtype)

    eprint("Done.")


    # Stack datasets

    eprint("Copying data...")

    if sampleSizesAreEqual:
        stackedBlockArray[:,0] = blocks1.reshape((-1, *sampleSize))
        stackedBlockArray[:,1] = blocks2.reshape((-1, *sampleSize))
    else:
        stackedBlockArray[:,smallDatasetIndex] = smallBlockArray.reshape((-1, *sampleSize))
        stackedBlockArray[:,largeDatasetIndex] = largeBlockArray[offsetX:offsetX+sampleCounts[0], offsetZ:offsetZ+sampleCounts[1]].reshape((-1, *sampleSize))

    eprint("Done.")

    if not sampleSizesAreEqual and pruneLargeDatasetPalette:
        eprint("Pruning unused blocks from the sliced larger dataset's palette...")
        prunePalette(stackedBlockArray[:,largeDatasetIndex], [palette1, palette2][largeDatasetIndex], progressCallback=PROGRESS)
        eprint("Done.")


    # Merge palettes

    eprint("Merging palettes...")

    mergedPalette = mergePalettes([(stackedBlockArray[:,0], palette1), (stackedBlockArray[:,1], palette2)])

    savePalette(f"{output_dir}/palette.json", mergedPalette)

    eprint("Done.")


# ==================================================================================================


@cli.command("slice", short_help="Slice a segment out of a dataset.")
@click.pass_context
@click.argument("input-dir",  type=str)
@click.argument("output-dir", type=str)
@click.argument("slice_str",   type=str, metavar="slice")
def slice_(ctx: click.Context, input_dir: str, output_dir: str, slice_str: str):
    """
    Slice a dataset's block array.

    \b
    Input:
    - A dataset, containing (at least):
        - blocks.npy:   A block array
        - palette.json: The block palette that `blocks.npy` indexes.
    - A string of comma-separated Python-style slice expressions.

    \b
    Output:
    A dataset, consisting of:
    - blocks.npy:   The specified slice from the input block array.
    - palette.json: The block palette that `blocks.npy` indexes.

    The input and output directories may be the same.
    """

    def operation(blocks: np.ndarray, _, tempDir: str):
        slice_ = tuple(slice(*(None if n == "" else int(n) for n in s.split(":"))) if ":" in s else int(s) for s in slice_str.split(","))

        blockSlice = blocks[slice_]

        eprint(f"New shape: {blockSlice.shape}")

        np.save(f"{tempDir}/blocks.npy", blockSlice)

    doSingleDatasetOperation(ctx, input_dir, output_dir, operation, arrayMode="rt", paletteMode="", makeTempDir=True)


# ==================================================================================================


@cli.command(short_help="Erase invalid samples.")
@click.pass_context
@click.argument("input-dir",  type=str)
@click.argument("output-dir", type=str)
def erase_invalid(ctx: click.Context, input_dir: str, output_dir: str):
    """
    Erases dataset samples that contain a block index that does not appear in the
    dataset's palette.
    This can be useful for sanitizing slightly bugged datasets.

    \b
    Input:
    A dataset, containing (at least):
    - blocks.npy:   A block array. It must start with exactly one flat sample axis,
                    like the output of the stack command.
    - palette.json: The block palette that `blocks.npy` indexes.

    \b
    Output:
    A dataset, consisting of:
    - blocks.npy:   A block array of the same shape, but with a possible shorter
                    first axis.
    - palette.json: The block palette that `blocks.npy` indexes.

    The input and output directories may be the same.
    """

    def operation(blocks: np.ndarray, palette: Palette, tempDir):

        with tempArray(blocks.shape, dtype=bool) as mask:
            np.greater_equal(blocks, len(palette), out=mask)
            toKeep = np.nonzero(~np.any(mask, axis=tuple(range(1, blocks.ndim))))[0]

        eprint(f"Erasing {len(blocks) - len(toKeep)} samples ({len(blocks)} -> {len(toKeep)}).")

        goodSamples: np.memmap = np.lib.format.open_memmap(f"{tempDir}/blocks.npy", mode="w+", shape=tuple([len(toKeep)] + list(blocks.shape)[1:]), dtype=blocks.dtype)

        for newIndex, oldIndex in tqdm(enumerate(toKeep), total=len(toKeep), **TQDM_SETTINGS):
            goodSamples[newIndex] = blocks[oldIndex]

    doSingleDatasetOperation(ctx, input_dir, output_dir, operation, arrayMode="rt", paletteMode="r", makeTempDir=True)


# ==================================================================================================


@cli.command(short_help="Erase pairs of identical samples.")
@click.pass_context
@click.argument("input-dir",  type=str)
@click.argument("output-dir", type=str)
def erase_equal_pairs(ctx: click.Context, input_dir: str, output_dir: str):
    """
    Erases pairs of samples that are identical to each other.
    This is mainly useful for removing samples where a generator failed to execute.

    \b
    Input:
    A dataset, containing (at least):
    - blocks.npy:   A block array. It must start with exactly one flat sample axis and exactly one
                    pair index axis (with length 2), like the output of the stack command.
    - palette.json: The block palette that `blocks.npy` indexes.

    \b
    Output:
    A dataset, consisting of:
    - blocks.npy:   A block array of the same shape, but with a possible shorter
                    first axis.
    - palette.json: The block palette that `blocks.npy` indexes.

    The input and output directories may be the same.
    """

    def operation(blocks: np.ndarray, _, tempDir):

        with tempArray((blocks.shape[0], *blocks.shape[2:]), dtype=bool) as mask:
            np.equal(blocks[:,0], blocks[:,1], out=mask)
            toKeep = np.nonzero(~np.all(mask, axis=tuple(range(1, blocks.ndim-1))))[0]

        eprint(f"Erasing {len(blocks) - len(toKeep)} samples ({len(blocks)} -> {len(toKeep)}).")

        goodSamples: np.memmap = np.lib.format.open_memmap(f"{tempDir}/blocks.npy", mode="w+", shape=tuple([len(toKeep)] + list(blocks.shape)[1:]), dtype=blocks.dtype)

        for newIndex, oldIndex in tqdm(enumerate(toKeep), total=len(toKeep), **TQDM_SETTINGS):
            goodSamples[newIndex] = blocks[oldIndex]

    doSingleDatasetOperation(ctx, input_dir, output_dir, operation, arrayMode="rt", paletteMode="r", makeTempDir=True)


# ==================================================================================================


@cli.command(short_help="Prune unused blocks from a dataset's palette.")
@click.pass_context
@click.argument("input-dir",  type=str)
@click.argument("output-dir", type=str)
def prune_unused(ctx: click.Context, input_dir: str, output_dir: str):
    """
    Prunes a dataset's palette to only contain blocks that are actually used in the
    dataset, and reindexes the dataset's block array accordingly.

    \b
    Input:
    A dataset, containing (at least):
    - blocks.npy:   A block array
    - palette.json: The block palette that `blocks.npy` indexes.

    \b
    Output:
    A dataset, consisting of:
    - blocks.npy:   A block array of the same shape as the input array.
    - palette.json: The block palette that `blocks.npy` indexes (pruned).

    The input and output directories may be the same.
    """

    def operation(blocks: np.ndarray, palette: Palette, _):
        lenBefore = len(palette)

        prunePalette(blocks, palette, progressCallback=PROGRESS)

        eprint(f"Pruned {lenBefore - len(palette)} palette entries ({lenBefore} -> {len(palette)}).")

    doSingleDatasetOperation(ctx, input_dir, output_dir, operation, arrayMode="i", paletteMode="i")


# ==================================================================================================


@cli.command(short_help="Prune automatic block states from a dataset's palette.")
@click.pass_context
@click.argument("input-dir",  type=str)
@click.argument("output-dir", type=str)
def prune_auto_states(ctx: click.Context, input_dir: str, output_dir: str):
    """
    Removes all block states from the dataset's palette that are automatically
    assigned on placement, removes the resulting duplicate palette entries, and
    reindexes the dataset's block array accordingly.

    \b
    Input:
    A dataset, containing (at least):
    - blocks.npy:   A block array
    - palette.json: The block palette that `blocks.npy` indexes.

    \b
    Output:
    A dataset, consisting of:
    - blocks.npy:   A block array of the same shape as the input array.
    - palette.json: The block palette that `blocks.npy` indexes (pruned).

    The input and output directories may be the same.
    """

    def operation(blocks: np.ndarray, palette: Palette, _):
        lenBefore = len(palette)

        removeStatesFromPalette(blocks, palette, AUTOMATIC_BLOCK_STATES, progressCallback=PROGRESS)

        eprint(f"Pruned {lenBefore - len(palette)} palette entries ({lenBefore} -> {len(palette)}).")

    doSingleDatasetOperation(ctx, input_dir, output_dir, operation, arrayMode="i", paletteMode="i",)


# ==================================================================================================


@cli.command(short_help="Replace flowing liquid blocks with air.")
@click.pass_context
@click.argument("input-dir",  type=str)
@click.argument("output-dir", type=str)
def remove_flow(ctx: click.Context, input_dir: str, output_dir: str):
    """
    Replaces all flowing liquid blocks with air.
    Note that this will never place cave_air, even if the flowing liquid is in a cave.

    \b
    Input:
    A dataset, containing (at least):
    - blocks.npy:   A block array
    - palette.json: The block palette that `blocks.npy` indexes.

    \b
    Output:
    A dataset, consisting of:
    - blocks.npy:   The block array, with flowing liquid replaced with air.
    - palette.json: The block palette that `blocks.npy` indexes.

    The input and output directories may be the same.
    """

    def operation(blocks: np.ndarray, palette: Palette, _):
        lenBefore = len(palette)

        def mapFunc(blockTuple: BlockTuple):
            if blockTuple[0] not in LIQUID_BLOCKS or "level" not in blockTuple[1]:
                return blockTuple
            if blockTuple[1]["level"] == "0":
                del blockTuple[1]["level"]
                return blockTuple
            return ("air", {})

        mapPaletteBlocks(blocks, palette, mapFunc, progressCallback=PROGRESS)

        eprint(f"Reduced palette size by {lenBefore - len(palette)} ({lenBefore} -> {len(palette)}).")

    doSingleDatasetOperation(ctx, input_dir, output_dir, operation, arrayMode="i", paletteMode="i",)


# ==================================================================================================


@cli.command(short_help="Re-index a dataset's blocks to sort by frequency.")
@click.pass_context
@click.argument("input-dir",  type=str)
@click.argument("output-dir", type=str)
def sort_by_freq(ctx: click.Context, input_dir: str, output_dir: str):
    """
    Reindexes a dataset's blocks such that they are sorted by their
    occurence frequency (from common to rare).

    \b
    Input:
    A dataset, containing (at least):
    - blocks.npy:   A block array
    - palette.json: The block palette that `blocks.npy` indexes.

    \b
    Output:
    A dataset, consisting of:
    - blocks.npy:   The reindexed block array.
    - palette.json: The block palette that `blocks.npy` indexes.

    The input and output directories may be the same.
    """

    def operation(blocks: np.ndarray, palette: Palette, _):
        eprint("Counting block type frequencies...")
        try:
            blockTypeCounts = countBlockTypeCounts(blocks, len(palette), progressCallback=PROGRESS)
        except CountBlockTypeCountsError as e:
            eprint(ERROR_PREFIX + f"The block index array contains the index {e.badIndex}, but the palette contains only {len(palette)} blocks. (Max index should be {len(palette)-1}.)")
            sys.exit(1)
        eprint("Done.")

        eprint("Sorting and reindexing...")
        sortPalette(blocks, palette, lambda blockTypeIndex: -blockTypeCounts[blockTypeIndex], progressCallback=PROGRESS)
        eprint("Done.")

    doSingleDatasetOperation(ctx, input_dir, output_dir, operation, arrayMode="i", paletteMode="i")


# ==================================================================================================


@cli.command(short_help="Embed blocks to a binary air/non-air mask.")
@click.pass_context
@click.argument("input-dir",  type=str)
@click.argument("output-dir", type=str)
def embed_binary(ctx: click.Context, input_dir: str, output_dir: str):
    """
    Embed a dataset's block array by converting air blocks to 0 and all other
    blocks to 1.

    \b
    Input:
    - A dataset, containing (at least):
        - blocks.npy:   A block array
        - palette.json: The block palette that `blocks.npy` indexes.
    - A string of comma-separated Python-style slice expressions.

    \b
    Output:
    A dataset, consisting of:
    - blocks.npy:   The binary-embedded block array

    The input and output directories may be the same. In this case, the input
    palette is deleted.
    """

    def operation(blocks: np.ndarray, palette: Palette, tempDir: str):
        embedding: np.memmap = np.lib.format.open_memmap(f"{tempDir}/blocks.npy", mode="w+", shape=blocks.shape, dtype=bool)

        embedBlocksBinary(blocks, palette, out=embedding)

        embedding.flush()
        del embedding

    doSingleDatasetOperation(ctx, input_dir, output_dir, operation, arrayMode="rt", paletteMode="rd", makeTempDir=True)


# ==================================================================================================


def main():
    cli(obj={}) # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    main()
