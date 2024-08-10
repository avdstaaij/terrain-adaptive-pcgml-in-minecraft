from typing import Any, Callable, Container, List, Dict, Optional, Tuple, Sequence
from copy import copy, deepcopy
import json

import numpy as np

from lib.util import ProgressCallback, invertPermutation, tempArray


BlockTuple = Tuple[str, Dict[str, str]]
"""(basename, states)"""

Palette = List[BlockTuple]


def loadPalette(filename: str):
    with open(filename, "r", encoding="utf-8") as f:
        rawPalette = json.load(f)
    palette: Palette = [tuple(block) for block in rawPalette] # Convert block lists to block tuples
    return palette


def savePalette(filename: str, palette: Palette):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(palette, f, separators=(",", ":"))


def blockTupleToString(blockTuple: BlockTuple):
    name =  blockTuple[0]
    blockDataStr = "[" + ", ".join(f"{key}={value}" for key, value in blockTuple[1].items()) + "]" if len(blockTuple[1]) > 0 else ""
    return name + blockDataStr


def mergePalettes(datasets: Sequence[Tuple[np.ndarray, Palette]]):
    """Merges the palettes of the given datasets.

    Each dataset should be given as (block array, palette)-tuple, with the block array containing
    indices into the palette.

    Returns the merged palette and modifies the given block arrays in-place to contain indices into
    the merged palette.
    """

    if len(datasets) == 0:
        raise ValueError("At least one dataset should be given.")

    mergedPalette = copy(datasets[0][1])

    for oldBlockArray, palette in datasets[1:]:
        mergedPalettePrev = copy(mergedPalette)

        with tempArray(oldBlockArray.shape, dtype=oldBlockArray.dtype) as newBlockArray:
            with tempArray(oldBlockArray.shape, dtype=bool) as mask:
                newBlockArray[:] = oldBlockArray

                for oldIndex, blockTuple in enumerate(palette):
                    try:
                        newIndex = mergedPalettePrev.index(blockTuple)
                    except ValueError:
                        newIndex = len(mergedPalette)
                        mergedPalette.append(blockTuple)

                    if newIndex != oldIndex:
                        np.equal(oldBlockArray, oldIndex, out=mask)
                        newBlockArray[mask] = newIndex

                oldBlockArray[:] = newBlockArray

    return mergedPalette


def prunePalette(blocks: np.ndarray, palette: Palette, progressCallback: Optional[ProgressCallback] = None):
    """
    Removes palette entries that do not occur in <blocks>.\n
    Modifies <blocks> and <palette> in-place.\n
    """

    curr = 0
    last = len(palette) - 1

    if progressCallback is not None: progressCallback = progressCallback(len(palette))

    with tempArray(blocks.shape, dtype=bool) as mask:
        while curr <= last:
            if curr not in blocks:
                lastBlockTuple = palette.pop(last)
                if curr != last:
                    palette[curr] = lastBlockTuple
                    np.equal(blocks, last, out=mask)
                    blocks[mask] = curr
                last -= 1
            else:
                curr += 1
            if progressCallback is not None: progressCallback(1)


def mapPaletteBlocks(blocks: np.ndarray, palette: Palette, mapFunc: Callable[[BlockTuple], BlockTuple], progressCallback: Optional[ProgressCallback] = None):
    """
    Maps blocks from <palette> according to <mapFunc>, removes any resulting duplicate entries, and
    reindexes <blocks> accordingly.\n
    The <mapFunc> is allowed to modify the passed block in-place (though it must still return the
    result).\n
    Modifies <blocks> and <palette> in-place.
    """

    curr = 0
    last = len(palette) - 1

    if progressCallback is not None: progressCallback = progressCallback(len(palette))

    with tempArray(blocks.shape, dtype=bool) as mask:
        while curr <= last:
            blockTuple = palette[curr]

            newBlockTuple = mapFunc(deepcopy(blockTuple))

            if newBlockTuple != blockTuple:
                try:
                    palette[curr] = None
                    newIndex = palette.index(newBlockTuple)
                except ValueError:
                    palette[curr] = newBlockTuple
                else:
                    np.equal(blocks, curr, out=mask)
                    blocks[mask] = newIndex

                    lastBlockTuple = palette.pop(last)
                    if curr != last:
                        palette[curr] = lastBlockTuple
                        np.equal(blocks, last, out=mask)
                        blocks[mask] = curr
                        last -= 1
                        curr -= 1

            curr += 1
            if progressCallback is not None: progressCallback(1)


def removeStatesFromPalette(blocks: np.ndarray, palette: Palette, statesToRemove: Container[str], progressCallback: Optional[ProgressCallback] = None):
    """
    Removes all block states from <palette> that are automatically assigned on
    placement and removes the resulting duplicate palette entries.\n
    Modifies <blocks> and <palette> in-place.
    """
    def mapFunc(blockTuple: BlockTuple):
        for state in statesToRemove:
            if state in blockTuple[1]:
                del blockTuple[1][state]
        return blockTuple

    mapPaletteBlocks(blocks, palette, mapFunc, progressCallback)



def reindexPalette(blocks: np.ndarray, palette: Palette, permutation: Sequence[int], progressCallback: Optional[ProgressCallback] = None):
    """
    Reindexes the palette according to the given permutation.\n
    <permutation> should "map" new_index->old_index, not the other way around!\n
    Modifies <blocks> and <palette> in-place.
    """

    assert len(permutation) == len(palette)

    permutation = np.array(permutation, dtype=blocks.dtype)
    invPermutation = invertPermutation(permutation)

    if progressCallback is not None: progressCallback = progressCallback(np.prod(blocks.shape[:-3]))

    for metaPos in np.ndindex(blocks.shape[:-3]):
        newSample = invPermutation[blocks[metaPos]]
        blocks[metaPos] = newSample
        if progressCallback is not None: progressCallback(1)

    newPalette = [palette[i] for i in permutation]
    palette[:] = newPalette


def sortPalette(blocks: np.ndarray, palette: Palette, key: Callable[[int], Any], progressCallback: Optional[ProgressCallback] = None):
    """
    Sorts the palette according to the given key, and reindexes the block array accordingly.\n
    Modifies <blocks> and <palette> in-place.
    """

    permutation = np.argsort([key(i) for i in range(len(palette))])
    reindexPalette(blocks, palette, permutation, progressCallback=progressCallback)


class CountBlockTypeCountsError(Exception):
    def __init__(self, badIndex: int):
        super().__init__(f"The block index array contains the index {badIndex}, which exceeds the specified amount of block types.")
        self.badIndex = badIndex

def countBlockTypeCounts(blocks: np.ndarray, blockTypeCount: int, progressCallback: Optional[ProgressCallback] = None):
    """
    Returns the number of blocks of each type in the given block array.\n
    The number of block types must not exceed <blockTypeCount>.
    """

    metaShape = blocks.shape[:-3]
    metaSize  = int(np.prod(metaShape))

    if progressCallback is not None: progressCallback = progressCallback(metaSize)

    # This wacky counting method seems to be the fastest one that does not exceed memory.
    partialBlockTypeCounts = [None] * metaSize
    for i in range(metaSize):
        sample = blocks.view()[np.unravel_index(i, metaShape)]
        sample.shape = (-1,)
        partialBlockTypeCounts[i] = np.bincount(sample, minlength=blockTypeCount)
        if len(partialBlockTypeCounts[i]) != blockTypeCount:
            raise CountBlockTypeCountsError(len(partialBlockTypeCounts[i]-1))
        if progressCallback is not None: progressCallback(1)
    blockTypeCounts = np.sum(partialBlockTypeCounts, axis=0)

    return blockTypeCounts
