from typing import Dict, FrozenSet, List, Tuple
from contextlib import contextmanager

import numpy as np

from .vector_tools import Rect, addY, loop2D
from .amulet_wrapper import amulet
from .constants import MINECRAFT_VERSION


AMULET_MINECRAFT_VERSION = ("java", tuple(int(x) for x in MINECRAFT_VERSION.split(".")))


BlockTuple = Tuple[str, Dict[str, str]]
"""(basename, states)"""

HashableBlockTuple = Tuple[str, FrozenSet[Tuple[str, str]]]
"""Hashable version of BlockTuple, with <states> as a frozenset"""


@contextmanager
def loadLevel(path: str):
    """
    Wrapper around amulet.load_level() that closes the world after use.\n
    If the world would not be closed, the next amulet.load_level() call would get stuck trying to
    aquire a lock on session.lock.
    """
    world = amulet.load_level(path)
    try:
        yield world
    finally:
        world.close()


def extractChunkBox(world: amulet.api.level.World, chunkRect: Rect, yBegin: int, yEnd: int, outPaletteIndices: np.ndarray, inOutPalette: List[BlockTuple], inOutPaletteLookup: Dict[HashableBlockTuple, int]):
    """
    Extracts the blocks from the specified world area into
    <outPaletteIndices>, <inOutPalette> and <inOutPaletteLookup>.

    The blocks are extracted in a palettized manner: outPaletteIndices should be a 3D numpy array
    with dtype=np.uint16, and will be filled with indices into <inOutPalette>. Blocks are stored in
    YXZ order.
    <inOutPalette> will describe the actual blocks, and <inOutPaletteLookup> will map block
    descriptions to their palette index.

    If <inOutPalette> and <inOutPaletteLookup> aready contain blocks, new blocks will be appended to
    them. <inOutPaletteLookup> must match <inOutPalette>.
    """

    for chunkPos in chunkRect.inner:
        chunk = world.get_chunk(chunkPos.x, chunkPos.y, "minecraft:overworld")

        for y in range(yBegin, yEnd):
            for offset in loop2D((16,16)):
                paletteIndex = chunk.blocks[offset.x, y, offset.y]

                # Ignore block entity NBT
                # TODO: directly extract numpy arrays from chunk.blocks for even better performance?
                universalBlock = chunk.block_palette[paletteIndex]
                block, _, _ = world.translation_manager.get_version(*AMULET_MINECRAFT_VERSION).block.from_universal(universalBlock)

                blockPos = addY((chunkPos - chunkRect.offset) * 16 + offset, y - yBegin)

                # Construct block tuple
                blockTuple = (block.base_name, {key: val.py_str for key, val in block.properties.items()})
                hashableBlockTuple = (blockTuple[0], frozenset(blockTuple[1].items()))

                index = inOutPaletteLookup.get(hashableBlockTuple)

                if index is None:
                    index = len(inOutPalette)
                    inOutPalette.append(blockTuple)
                    inOutPaletteLookup[hashableBlockTuple] = index

                outPaletteIndices[blockPos.y, blockPos.x, blockPos.z] = index # YXZ order


def reconstructPaletteLookup(palette: List[BlockTuple]):
    """Reconstructs a palette lookup dict from a palette."""
    return { (blockTuple[0], frozenset(blockTuple[1].items())): i for i, blockTuple in enumerate(palette) }
