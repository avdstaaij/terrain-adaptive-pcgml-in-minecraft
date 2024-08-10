from typing import Dict, Optional, Tuple

import numpy as np
from glm import ivec3

from .util import tempArray
from .palette_tools import Palette


# ==================================================================================================
# Binary embedding
# ==================================================================================================


def embedBlocksBinary(blocks: np.ndarray, palette: Palette, out: Optional[np.ndarray] = None):
    """Converts air blocks to 0 and all other blocks to 1.

    If given, <out> should be a binary array with the same shape as <blocks>.
    """

    if out is None:
        out = np.ones(blocks.shape, dtype=bool)
    else:
        out[:] = True

    try:
        airIndex = palette.index(("air", {}))
    except ValueError:
        airIndex = None
    try:
        caveAirIndex = palette.index(("cave_air", {}))
    except ValueError:
        caveAirIndex = None

    with tempArray(blocks.shape, dtype=bool) as mask:
        if airIndex is not None:
            np.equal(blocks, airIndex, out=mask)
            out[mask] = False
        if caveAirIndex is not None:
            np.equal(blocks, caveAirIndex, out=mask)
            out[mask] = False

    return out


# ==================================================================================================
# Block state embedding (old code)
# ==================================================================================================

__FACING_TO_VECTOR = {
    "north": ivec3( 0, 0,-1),
    "south": ivec3( 0, 0, 1),
    "down":  ivec3( 0,-1, 0),
    "up":    ivec3( 0, 1, 0),
    "west":  ivec3(-1, 0, 0),
    "east":  ivec3( 1, 0, 0),
}
__AXIS_TO_VECTOR = {
    "x": ivec3(1,0,0),
    "y": ivec3(0,1,0),
    "z": ivec3(0,0,1),
}
def blockStatesToDirectionVector(blockStates: Dict[str,str]):
    """Returns the direction vector for the given block states.

    The following block states are considered:
    - facing
    - axis

    If none of these block states are present, returns ivec3(0,0,0).
    """

    facing = blockStates.get("facing", None)
    if facing is not None:
        return __FACING_TO_VECTOR[facing]

    axis = blockStates.get("axis", None)
    if axis is not None:
        return __AXIS_TO_VECTOR[axis]

    return ivec3(0,0,0)


def directionVectorToFacing(vec: ivec3):
    axis = np.argmax(np.abs(vec))
    if axis == 0:
        return "east"  if vec.x > 0 else "west"
    elif axis == 1:
        return "up"    if vec.y > 0 else "down"
    elif axis == 2:
        return "south" if vec.z > 0 else "north"


__AXES = ["x", "y", "z"]
def directionVectorToAxis(vec: ivec3):
    axis = np.argmax(np.abs(vec))
    return __AXES[axis]


def blockStatesToDirectionVectors(blockStates: Dict[Tuple[int,int,int], Dict[str,str]], shape: Tuple[int,int,int]):
    directionVectors = np.zeros((shape[0], shape[1], shape[2], 3), dtype=np.int8)
    for pos, states in blockStates.items():
        directionVectors[pos] = blockStatesToDirectionVector(states)
    return directionVectors
