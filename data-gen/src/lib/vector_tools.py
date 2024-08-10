"""
Shortened version of
https://github.com/avdstaaij/gdpc/blob/v6.0.2/gdpc/vector_tools.py
"""


from typing import Sequence, Any, Iterable, Optional, Union
from abc import ABC
from dataclasses import dataclass

import numpy as np
from glm import ivec2, ivec3

from .util import nonZeroSign


# ==================================================================================================
# VecLike ABCs
# ==================================================================================================


class Vec2iLike(ABC, Sequence[int]):
    """An abstract base class. A class is a Vec2i if it acts like a sequence of two ints."""
    @classmethod
    def __subclasshook__(cls, C):
        try:
            return len(C) == 2 and all(isinstance(C[i], int) for i in range(2))
        except TypeError:
            return False

class Vec3iLike(ABC, Sequence[int]):
    """An abstract base class. A class is a Vec3i if it acts like a sequence of three ints."""
    @classmethod
    def __subclasshook__(cls, C):
        try:
            return len(C) == 3 and all(isinstance(C[i], int) for i in range(3))
        except TypeError:
            return False

class Vec2bLike(ABC, Sequence[bool]):
    """An abstract base class. A class is a Vec2b if it acts like a sequence of two bools."""
    @classmethod
    def __subclasshook__(cls, C):
        try:
            return len(C) == 2 and all(isinstance(C[i], bool) for i in range(2))
        except TypeError:
            return False

class Vec3bLike(ABC, Sequence[bool]):
    """An abstract base class. A class is a Vec3b if it acts like a sequence of three bools."""
    @classmethod
    def __subclasshook__(cls, C):
        try:
            return len(C) == 3 and all(isinstance(C[i], bool) for i in range(3))
        except TypeError:
            return False


# ==================================================================================================
# Constants
# ==================================================================================================


UP    = ivec3( 0, 1, 0)
DOWN  = ivec3( 0,-1, 0)
EAST  = ivec3( 1, 0, 0)
WEST  = ivec3(-1, 0, 0)
NORTH = ivec3( 0, 0,-1)
SOUTH = ivec3( 0, 0, 1)
X     = ivec3( 1, 0, 0)
Y     = ivec3( 0, 1, 0)
Z     = ivec3( 0, 0, 1)
XY    = ivec3( 1, 1, 0)
XZ    = ivec3( 1, 0, 1)
YZ    = ivec3( 0, 1, 1)
XYZ   = ivec3( 1, 1, 1)

DIAGONALS_2D = (
    ivec2( 1,  1),
    ivec2( 1, -1),
    ivec2(-1,  1),
    ivec2(-1, -1),
)

DIAGONALS_3D = (
    ivec3( 1,  1,  0),
    ivec3( 1,  0,  1),
    ivec3( 0,  1,  1),
    ivec3( 1, -1,  0),
    ivec3( 1,  0, -1),
    ivec3( 0,  1, -1),
    ivec3(-1,  1,  0),
    ivec3(-1,  0,  1),
    ivec3( 0, -1,  1),
    ivec3(-1, -1,  0),
    ivec3(-1,  0, -1),
    ivec3( 0, -1, -1),
    ivec3( 1,  1,  1),
    ivec3( 1,  1, -1),
    ivec3( 1, -1,  1),
    ivec3(-1,  1,  1),
    ivec3( 1, -1, -1),
    ivec3(-1, -1,  1),
    ivec3(-1,  1, -1),
    ivec3(-1, -1, -1),
)


# ==================================================================================================
# General
# ==================================================================================================


def dropDimension(vec: Vec2iLike, dimension: int):
    """Returns <vec> without its <dimension>-th component"""
    if dimension == 0: return ivec2(vec[1], vec[2])
    if dimension == 1: return ivec2(vec[0], vec[2])
    if dimension == 2: return ivec2(vec[0], vec[1])
    raise ValueError(f'Invalid dimension "{dimension}"')


def addDimension(vec: Vec2iLike, dimension: int, value=0):
    """Inserts <value> into <vec> at <dimension> and returns the resulting 3D vector"""
    l = list(vec)
    return ivec3(*l[:dimension], value, *l[dimension:])


def dropY(vec: Vec2iLike):
    """Returns [vec] without its y-component (i.e., projected on the XZ-plane)"""
    return ivec2(vec[0], vec[2])


def addY(vec: Vec2iLike, y=0):
    """Returns a 3D vector (vec[0], y, vec[1])"""
    return ivec3(vec[0], y, vec[1])


def setY(vec: Vec2iLike, y=0):
    """Returns [vec] with its y-component set to [y]"""
    return ivec3(vec[0], y, vec[1])


def trueMod2D(vec: Vec2iLike, modulus):
    """Returns <v> modulo <modulus>.\n
    Negative numbers are handled just like Python's built-in integer modulo."""
    return ivec2(vec[0] % modulus, vec[1] % modulus)

def trueMod3D(vec: Vec3iLike, modulus):
    """Returns <v> modulo <modulus>.\n
    Negative numbers are handled just like Python's built-in integer modulo."""
    return ivec3(vec[0] % modulus, vec[1] % modulus, vec[2] % modulus)


def orderedCorners2D(corner1: Vec2iLike, corner2: Vec2iLike):
    """Returns two corners of the rectangle defined by <corner1> and <corner2>, such that the first
    corner is smaller than the second corner in each axis"""
    return (
        ivec2(
            corner1[0] if corner1[0] <= corner2[0] else corner2[0],
            corner1[1] if corner1[1] <= corner2[1] else corner2[1],
        ),
        ivec2(
            corner1[0] if corner1[0] > corner2[0] else corner2[0],
            corner1[1] if corner1[1] > corner2[1] else corner2[1],
        )
    )

def orderedCorners3D(corner1: Vec3iLike, corner2: Vec3iLike):
    """Returns two corners of the box defined by <corner1> and <corner2>, such that the first
    corner is smaller than the second corner in each axis"""
    return (
        ivec3(
            corner1[0] if corner1[0] <= corner2[0] else corner2[0],
            corner1[1] if corner1[1] <= corner2[1] else corner2[1],
            corner1[2] if corner1[2] <= corner2[2] else corner2[2],
        ),
        ivec3(
            corner1[0] if corner1[0] > corner2[0] else corner2[0],
            corner1[1] if corner1[1] > corner2[1] else corner2[1],
            corner1[2] if corner1[2] > corner2[2] else corner2[2],
        )
    )


# ==================================================================================================
# Rect and Box
# ==================================================================================================


# TODO: If someone knows how to fix the duplication in Rect and Box, please do tell.


@dataclass
class Rect:
    """A rectangle, defined by an offset and a size"""

    _offset: ivec2
    _size:   ivec2

    def __init__(self, offset: Vec2iLike = (0,0), size: Vec2iLike = (0,0)):
        self._offset = ivec2(*offset)
        self._size   = ivec2(*size)

    def __repr__(self):
        return f"Rect({self._offset}, {self._size})"

    @property
    def offset(self):
        """This Rect's offset"""
        return self._offset

    @offset.setter
    def offset(self, value: Vec2iLike):
        self._offset = ivec2(*value)

    @property
    def size(self):
        """This Rect's size"""
        return self._size

    @size.setter
    def size(self, value: Vec2iLike):
        self._size = ivec2(*value)

    @property
    def begin(self):
        """Equivalent to self.offset. Setting will modify self.offset."""
        return self._offset

    @begin.setter
    def begin(self, value: Vec2iLike):
        self._offset = ivec2(*value)

    @property
    def end(self):
        """Equivalent to self.offset + self.size. Setting will modify self.size."""
        return self.begin + self._size

    @end.setter
    def end(self, value: Vec2iLike):
        self._size = ivec2(*value) - self.begin

    @property
    def last(self):
        """Equivalent to self.offset + self.size - 1. Setting will modify self.size."""
        return self._offset + self._size - 1

    @last.setter
    def last(self, value: Vec2iLike):
        self._size = ivec2(*value) - self._offset + 1

    @property
    def middle(self):
        """This Rect's middle point, rounded down"""
        return self._offset + self._size // 2

    @property
    def center(self):
        """Equivalent to .middle"""
        return self.middle

    @property
    def inner(self):
        """Yields all points contained in this Rect"""
        return (
            ivec2(x, y)
            for x in range(self.begin.x, self.end.x)
            for y in range(self.begin.y, self.end.y)
        )

    @property
    def area(self):
        """This Rect's surface area"""
        return self._size.x*self._size.y

    def contains(self, vec: Vec2iLike):
        """Returns whether this Rect contains [vec]"""
        return (
            self.begin.x <= vec[0] < self.end.x and
            self.begin.y <= vec[1] < self.end.y
        )

    def collides(self, other: 'Rect'):
        """Returns whether this Rect and [other] have any overlap"""
        return (
            self.begin.x <= other.end  .x and
            self.end  .x >= other.begin.x and
            self.begin.y <= other.end  .y and
            self.end  .y >= other.begin.y
        )

    def translated(self, translation: Union[Vec2iLike, int]):
        """Returns a copy of this Rect, translated by [translation]"""
        return Rect(self._offset + ivec2(*translation), self._size)

    def dilate(self, dilation: int = 1):
        """Morphologically dilates this rect by [dilation]"""
        self._offset  -= dilation
        self._size    += dilation*2

    def dilated(self, dilation: int = 1):
        """Returns a copy of this Rect, morphologically dilated by [dilation]"""
        return Rect(self._offset - dilation, self._size + dilation*2)

    def erode(self, erosion: int = 1):
        """Morphologically erodes this rect by [erosion]"""
        self.dilate(-erosion)

    def eroded(self, erosion: int = 1):
        """Returns a copy of this Rect, morphologically eroded by [erosion]"""
        return self.dilated(-erosion)

    def centeredSubRectOffset(self, size: Vec2iLike):
        """Returns an offset such that Rect(offset, [size]).middle == self.middle"""
        difference = self._size - ivec2(*size)
        return self._offset + difference/2

    def centeredSubRect(self, size: Vec2iLike):
        """Returns a rect of size [size] with the same middle as this rect"""
        return Rect(self.centeredSubRectOffset(size), size)

    @staticmethod
    def between(cornerA: Vec2iLike, cornerB: Vec2iLike):
        """Returns the Rect between [cornerA] and [cornerB] (inclusive),
        which may be any opposing corners."""
        first, last = orderedCorners2D(cornerA, cornerB)
        return Rect(first, (last - first) + 1)

    @staticmethod
    def bounding(points: Iterable[Vec2iLike]):
        """Returns the smallest Rect containing all [points]"""
        pointArray = np.array(points)
        minPoint = np.min(pointArray, axis=0)
        maxPoint = np.max(pointArray, axis=0)
        return Rect(minPoint, maxPoint - minPoint + 1)

    def toBox(self, offsetY = 0, sizeY = 0):
        """Returns a corresponding Box"""
        return Box(addY(self.offset, offsetY), addY(self._size, sizeY))

    @property
    def outline(self):
        """Yields this Rect's outline points"""
        # It's surprisingly difficult to get this right without duplicates. (Think of the corners!)
        first = self.begin
        last  = self.end - 1
        yield from loop2D(ivec2(first.x, first.y), ivec2(last.x  -1, first.y   ) + 1)
        yield from loop2D(ivec2(last.x,  first.y), ivec2(last.x,     last.y  -1) + 1)
        yield from loop2D(ivec2(last.x,  last.y),  ivec2(first.x +1, last.y    ) - 1)
        yield from loop2D(ivec2(first.x, last.y),  ivec2(first.x,    first.y +1) - 1)


@dataclass()
class Box:
    """A box, defined by an offset and a size"""

    _offset: ivec3
    _size:   ivec3

    def __init__(self, offset: Vec3iLike = (0,0,0), size: Vec3iLike = (0,0,0)):
        self._offset = ivec3(*offset)
        self._size   = ivec3(*size)

    def __repr__(self):
        return f"Box({self._offset}, {self._size})"

    @property
    def offset(self):
        """This Box's offset"""
        return self._offset

    @offset.setter
    def offset(self, value: Vec3iLike):
        self._offset = ivec3(*value)

    @property
    def size(self):
        """This Box's size"""
        return self._size

    @size.setter
    def size(self, value: Vec3iLike):
        self._size = ivec3(*value)

    @property
    def begin(self):
        """Equivalent to self.offset. Setting will modify self.offset."""
        return self._offset

    @begin.setter
    def begin(self, value: Vec3iLike):
        self._offset = ivec3(*value)

    @property
    def end(self):
        """Equivalent to self.offset + self.size. Setting will modify self.size."""
        return self.begin + self._size

    @end.setter
    def end(self, value: Vec3iLike):
        self._size = ivec3(*value) - self.begin

    @property
    def last(self):
        """Equivalent to self.offset + self.size - 1. Setting will modify self.size."""
        return self._offset + self._size - 1

    @last.setter
    def last(self, value: Vec3iLike):
        self._size = ivec3(*value) - self._offset + 1

    @property
    def middle(self):
        """This Box's middle point, rounded down"""
        return (self.begin + self._size) // 2

    @property
    def center(self):
        """Equivalent to .middle"""
        return self.middle

    @property
    def inner(self):
        """Yields all points contained in this Box"""
        return (
            ivec3(x, y, z)
            for x in range(self.begin.x, self.end.x)
            for y in range(self.begin.y, self.end.y)
            for z in range(self.begin.z, self.end.z)
        )

    @property
    def volume(self):
        """This Box's volume"""
        return self._size.x*self._size.y*self._size.z

    def contains(self, vec: Vec3iLike):
        """Returns whether this Box contains [vec]"""
        return (
            self.begin.x <= vec[0] < self.end.x and
            self.begin.y <= vec[1] < self.end.y and
            self.begin.z <= vec[2] < self.end.z
        )

    def collides(self, other: 'Box'):
        """Returns whether this Box and [other] have any overlap"""
        return (
            self.begin.x <= other.end  .x and
            self.end  .x >= other.begin.x and
            self.begin.y <= other.end  .y and
            self.end  .y >= other.begin.y and
            self.begin.z <= other.end  .z and
            self.end  .z >= other.begin.z
        )

    def translated(self, translation: Union[Vec3iLike, int]):
        """Returns a copy of this Box, translated by [translation]"""
        return Box(self._offset + ivec3(*translation), self._size)

    def dilate(self, dilation: int = 1):
        """Morphologically dilates this box by [dilation]"""
        self._offset -= dilation
        self._size   += dilation*2

    def dilated(self, dilation: int = 1):
        """Returns a copy of this Box, morphologically dilated by [dilation]"""
        return Box(self._offset - dilation, self._size + dilation*2)

    def erode(self, erosion: int = 1):
        """Morphologically erodes this box by [erosion]"""
        self.dilate(-erosion)

    def eroded(self, erosion: int = 1):
        """Returns a copy of this Box, morphologically eroded by [erosion]"""
        return self.dilated(-erosion)

    def centeredSubBoxOffset(self, size: Vec3iLike):
        """Returns an offset such that Box(offset, [size]).middle == self.middle"""
        difference = self._size - ivec3(*size)
        return self._offset + difference/2

    def centeredSubBox(self, size: Vec3iLike):
        """Returns an box of size [size] with the same middle as this box"""
        return Box(self.centeredSubBoxOffset(size), size)

    @staticmethod
    def between(cornerA: Vec3iLike, cornerB: Vec3iLike):
        """Returns the Box between [cornerA] and [cornerB] (both inclusive),
        which may be any opposing corners"""
        first, last = orderedCorners3D(cornerA, cornerB)
        return Box(first, last - first + 1)

    @staticmethod
    def bounding(points: Iterable[Vec3iLike]):
        """Returns the smallest Box containing all [points]"""
        pointArray = np.array(points)
        minPoint = np.min(pointArray, axis=0)
        maxPoint = np.max(pointArray, axis=0)
        return Box(minPoint, maxPoint - minPoint + 1)

    def toRect(self):
        """Returns this Box's XZ-plane as a Rect"""
        return Rect(dropY(self._offset), dropY(self._size))

    @property
    def shell(self):
        """Yields all points on this Box's surface"""
        # It's surprisingly difficult to get this right without duplicates. (Think of the corners!)
        first = self.begin
        last  = self.end - 1
        # Bottom face
        yield from loop3D(ivec3(first.x, first.y, first.z), ivec3(last.x, first.y, last.z) + 1)
        # Top face
        yield from loop3D(ivec3(first.x, last.y, first.z), ivec3(last.x, last.y, last.z) + 1)
        # Sides
        yield from loop3D(ivec3(first.x, first.y+1, first.z), ivec3(last.x -1,  last.y-1, first.z   ) + 1)
        yield from loop3D(ivec3(last.x,  first.y+1, first.z), ivec3(last.x,     last.y-1, last.z  -1) + 1)
        yield from loop3D(ivec3(last.x,  first.y+1, last.z ), ivec3(first.x +1, last.y+1, last.z    ) - 1)
        yield from loop3D(ivec3(first.x, first.y+1, last.z ), ivec3(first.x,    last.y+1, first.z +1) - 1)

    @property
    def wireframe(self):
        """Yields all points on this Box's edges"""
        # It's surprisingly difficult to get this right without duplicates. (Think of the corners!)
        first = self.begin
        last  = self.end - 1
        # Bottom face
        yield from loop3D(ivec3(first.x, first.y, first.z), ivec3(last.x -1,  first.y, first.z   ) + 1)
        yield from loop3D(ivec3(last.x,  first.y, first.z), ivec3(last.x,     first.y, last.z  -1) + 1)
        yield from loop3D(ivec3(last.x,  first.y, last.z ), ivec3(first.x +1, first.y, last.z    ) - 1)
        yield from loop3D(ivec3(first.x, first.y, last.z ), ivec3(first.x,    first.y, first.z +1) - 1)
        # top face
        yield from loop3D(ivec3(first.x, last.y,  first.z), ivec3(last.x -1,  last.y,  first.z   ) + 1)
        yield from loop3D(ivec3(last.x,  last.y,  first.z), ivec3(last.x,     last.y,  last.z  -1) + 1)
        yield from loop3D(ivec3(last.x,  last.y,  last.z ), ivec3(first.x +1, last.y,  last.z    ) - 1)
        yield from loop3D(ivec3(first.x, last.y,  last.z ), ivec3(first.x,    last.y,  first.z +1) - 1)
        # sides
        yield from loop3D(ivec3(first.x, first.y+1, first.z), ivec3(first.x, last.y-1, first.z) + 1)
        yield from loop3D(ivec3(last.x,  first.y+1, first.z), ivec3(last.x,  last.y-1, first.z) + 1)
        yield from loop3D(ivec3(last.x,  first.y+1, last.z ), ivec3(last.x,  last.y-1, last.z ) + 1)
        yield from loop3D(ivec3(first.x, first.y+1, last.z ), ivec3(first.x, last.y-1, last.z ) + 1)


def rectSlice(array: np.ndarray, rect: Rect):
    """Returns the slice from [array] defined by [rect]"""
    return array[rect.begin.x:rect.end.x, rect.begin.y:rect.end.y]


def setRectSlice(array: np.ndarray, rect: Rect, value: Any):
    """Sets the slice from [array] defined by [rect] to [value]"""
    array[rect.begin.x:rect.end.x, rect.begin.y:rect.end.y] = value


def boxSlice(array: np.ndarray, box: Box):
    """Returns the slice from [array] defined by [box]"""
    return array[box.begin.x:box.end.x, box.begin.y:box.end.y, box.begin.z:box.end.z]


def setBoxSlice(array: np.ndarray, box: Box, value: Any):
    """Sets the slice from [array] defined by [box] to [value]"""
    array[box.begin.x:box.end.x, box.begin.y:box.end.y, box.begin.z:box.end.z] = value


# ==================================================================================================
# Point generation
# ==================================================================================================


def loop2D(begin: Vec2iLike, end: Optional[Vec2iLike] = None):
    """Yields all points between <begin> and <end> (end-exclusive).\n
    If <end> is not given, yields all points between (0,0) and <begin>."""
    if end is None:
        begin, end = (0, 0), begin

    for x in range(begin[0], end[0], nonZeroSign(end[0] - begin[0])):
        for y in range(begin[1], end[1], nonZeroSign(end[1] - begin[1])):
            yield ivec2(x, y)


def loop3D(begin: Vec3iLike, end: Optional[Vec3iLike] = None):
    """Yields all points between <begin> and <end> (end-exclusive).\n
    If <end> is not given, yields all points between (0,0,0) and <begin>."""
    if end is None:
        begin, end = (0, 0, 0), begin

    for x in range(begin[0], end[0], nonZeroSign(end[0] - begin[0])):
        for y in range(begin[1], end[1], nonZeroSign(end[1] - begin[1])):
            for z in range(begin[2], end[2], nonZeroSign(end[2] - begin[2])):
                yield ivec3(x, y, z)
