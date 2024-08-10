#!/usr/bin/env python3


from gdpc import interface
from gdpc.worldLoader import WorldSlice


def ring(x1, z1, x2, z2):
    for x in range(x1, x2+1):
        yield x, z1
        yield x, z2
    for z in range(z1+1, z2):
        yield x1, z
        yield x2, z


def main():
    x1, y1, z1, x2, y2, z2 = interface.requestBuildArea()

    worldSlice = WorldSlice(x1, z1, x2+1, z2+1)
    heightmap = worldSlice.heightmaps["MOTION_BLOCKING_NO_LEAVES"]

    for x, z in ring(x1, z1, x2, z2):
        y = heightmap[x-x1, z-z1]
        interface.placeBlock(x, y, z, "red_concrete")


if __name__ == "__main__":
    main()
