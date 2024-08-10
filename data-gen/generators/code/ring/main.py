#!/usr/bin/env python3


from gdpc import interface
from gdpc import geometry


def main():
    x1, y1, z1, x2, y2, z2 = interface.requestBuildArea()

    y = max(y1, min(y2, 100))

    geometry.placeLine(x1, y, z1, x2, y, z1, "red_concrete")
    geometry.placeLine(x1, y, z1, x1, y, z2, "red_concrete")
    geometry.placeLine(x1, y, z2, x2, y, z2, "red_concrete")
    geometry.placeLine(x2, y, z1, x2, y, z2, "red_concrete")


if __name__ == "__main__":
    main()
