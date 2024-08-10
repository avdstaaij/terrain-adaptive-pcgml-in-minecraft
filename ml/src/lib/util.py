"""Generic utilities"""

from typing import Callable, Optional, Union
from functools import partial
import warnings
from contextlib import contextmanager
import tempfile
import sys
import os

import psutil
import numpy as np
from tqdm import tqdm as _tqdm


class tqdm(_tqdm):
    """A tqdm wrapper with some better defaults."""
    def __init__(self, *args, **kwargs):
        if "file"          not in kwargs: kwargs["file"]          = sys.stderr
        if "dynamic_ncols" not in kwargs: kwargs["dynamic_ncols"] = True
        super().__init__(*args, **kwargs)


@contextmanager
def suppressWarnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def invertPermutation(permutation: np.ndarray):
    assert permutation.ndim == 1
    inverted = np.empty_like(permutation)
    inverted[permutation] = np.arange(len(permutation))
    return inverted


@contextmanager
def tempMmapArray(shape, dtype):
    with tempfile.TemporaryDirectory() as tempdir:
        array = np.memmap(os.path.join(tempdir, "array.dat"), mode="w+", shape=shape, dtype=dtype)
        yield array
        del array


@contextmanager
def tempArray(shape, dtype, maxMemory: Union[int, float] = 0.75):
    if maxMemory <= 0:
        raise ValueError("maxMemory should be nonnegative.")
    if maxMemory <= 1:
        maxMemory = maxMemory * psutil.virtual_memory().total
    maxMemory = min(int(maxMemory), psutil.virtual_memory().available)

    if np.dtype(dtype).itemsize * np.prod(shape) > maxMemory:
        with tempMmapArray(shape, dtype) as array:
            yield array
    else:
        array = np.empty(shape, dtype=dtype)
        yield array
        del array


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)


def nextAvailablePath(path: str):
    base, extension = os.path.splitext(path)
    i = 2
    while os.path.exists(path):
        path = f"{base} ({i}){extension}"
        i += 1
    return path


# Based on https://stackoverflow.com/a/3041990
def promptConfirmation(prompt, default: Optional[bool] = None, postfix = ": "):
    options = {"yes": True, "y": True, "ye": True, "no": False, "n": False}

    if default is None:
        optionPrompt = " [y/n]"
    elif default:
        optionPrompt = " [Y/n]"
    else:
        optionPrompt = " [y/N]"

    while True:
        eprint(f"{prompt}{optionPrompt}{postfix}", end="")
        choice = input().lower()
        if default is not None and choice == "":
            return default
        if choice in options:
            return options[choice]
        eprint('Please respond with "yes" or "no" (or "y" or "n").')


def promptInteger(prompt, default: Optional[int] = None, min: Optional[int] = None, max: Optional[int] = None, postfix = ": "): # pylint: disable=redefined-builtin
    defaultPrompt = f" ({default})" if default is not None else ""

    if min is not None and max is not None:
        optionPrompt = f" [{min} <= . <= {max}]"
    elif min is not None:
        optionPrompt = f" [>={min}]"
    elif max is not None:
        optionPrompt = f" [<={max}]"
    else:
        optionPrompt = ""

    while True:
        eprint(f"{prompt}{optionPrompt}{defaultPrompt}{postfix}", end="")
        answer = input()
        if default is not None and answer == "":
            return default
        try:
            value = int(answer)
            if (min is None or value >= min) and (max is None or value <= max):
                return value
        except ValueError:
            pass
        eprint("Please enter a valid integer.")


ProgressCallback = Optional[Callable[[int], Callable[[int], None]]]

def tqdmProgressCallback(*tqdm_args, **tqdm_kwargs):
    def progress(total: int):
        progressBar = tqdm(total=total, *tqdm_args, **tqdm_kwargs)

        def step(steps: int):
            progressBar.update(steps)
            if progressBar.n >= progressBar.total:
                progressBar.close()

        return step

    return progress
