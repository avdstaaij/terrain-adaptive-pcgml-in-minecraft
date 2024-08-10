"""Generic utilities"""

from typing import Optional, TypeVar, Callable
from contextlib import contextmanager
import sys
import signal
import time
import logging


T = TypeVar("T")


def sign(x) -> int:
    """Returns the sign of [x]"""
    return (x > 0) - (x < 0)


def nonZeroSign(x) -> int:
    """Returns the sign of [x], except that non_zero_sign(0) == 1"""
    return 1 if x >= 0 else -1


def ceildiv(a: T, b: T) -> T:
    """Integer division that rounds up instead of down"""
    return -(-a // b)


@contextmanager
def timeLimitUnsafe(seconds: int):
    """Raises a TimeoutError if the code inside the context manager takes longer than <seconds>\n
    This uses UNIX signals, so it will not work on Windows. Furthermore, the time limited code may
    be interrupted even while it is handling an exception. This function should only be used as a
    last resort."""

    def signalHandler(signum, frame):
        raise TimeoutError(f"Function call timed out after {seconds} seconds")

    originalHandler = None
    isAlarmSet = False
    try:
        originalHandler = signal.signal(signal.SIGALRM, signalHandler)
        isAlarmSet = True
    except ValueError:
        pass
    if isAlarmSet:
        signal.alarm(seconds)

    try:
        yield
    finally:
        if isAlarmSet:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, originalHandler)


def withRetries(
    function:      Callable[[], T],
    exceptionType: type                             = Exception,
    retries:       int                              = 1,
    onRetry:       Callable[[Exception, int], None] = lambda *_: time.sleep(1),
    reRaise:       bool                             = True
):
    """Retries <function> up to <retries> times if an exception occurs.\n
    Before retrying, calls <onRetry>(last exception, remaining retries).
    The default callback sleeps for one second.\n
    If the retries have ran out and <reRaise> is True, the last exception is re-raised."""
    while True:
        try:
            return function()
        except exceptionType as e: # pylint: disable=broad-except
            if retries == 0:
                if reRaise:
                    raise e
                return None
            onRetry(e, retries)
            retries -= 1


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, flush=True, **kwargs)


@contextmanager
def timeAndPrint(taskName: str):
    eprint(f"{taskName} start", flush=True)
    startTime = time.time()
    yield
    eprint(f"{taskName} done ({time.time() - startTime:.2f}s)", flush=True)


@contextmanager
def timeAndLog(logger: logging.Logger, taskDescripton: Optional[str] = None, taskCompletedDescription: Optional[str] = None, level=logging.INFO):
    if taskDescripton is not None:
        logger.log(level, taskDescripton)
    startTime = time.time()
    yield
    if taskCompletedDescription is not None:
        logger.log(level, taskCompletedDescription % (time.time() - startTime))


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
