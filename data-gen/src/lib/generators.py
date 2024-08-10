from typing import Callable, Optional
import subprocess
import os
import logging

from .vector_tools import Box
from .minecraft_server import MinecraftServer


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
GENERATOR_DIR = f"{SCRIPT_DIR}/../../generators/code"


class GeneratorError(Exception):
    """An error occured while running a generator."""

class GeneratorTimeoutError(GeneratorError):
    """A generator timed out."""

class GeneratorFailedError(GeneratorError):
    """A generator failed to complete."""


logger = logging.getLogger(__name__)


def generateGdmcHttp(server: MinecraftServer, buildArea: Box, runProgram: Callable[[], None]):
    x1, y1, z1 = buildArea.offset
    x2, y2, z2 = buildArea.last
    server.start()
    server.runCommandRcon(f"/setbuildarea {x1} {y1} {z1} {x2} {y2} {z2}")
    try:
        runProgram()
    except GeneratorError as e:
        # Safety measure to ensure the server is restarted when a generator fails.
        # A generator fail may cause the server to become nonresponsive.
        server.stop()
        raise e


def generateGdmcHttpPython(server: MinecraftServer, buildArea: Box, workingDir: str, entryPoint: str, timeoutSeconds: Optional[float] = None):
    def generate():
        def logOutput(stdout: Optional[bytes]):
            if stdout is None:
                # This happened once when logging in to a server from WSL.
                # I'm not sure if WSL was the cause, but it happened exacly on login while the
                # generator was already being ran for many hours.
                logger.warning("Generator subprocess stdout is None even though it was captured!")
                return
            for line in stdout.decode("utf-8").splitlines():
                logger.debug("Gen: %s", line)

        try:
            result = subprocess.run(
                [".venv/bin/python", entryPoint],
                cwd=workingDir,
                check=True,
                timeout=timeoutSeconds,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            logOutput(result.stdout)

        except subprocess.TimeoutExpired as e:
            logOutput(e.stdout)
            raise GeneratorError(
                f"Generator \"{workingDir}/{entryPoint}\" timed out!\n"
                f"Re-raised from subprocess.TimeoutExpired:\n{e}"
            ) from e

        except subprocess.CalledProcessError as e:
            logOutput(e.stdout)
            raise GeneratorError(
                f"Generator \"{workingDir}/{entryPoint}\" failed!\n"
                f"Re-raised from subprocess.CalledProcessError:\n{e}"
            ) from e

    generateGdmcHttp(server, buildArea, generate)


def generateMikesAngels(server: MinecraftServer, buildArea: Box, timeoutSeconds: Optional[float] = None):
    generateGdmcHttpPython(server, buildArea, f"{GENERATOR_DIR}/gdmc-2022/Mikes-Angels", "src/main.py", timeoutSeconds)

def generateMikesAngelsRoads(server: MinecraftServer, buildArea: Box, timeoutSeconds: Optional[float] = None):
    generateGdmcHttpPython(server, buildArea, f"{GENERATOR_DIR}/gdmc-2022/Mikes-Angels-Roads", "src/main.py", timeoutSeconds)

def generateMikesAngelsWall(server: MinecraftServer, buildArea: Box, timeoutSeconds: Optional[float] = None):
    generateGdmcHttpPython(server, buildArea, f"{GENERATOR_DIR}/gdmc-2022/Mikes-Angels-Wall", "src/main.py", timeoutSeconds)

def generateMikesAngelsRoadsWall(server: MinecraftServer, buildArea: Box, timeoutSeconds: Optional[float] = None):
    generateGdmcHttpPython(server, buildArea, f"{GENERATOR_DIR}/gdmc-2022/Mikes-Angels-Roads-Wall", "src/main.py", timeoutSeconds)

def generateRing(server: MinecraftServer, buildArea: Box, timeoutSeconds: Optional[float] = None):
    generateGdmcHttpPython(server, buildArea, f"{GENERATOR_DIR}/ring", "main.py", timeoutSeconds)

def generateRingAdaptive(server: MinecraftServer, buildArea: Box, timeoutSeconds: Optional[float] = None):
    generateGdmcHttpPython(server, buildArea, f"{GENERATOR_DIR}/ring-adaptive", "main.py", timeoutSeconds)
