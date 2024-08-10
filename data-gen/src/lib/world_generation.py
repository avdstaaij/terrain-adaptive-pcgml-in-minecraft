import time
from uuid import uuid4
import logging

from glm import ivec2

from .util import ceildiv
from .vector_tools import Rect
from .minecraft_server import MinecraftServer


CHUNK_RECT_DIAMETER_MAX = 64


logger = logging.getLogger(__name__)


def _generateChunksRadius(server: MinecraftServer, chunkOrigin: ivec2, chunkRadius: int, completionPollDelaySeconds = 1.0, overwrite=False):
    """Generates chunks.\n
    Note: the chunk size of the generated area will always be 2 * <chunkRadius> in each axis, so
    generating an area with with an odd size is not possible."""

    server.start()

    logger.debug("Generating chunks around %s with radius %s.", chunkOrigin, chunkRadius)

    def doTask(isDeletion: bool):
        taskName = "task-" + str(uuid4())
        server.runCommandRcon(f"pregen start {'delete' if isDeletion else 'gen'} radius {taskName} SQUARE {chunkOrigin.x} {chunkOrigin.y} {chunkRadius}")
        while True:
            taskList = server.runCommandRcon(f"pregen taskList {'deletion' if isDeletion else 'gen'}")
            if not any(line.startswith(f"[Name={taskName}") for line in taskList.splitlines()):
                break
            time.sleep(completionPollDelaySeconds)

    if overwrite:
        doTask(isDeletion=True)
        # It seems that we have to restart the server here to prevent deletion/generation issues.
        # The performance hit is significant, but I see no other way to prevent these issues.
        server.stop()
        server.start()
    doTask(isDeletion=False)


def generateChunksRect(server: MinecraftServer, chunkRect: Rect, completionPollDelaySeconds = 1.0, overwrite=False):
    """Generates chunks.\n
    Note: uneven size axes will be rounded up to the next even number."""
    settings = (completionPollDelaySeconds, overwrite)

    # If the chunkRect is not square, we recursively split it into a square rect and a
    # remainder rect.
    if chunkRect.size.x < chunkRect.size.y:
        generateChunksRect(server, Rect(chunkRect.offset, ivec2(chunkRect.size.x, chunkRect.size.x)), *settings)
        generateChunksRect(server, Rect(chunkRect.offset + ivec2(0, chunkRect.size.x), ivec2(chunkRect.size.x, chunkRect.size.y - chunkRect.size.x)), *settings)
        return
    if chunkRect.size.y < chunkRect.size.x:
        generateChunksRect(server, Rect(chunkRect.offset, ivec2(chunkRect.size.y, chunkRect.size.y)), *settings)
        generateChunksRect(server, Rect(chunkRect.offset + ivec2(chunkRect.size.y, 0), ivec2(chunkRect.size.x - chunkRect.size.y, chunkRect.size.y)), *settings)
        return

    # At this point, chunkRect is square.
    diameter = chunkRect.size.x

    # If the diameter is too large, we recursively split chunkRect into four quadrants.
    # We perform this split because the Chunk Pregenerator mod sometimes fails for large tasks.
    if diameter > CHUNK_RECT_DIAMETER_MAX:
        diameterHalf1 = diameter // 2
        diameterHalf2 = diameter - diameterHalf1
        generateChunksRect(server, Rect(chunkRect.offset                                      , ivec2(diameterHalf1, diameterHalf1)), *settings)
        generateChunksRect(server, Rect(chunkRect.offset + ivec2(diameterHalf1, 0            ), ivec2(diameterHalf2, diameterHalf1)), *settings)
        generateChunksRect(server, Rect(chunkRect.offset + ivec2(0            , diameterHalf1), ivec2(diameterHalf1, diameterHalf2)), *settings)
        generateChunksRect(server, Rect(chunkRect.offset + ivec2(diameterHalf1, diameterHalf1), ivec2(diameterHalf2, diameterHalf2)), *settings)
        return

    # Actually generate the chunks.
    _generateChunksRadius(server, chunkRect.center, ceildiv(chunkRect.size.x, 2), *settings)


def deleteGenerationTasks(server: MinecraftServer):
    """Stops and deletes all generation tasks"""
    server.start()
    for taskType in ("gen", "deletion"):
        taskList = server.runCommandRcon(f"pregen taskList {taskType}")
        for line in taskList.splitlines():
            if line.startswith("[Name="):
                taskName = line.split("Name=")[1].split(",")[0]
                server.runCommandRcon(f"pregen stop {taskName} true")
