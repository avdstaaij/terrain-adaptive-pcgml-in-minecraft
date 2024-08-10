#!/usr/bin/env python3

from typing import Any, Callable, Dict, Optional
from contextlib import suppress
from functools import partial
import json
import time
import os
import os.path
import sys
import shutil
import logging
import argparse

import numpy as np
import glm
from glm import ivec2
from termcolor import colored
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from lib.util import ceildiv, eprint, promptConfirmation, withRetries, timeAndLog
from lib.vector_tools import Box, Rect, addY, loop2D, setRectSlice
from lib.constants import SPAWN_CHUNK_DIAMETER_BORDER, FILENAME_STYLE, ERROR_PREFIX
from lib.minecraft_server import MinecraftServer
from lib.world_generation import generateChunksRect, deleteGenerationTasks
from lib.generators import GeneratorError, generateMikesAngels, generateMikesAngelsRoads, generateMikesAngelsWall, generateMikesAngelsRoadsWall, generateRing, generateRingAdaptive
from lib.extraction import extractChunkBox, loadLevel, reconstructPaletteLookup


# ==================================================================================================


SCRIPT_DIR  = os.path.dirname(os.path.realpath(__file__))
SERVER_DIR  = f"{SCRIPT_DIR}/../minecraft-server/server"
SERVER_JAR_FILENAME = "forge-1.16.5-36.2.39.jar"
CHECKPOINT_SUBDIR = "checkpoints"
LAST_CHECKPOINT_FILENAME = "last-checkpoint.json"

GENERATORS = {
    "mikes_angels":            generateMikesAngels,
    "mikes_angels_roads":      generateMikesAngelsRoads,
    "mikes_angels_wall":       generateMikesAngelsWall,
    "mikes_angels_roads_wall": generateMikesAngelsRoadsWall,
    "ring":                    generateRing,
    "ring_adaptive":           generateRingAdaptive,
}
GENERATOR_NOOP = "none"

# Surely there won't be more than 65536 different blockStates, right?
BLOCKS_ARRAY_DTYPE = np.uint16

# Just in case it helps.
CHECKPOINT_RETRY_SLEEP_SECONDS = 60


# ==================================================================================================


def make_color_formatter(fg=None, bg=None, attrs=None):
    return logging.Formatter(colored("[%(asctime)s] [%(levelname)s]:", fg, bg, attrs) + " %(message)s")

COLOR_FORMATTERS = {
    logging.DEBUG:    make_color_formatter("green"),
    logging.INFO:     make_color_formatter("cyan"),
    logging.WARNING:  make_color_formatter("yellow"),
    logging.ERROR:    make_color_formatter("red"),
    logging.CRITICAL: make_color_formatter("red", attrs=["bold"]),
}

class CustomLoggingFormatter(logging.Formatter):
    def format(self, record):
        formatter = COLOR_FORMATTERS.get(record.levelno)
        return formatter.format(record)


consoleFormatter = CustomLoggingFormatter()
fileFormatter    = logging.Formatter("[%(asctime)s] [%(levelname)s]: %(message)s")

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(consoleFormatter)
consoleHandler.setLevel(logging.DEBUG) # Setting another level is not possible because logging_redirect_tqdm() does not copy it (https://github.com/tqdm/tqdm/pull/1333). Changing the root logger level does work, but also affects the file handler.

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(consoleHandler)

logging.getLogger("amulet").setLevel(logging.WARNING)


# ==================================================================================================


def runGeneratorWithRetries(
    generator:      Callable[[MinecraftServer, Box, int], None],
    server:         MinecraftServer,
    buildArea:      Box,
    timeoutSeconds: int,
    retries:        int
):
    """Runs <generator>(<server>, <buildAreaOffset>, <buildAreaSize>, <timeoutSeconds>) with retries."""

    buildRect = buildArea.toRect()

    def resetChunks():
        with timeAndLog(logger, "Resetting chunks...", "Chunks reset in %.2fs"):
            generateChunksRect(server, Rect(buildRect.offset // 16, buildRect.size // 16), overwrite=True)

    def onRetry(_, retriesLeft: int):
        logger.warning("Generator failed. Retrying... (%i retries left)", retriesLeft, exc_info=True)
        resetChunks()
        logger.info("Re-running generator.")

    try:
        withRetries(
            lambda: generator(server, buildArea, timeoutSeconds),
            GeneratorError, retries=retries, onRetry=onRetry, reRaise=True
        )
    except GeneratorError:
        logger.error("Generator failed %i times; skipping this sample. Last exception:", retries+1, exc_info=True)
        resetChunks()


def getCheckpointSampleRect(sampleCounts: ivec2, checkpointSize: ivec2, checkpointPosition: ivec2):
    checkpointSampleOffset = checkpointPosition * checkpointSize
    return Rect(
        checkpointSampleOffset,
        glm.min(checkpointSize, sampleCounts - checkpointSampleOffset)
    )


def getCheckpointDir(outputDir: str, checkpointPosition: Optional[ivec2]):
    if checkpointPosition is None:
        return f"{outputDir}/{CHECKPOINT_SUBDIR}/initialization"
    return f"{outputDir}/{CHECKPOINT_SUBDIR}/{checkpointPosition.x}-{checkpointPosition.y}"


def generateDatasetContinue(
    outputDir:         str,
    serverMemoryGigs:  int,
    sampleTimeoutSecs: int,
    sampleRetries:     int,
):
    with open(f"{outputDir}/{LAST_CHECKPOINT_FILENAME}", encoding="utf-8") as f:
        checkpointPosition = json.load(f)

    if checkpointPosition is None:
        logger.info("Starting generation process from initial checkpoint.")
    else:
        checkpointPosition = ivec2(checkpointPosition)
        logger.info("Continuing generation process from checkpoint %s.", tuple(checkpointPosition))

    server = MinecraftServer(
        SERVER_DIR, SERVER_JAR_FILENAME,
        javaArgs=[f"-Xmx{serverMemoryGigs}G", f"-Xms{serverMemoryGigs}G"],
    )

    with timeAndLog(logger, "Loading settings and state...", "Settings and state loaded in %.2fs"):
        with open(f"{getCheckpointDir(outputDir, None)}/settings.json", encoding="utf-8") as f:
            settings: Dict[str, Any] = json.load(f)
            generatorName      = str  (settings["generatorName"])
            buildAreaChunkSize = ivec2(settings["buildAreaChunkSize"])
            yBegin             = int  (settings["yBegin"])
            yEnd               = int  (settings["yEnd"])
            chunkOffset        = ivec2(settings["chunkOffset"])
            sampleCounts       = ivec2(settings["sampleCounts"])
            regenerateTerrain  = bool (settings["regenerateTerrain"])
            checkpointSize     = ivec2(settings["checkpointSize"])

        with suppress(FileNotFoundError): shutil.rmtree(server.worldPath)
        with suppress(FileNotFoundError): shutil.copytree(f"{getCheckpointDir(outputDir, checkpointPosition)}/world", server.worldPath)

        palette = []
        with suppress(FileNotFoundError):
            with open(f"{getCheckpointDir(outputDir, checkpointPosition)}/palette.json", encoding="utf-8") as f:
                palette = json.load(f)

        if generatorName == GENERATOR_NOOP:
            generator = None
        else:
            generator = GENERATORS[generatorName]

        # Number of checkpoints to create in each axis.
        checkpointCounts = ceildiv(sampleCounts, checkpointSize)

        paletteLookup = reconstructPaletteLookup(palette)

    server.start()
    server.runCommandRcon("gamerule randomTickSpeed 0")
    server.runCommandRcon("gamerule doDaylightCycle false")
    server.runCommandRcon("gamerule doWeatherCycle false")
    server.runCommandRcon("gamerule doFireTick false")
    server.runCommandRcon("time set day")
    server.runCommandRcon("weather clear")

    worldSpawn = addY((chunkOffset - ivec2(ceildiv(SPAWN_CHUNK_DIAMETER_BORDER+2, 2))) * 16, 64)
    server.runCommandRcon(f"setworldspawn {worldSpawn.x} {worldSpawn.y} {worldSpawn.z}")
    logger.info("Set world spawn to %s.", tuple(worldSpawn))

    checkpointPositions = list(loop2D(checkpointCounts))
    firstCheckpointIndex = 0 if checkpointPosition is None else checkpointPositions.index(checkpointPosition) + 1

    prevCheckpointPosition = None

    for checkpointPosition in tqdm.tqdm(checkpointPositions[firstCheckpointIndex:], total=len(checkpointPositions), initial=firstCheckpointIndex, desc="Checkpoints"):
        with timeAndLog(logger, f"Generating data for checkpoint {tuple(checkpointPosition)}...", "Checkpoint completed in %.2fs"):

            checkpointSampleRect = getCheckpointSampleRect(sampleCounts, checkpointSize, checkpointPosition)

            blocksShape = (*checkpointSampleRect.size, yEnd - yBegin, *(buildAreaChunkSize * 16)) # Each sample in YXZ order
            blocks: np.ndarray = np.full(blocksShape, -1, dtype=BLOCKS_ARRAY_DTYPE)

            if regenerateTerrain:
                with timeAndLog(logger, "(Re)generating terrain...", "Terrain (re)generated in %.2fs"):
                    for _ in tqdm.tqdm([0], desc="(Re)generating terrain", leave=False): # For consistency
                        deleteGenerationTasks(server)
                        generateChunksRect(server, Rect(chunkOffset + checkpointSampleRect.offset * buildAreaChunkSize, checkpointSampleRect.size * buildAreaChunkSize), overwrite=True)

            if generator is not None:
                with timeAndLog(logger, "Running generator...", "All generator runs completed in %.2fs"):
                    for samplePosition in tqdm.tqdm(list(checkpointSampleRect.inner), desc="Running generator", leave=False):
                        sampleBox = Box(
                            addY((chunkOffset + samplePosition * buildAreaChunkSize) * 16, yBegin),
                            addY(buildAreaChunkSize                                  * 16, yEnd - yBegin),
                        )
                        with timeAndLog(logger, f"Running generator at {tuple(sampleBox.offset)}...", "Generation completed in %.2fs"):
                            runGeneratorWithRetries(generator, server, sampleBox, sampleTimeoutSecs, sampleRetries)

            server.stop()

            with timeAndLog(logger, "Extracting samples...", "Samples extracted in %.2fs"):
                with loadLevel(server.worldPath) as world:
                    for samplePosition in tqdm.tqdm(list(checkpointSampleRect.inner), desc="Extracting samples", leave=False):
                        sampleChunkRect = Rect(
                            chunkOffset + samplePosition * buildAreaChunkSize,
                            buildAreaChunkSize,
                        )
                        extractChunkBox(world, sampleChunkRect, yBegin, yEnd, blocks[tuple(samplePosition - checkpointSampleRect.offset)], palette, paletteLookup)

            with timeAndLog(logger, "Saving checkpoint...", "Checkpoint saved in %.2fs"):
                checkpointOutputDir = getCheckpointDir(outputDir, checkpointPosition)
                os.makedirs(f"{checkpointOutputDir}", exist_ok=True)
                np.savez_compressed(f"{checkpointOutputDir}/blocks.npz", blocks=blocks)
                with open(f"{checkpointOutputDir}/palette.json", "w", encoding="utf-8") as f:
                    json.dump(palette, f, separators=(',', ':'))
                shutil.copytree(server.worldPath, f"{checkpointOutputDir}/world", symlinks=True, dirs_exist_ok=False)

                prevCheckpointOutputDir = getCheckpointDir(outputDir, prevCheckpointPosition)
                with suppress(FileNotFoundError): os.remove(f"{prevCheckpointOutputDir}/palette.json")
                with suppress(FileNotFoundError): shutil.rmtree(f"{prevCheckpointOutputDir}/world")

                with open(f"{outputDir}/{LAST_CHECKPOINT_FILENAME}", "w", encoding="utf-8") as f:
                    json.dump(list(checkpointPosition), f)

        prevCheckpointPosition = checkpointPosition

    with timeAndLog(logger, "Consolidating checkpoints...", "Checkpoints consolidated in %.2fs"):
        blocksShape = (sampleCounts.x, sampleCounts.y, yEnd - yBegin, *(buildAreaChunkSize * 16)) # Each sample in YXZ order
        blocks: np.memmap = np.lib.format.open_memmap(f"{outputDir}/blocks.npy", mode="w+", shape=blocksShape, dtype=BLOCKS_ARRAY_DTYPE)
        for checkpointPosition in tqdm.tqdm(list(loop2D(checkpointCounts)), desc="Consolidating checkpoints", leave=False):
            checkpointOutputDir = getCheckpointDir(outputDir, checkpointPosition)
            checkpointBlocks = np.load(f"{checkpointOutputDir}/blocks.npz")["blocks"]
            checkpointSampleRect = getCheckpointSampleRect(sampleCounts, checkpointSize, checkpointPosition)
            setRectSlice(blocks, checkpointSampleRect, checkpointBlocks)
        blocks.flush()
        shutil.copy2(f"{checkpointOutputDir}/palette.json", f"{outputDir}/palette.json")
        with suppress(FileNotFoundError): shutil.rmtree(f"{outputDir}/world")
        shutil.copytree(f"{checkpointOutputDir}/world", f"{outputDir}/world", symlinks=True, dirs_exist_ok=False)
        shutil.rmtree(f"{outputDir}/checkpoints")
        os.remove(f"{outputDir}/{LAST_CHECKPOINT_FILENAME}")


def generateDatasetContinueWithRetries(
    outputDir:         str,
    serverMemoryGigs:  int,
    sampleTimeoutSecs: int,
    sampleRetries:     int,
    checkpointRetries: int,
):
    def onRetry(_, retriesLeft: int):
        # This could be a warning, but the time loss is so severe that an error is warranted.
        logger.error("Checkpoint failed. Retrying after %i seconds... (%i retries left)", CHECKPOINT_RETRY_SLEEP_SECONDS, retriesLeft, exc_info=True)
        time.sleep(CHECKPOINT_RETRY_SLEEP_SECONDS)

    try:
        withRetries(
            lambda: generateDatasetContinue(outputDir, serverMemoryGigs, sampleTimeoutSecs, sampleRetries),
            exceptionType=Exception, retries=checkpointRetries, onRetry=onRetry
        )
    except Exception: # pylint: disable=broad-except
        logger.critical("Checkpoint failed %i times. Last exception:", checkpointRetries+1, exc_info=True)


def generateDatasetNew(
    outputDir:          str,
    generatorName:      str,
    buildAreaChunkSize: ivec2,
    yBegin:             int,
    yEnd:               int,
    chunkOffset:        ivec2,
    sampleCounts:       ivec2,
    initialWorldPath:   Optional[str],
    initialPalettePath: Optional[str],
    regenerateTerrain:  bool,
    checkpointSize:     ivec2,
    serverMemoryGigs:   int,
    sampleTimeoutSecs:  int,
    sampleRetries:      int,
    checkpointRetries:  int,
):
    if generatorName != GENERATOR_NOOP and generatorName not in GENERATORS.keys():
        eprint(
            f"{ERROR_PREFIX} Invalid generator {colored(generatorName, **FILENAME_STYLE)}\n"
            "Available generators:\n"
            f"{', '.join(list(GENERATORS.keys()) + [GENERATOR_NOOP])}"
        )
        sys.exit(1)

    os.makedirs(outputDir, exist_ok=False)

    with open(f"{outputDir}/properties.json", "w", encoding="utf-8") as f:
        json.dump({
            "generatorName":      generatorName,
            "buildAreaChunkSize": list(buildAreaChunkSize),
            "yBegin":             yBegin,
            "yEnd":               yEnd,
            "chunkOffset":        list(chunkOffset),
            "sampleCounts":       list(sampleCounts),
            "withInitialWorld":   initialWorldPath is not None,
            "withInitialPalette": initialPalettePath is not None,
            "regenerateTerrain":  regenerateTerrain,
        }, f, indent=4)

    with timeAndLog(logger, "Saving initial checkpoint...", "Initial checkpoint saved in %.2fs"):
        initialCheckpointDir = getCheckpointDir(outputDir, None)
        os.makedirs(initialCheckpointDir, exist_ok=True)
        with open(f"{initialCheckpointDir}/settings.json", "w", encoding="utf-8") as f:
            json.dump({
                "generatorName":      generatorName,
                "buildAreaChunkSize": list(buildAreaChunkSize),
                "yBegin":             yBegin,
                "yEnd":               yEnd,
                "chunkOffset":        list(chunkOffset),
                "sampleCounts":       list(sampleCounts),
                "regenerateTerrain":  regenerateTerrain,
                "checkpointSize":     list(checkpointSize),
            }, f, indent=4)
        if initialWorldPath is not None:
            shutil.copytree(initialWorldPath, f"{initialCheckpointDir}/world")
        if initialPalettePath is not None:
            # Parse and re-dump palette file as a basic validity check and to remove whitespace.
            with open(initialPalettePath, encoding="utf-8") as f:
                palette = json.load(f)
            with open(f"{initialCheckpointDir}/palette.json", "w", encoding="utf-8") as f:
                json.dump(palette, f, separators=(',', ':'))
        with open(f"{outputDir}/{LAST_CHECKPOINT_FILENAME}", "w", encoding="utf-8") as f:
            json.dump(None, f)

    generateDatasetContinueWithRetries(
        outputDir         = outputDir,
        serverMemoryGigs  = serverMemoryGigs,
        sampleTimeoutSecs = sampleTimeoutSecs,
        sampleRetries     = sampleRetries,
        checkpointRetries = checkpointRetries,
    )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=partial(argparse.ArgumentDefaultsHelpFormatter, max_help_position=120),
        description="Generates a dataset of samples for the specified generator."
    )
    parser.add_argument("output_dir",           type=str,                   help="Directory to save the data in.")
    parser.add_argument("generator",            type=str,                   help="Generator to use.")
    parser.add_argument("chunk_offset_x",       type=int,                   help="Chunk offset in the X axis")
    parser.add_argument("chunk_offset_z",       type=int,                   help="Chunk offset in the Z axis")
    parser.add_argument("sample_count_x",       type=int,                   help="Number of samples in the X axis")
    parser.add_argument("sample_count_z",       type=int,                   help="Number of samples in the Z axis")
    parser.add_argument("--build-length",       type=int,   default=4,      metavar="<CHUNKS>",  help="Length/width of the build area in chunks")
    parser.add_argument("--y-min",              type=int,   default=0,      metavar="<BLOCKS>",  help="Min Y-coordinate in blocks.")
    parser.add_argument("--y-max",              type=int,   default=255,    metavar="<BLOCKS>",  help="Max Y-coordinate in blocks.")
    parser.add_argument("--initial-world",      type=str,   default=None,   metavar="<PATH>",    help="Initial Minecraft world directory")
    parser.add_argument("--initial-palette",    type=str,   default=None,   metavar="<PATH>",    help="Initial palette file")
    parser.add_argument("--no-chunk-regen",     action="store_true",                             help="Don't regen natural chunks before running generator.")
    parser.add_argument("--checkpoint-length",  type=int,   default=16,     metavar="<SAMPLES>", help="Length/width of area to gen before saving a checkpoint.")
    parser.add_argument("--server-memory",      type=int,   default=8,      metavar="<GB>",      help="Amount of memory to allocate to the server in GB.")
    parser.add_argument("--sample-timeout",     type=float, default=60*2,   metavar="<SECONDS>", help="Number of seconds to wait for the generator to finish.")
    parser.add_argument("--sample-retries",     type=int,   default=2,      metavar="<COUNT>",   help="Number of times to retry on generator failure.")
    parser.add_argument("--checkpoint-retries", type=int,   default=2,      metavar="<COUNT>",   help="Number of times to retry on checkpoint failure.")
    parser.add_argument("--log-level",          type=str,   default="info", metavar="<LEVEL>",   help="Log level to use.", choices=["debug", "info", "warning", "error", "critical"])
    parser.add_argument("--log-file",           type=str,   default=None,   metavar="<PATH>",    help="File to save (non-colored) log in.")
    args = parser.parse_args()

    # ----------------------------------------------------------------------------------------------

    # See note about logging and TQDM at the top of this file.
    logger.setLevel(args.log_level.upper())

    if args.log_file is not None:
        dirname = os.path.dirname(args.log_file)
        if dirname != "":
            os.makedirs(dirname, exist_ok=True)
        fileHandler = logging.FileHandler(args.log_file)
        fileHandler.setLevel(logging.DEBUG)
        fileHandler.setFormatter(fileFormatter)
        logger.addHandler(fileHandler)

    # ----------------------------------------------------------------------------------------------

    with logging_redirect_tqdm():
        try:

            outputDir = args.output_dir

            if os.path.exists(outputDir):
                if os.path.exists(f"{outputDir}/{LAST_CHECKPOINT_FILENAME}"):
                    eprint(
                        "\n"
                        f"It seems that {colored(outputDir, **FILENAME_STYLE)} contains an in-progress dataset.\n"
                        "If the dataset generation process crashed for some reason, you can continue from\n"
                        "the last checkpoint.\n"
                        "Note that you SHOULD NOT do this if the dataset generation process is still\n"
                        "running!\n"
                        "Also note that most specified generation settings will be ignored in favor of\n"
                        "the ones saved in the in-progress dataset.\n"
                        "If instead you want to overwrite the existing partial dataset, delete the\n"
                        "directory and run this program again.\n"
                    )
                    if promptConfirmation("Continue in-progress dataset from last checkpoint?"):
                        eprint()
                        with timeAndLog(logger, None, "Done in %.2fs"):
                            generateDatasetContinueWithRetries(
                                outputDir         = outputDir,
                                serverMemoryGigs  = args.server_memory,
                                sampleTimeoutSecs = args.sample_timeout,
                                sampleRetries     = args.sample_retries,
                                checkpointRetries = args.checkpoint_retries,
                            )
                    return

                eprint(f"{ERROR_PREFIX} Output directory {colored(outputDir, **FILENAME_STYLE)} already exists!")
                sys.exit(1)

            with timeAndLog(logger, None, "Done in %.2fs"):
                generateDatasetNew(
                    outputDir          = outputDir,
                    generatorName      = args.generator,
                    buildAreaChunkSize = ivec2(args.build_length),
                    yBegin             = args.y_min,
                    yEnd               = args.y_max+1,
                    chunkOffset        = ivec2(args.chunk_offset_x, args.chunk_offset_z),
                    sampleCounts       = ivec2(args.sample_count_x, args.sample_count_z),
                    initialWorldPath   = args.initial_world,
                    initialPalettePath = args.initial_palette,
                    regenerateTerrain  = not args.no_chunk_regen,
                    checkpointSize     = ivec2(args.checkpoint_length),
                    serverMemoryGigs   = args.server_memory,
                    sampleTimeoutSecs  = args.sample_timeout,
                    sampleRetries      = args.sample_retries,
                    checkpointRetries  = args.checkpoint_retries,
                )

        except Exception: # pylint: disable=broad-except
            logger.critical("Unhandled exception.", exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    main()
