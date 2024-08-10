#!/usr/bin/env python3

from typing import Callable, List, Optional, Tuple
import os
import sys

import cloup
import numpy as np
import torch
from glm import ivec2, ivec3
from gdpc import Editor, Block, Box
from gdpc.vector_tools import loop3D, dropY

from lib.util import tqdm, eprint, tqdmProgressCallback
from lib.constants import ERROR_PREFIX, WARNING_PREFIX
from lib.minecraft_util import AIR_BLOCKS, AUTOMATIC_BLOCK_STATES, LIQUID_BLOCKS
from lib.palette_tools import Palette, countBlockTypeCounts, loadPalette
from lib.ml.models.vox2vox import Vox2voxGenerator
from lib.amulet_wrapper import amulet
from lib.amulet_util import AMULET_MINECRAFT_VERSION, loadLevel


defaultDeviceName = "cuda" if torch.cuda.is_available() else "cpu"


# TODO: Amulet support was very haphazardly added, so the code is quite messy.
# Things can undoubtedly be cleaned up with an abstract "WorldEditor" class or something like that.


# ==================================================================================================


def getTerrainBinaryGdpc(size: ivec3, bottomY: int, remove_flow: bool):
    eprint("Setting up gdpc editor and loading world slice...")
    editor = Editor(buffering=True, bufferLimit=2048)
    buildBox = editor.getBuildArea()
    buildRect = buildBox.toRect()
    genRect = buildRect.centeredSubRect(dropY(size))
    genBox = genRect.toBox(bottomY, size.y)
    editor.loadWorldSlice(genRect, heightmapTypes=None, cache=True)

    eprint("Converting terrain to input tensor...")
    terrain = np.zeros((size.y, size.x, size.z), dtype=bool)
    for pos in tqdm(list(loop3D(genBox.size))):
        block = editor.getBlock(pos + genBox.offset)
        block.id = block.id[len("minecraft:"):]
        if block.id not in AIR_BLOCKS and not (remove_flow and block.id == "water" and block.states["level"] != "0"):
            terrain[pos.y, pos.x, pos.z] = True

    return terrain, genBox, editor


def getTerrainBinaryAmulet(worldPath: str, offsetXZ: ivec2, size: ivec3, bottomY: int, remove_flow: bool):
    # TODO: Go chunk-by-chunk for better performance.
    genBox = Box((offsetXZ[0], bottomY, offsetXZ[1]), size)

    eprint("Converting terrain to input tensor...")
    terrain = np.zeros((size.y, size.x, size.z), dtype=bool)

    with loadLevel(worldPath) as world:
        for pos in tqdm(list(loop3D(genBox.size))):
            block, _ = world.get_version_block(*(pos + genBox.offset), "minecraft:overworld", AMULET_MINECRAFT_VERSION)

            terrain[pos.y, pos.x, pos.z] = (
                isinstance(block, amulet.api.block.Block)
                and block.base_name not in AIR_BLOCKS
                and not (remove_flow and block.base_name == "water" and block.properties["level"].py_str != "0")
            )

    return terrain, genBox


def getTerrainEmbeddingGdpc(size: ivec3, bottomY: int, palette: Palette, embeddings: np.ndarray, prune_auto_states: bool, remove_flow: bool, fix_1_16_palette: bool):
    eprint("Setting up gdpc editor and loading world slice...")
    editor = Editor(buffering=True, bufferLimit=2048)
    buildBox = editor.getBuildArea()
    buildRect = buildBox.toRect()
    genRect = buildRect.centeredSubRect(dropY(size))
    genBox = genRect.toBox(bottomY, size.y)
    editor.loadWorldSlice(genRect, heightmapTypes=None, cache=True)

    eprint("Converting terrain to input tensor...")
    unknownBlocks = set()
    airIndex = palette.index(("air", {}))
    terrain = np.zeros((embeddings.shape[1], size.y, size.x, size.z), dtype=embeddings.dtype)
    for pos in tqdm(list(loop3D(genBox.size))):
        block = editor.getBlock(pos + genBox.offset)
        block.id = block.id[len("minecraft:"):]
        if prune_auto_states:
            for state in AUTOMATIC_BLOCK_STATES:
                if state in block.states:
                    del block.states[state]
        if remove_flow:
            if block.id in LIQUID_BLOCKS:
                if block.states["level"] != "0":
                    block.id = "air"
                    block.states = {}
                else:
                    del block.states["level"]
        if fix_1_16_palette:
            if block.id == "dirt_path":
                block.id = "grass_path"
            if "waterlogged" in block.states and "leaves" in block.id:
                del block.states["waterlogged"]
        try:
            index = palette.index((block.id, block.states))
        except ValueError:
            if block.id not in unknownBlocks:
                tqdm.write(WARNING_PREFIX + f"Found a block that does not appear in the palette: {block}")
                unknownBlocks.add(block.id)
            index = airIndex
        terrain[:, pos.y, pos.x, pos.z] = embeddings[index]

    return terrain, genBox, editor


def getTerrainEmbeddingAmulet(worldPath: str, offsetXZ: ivec2, size: ivec3, bottomY: int, palette: Palette, embeddings: np.ndarray, prune_auto_states: bool, remove_flow: bool, fix_1_16_palette: bool):
    genBox = Box((offsetXZ[0], bottomY, offsetXZ[1]), size)

    eprint("Converting terrain to input tensor...")
    unknownBlocks = set()
    airIndex = palette.index(("air", {}))
    terrain = np.zeros((embeddings.shape[1], size.y, size.x, size.z), dtype=embeddings.dtype)
    with loadLevel(worldPath) as world:
        for pos in tqdm(list(loop3D(genBox.size))):
            block, _ = world.get_version_block(*(pos + genBox.offset), "minecraft:overworld", AMULET_MINECRAFT_VERSION)
            bid    = block.base_name
            states = {key: val.py_str for key, val in block.properties.items()}
            if prune_auto_states:
                for state in AUTOMATIC_BLOCK_STATES:
                    if state in states:
                        del states[state]
            if remove_flow:
                if bid in LIQUID_BLOCKS:
                    if states["level"] != "0":
                        bid = "air"
                        states = {}
                    else:
                        del states["level"]
            if fix_1_16_palette:
                if bid == "dirt_path":
                    bid = "grass_path"
                if "waterlogged" in states and "leaves" in bid:
                    del states["waterlogged"]
            try:
                index = palette.index((bid, states))
            except ValueError:
                if bid not in unknownBlocks:
                    tqdm.write(WARNING_PREFIX + f"Found a block that does not appear in the palette: {block}")
                    unknownBlocks.add(bid)
                index = airIndex
            terrain[:, pos.y, pos.x, pos.z] = embeddings[index]

    return terrain, genBox


def placeOutputBinaryGdpc(output: np.ndarray, differenceMask: np.ndarray, genBox: Box, editor: Editor):
    eprint("Placing output...")
    with editor.pushTransform(genBox.offset):
        for pos in tqdm([p for p in loop3D(genBox.size) if differenceMask[p.y, p.x, p.z]]):
            editor.placeBlock(pos, Block("minecraft:red_concrete") if output[pos.y, pos.x, pos.z] else Block("minecraft:glass"))


def placeOutputBinaryAmulet(output: np.ndarray, differenceMask: np.ndarray, genBox: Box, worldPath: str):
    eprint("Placing output...")
    with loadLevel(worldPath) as world:
        for pos in tqdm([p for p in loop3D(genBox.size) if differenceMask[p.y, p.x, p.z]]):
            block = amulet.api.block.Block("minecraft", "red_concrete" if output[pos.y, pos.x, pos.z] else "glass")
            world.set_version_block(*(pos + genBox.offset), "minecraft:overworld", AMULET_MINECRAFT_VERSION, block)
        world.save()


def placeOutputEmbeddingGdpc(outputIndices: np.ndarray, palette: Palette, genBox: Box, editor: Editor, fix_1_16_palette: bool):
    eprint("Placing output...")
    with editor.pushTransform(genBox.offset):
        for pos in tqdm(list(loop3D(genBox.size))):
            blockTuple = palette[outputIndices[pos.y, pos.x, pos.z]]
            if fix_1_16_palette:
                if blockTuple[0] == "grass_path":
                    blockTuple = ("dirt_path", blockTuple[1])
                if blockTuple[0] == "cauldron" and "level" in blockTuple[1]:
                    blockTuple = ("water_cauldron", blockTuple[1])
            editor.placeBlock(pos, Block(blockTuple[0], blockTuple[1]))


def placeOutputEmbeddingAmulet(outputIndices: np.ndarray, palette: Palette, genBox: Box, worldPath: str, fix_1_16_palette: bool):
    eprint("Placing output...")
    with loadLevel(worldPath) as world:
        for pos in tqdm(list(loop3D(genBox.size))):
            blockTuple = palette[outputIndices[pos.y, pos.x, pos.z]]
            if fix_1_16_palette:
                if blockTuple[0] == "grass_path":
                    blockTuple = ("dirt_path", blockTuple[1])
                if blockTuple[0] == "cauldron" and "level" in blockTuple[1]:
                    blockTuple = ("water_cauldron", blockTuple[1])
            block = amulet.api.block.Block("minecraft", blockTuple[0], {key: amulet.from_snbt(repr(val)) for key, val in blockTuple[1].items()})
            world.set_version_block(*(pos + genBox.offset), "minecraft:overworld", AMULET_MINECRAFT_VERSION, block)
        world.save()


# ==================================================================================================


@cloup.group(context_settings={"show_default": True})
@cloup.pass_context
@cloup.option("--target", type=str, default="gdpc", help="Method and/or world to use as target. Must be \"gdpc\" (GDPC) or a world path (Amulet).")
@cloup.option("--offset", type=int, nargs=2, default=(0, 0), help="XZ offset of the area to generate in. Can only be used with the Amulet target.")
@cloup.option("--bottom-y", type=int, default=60, help="Bottom Y coordinate of the area to generate in")
@cloup.option("--size", type=int, nargs=3, default=(96, 64, 96), help="Size of the area to generate")
def cli(ctx: cloup.Context, target: str, offset: Tuple[int, int], bottom_y: int, size: Tuple[int, int, int]):
    ctx.obj["target"]   = target
    ctx.obj["offset"]   = ivec2(*offset)
    ctx.obj["bottom_y"] = bottom_y
    ctx.obj["size"]     = ivec3(*size)


@cloup.group()
@cloup.pass_context
@cloup.option("--remove-flow/--no-remove-flow", default=True, help="Whether to use \"remove-flow\" preprocessing.")
@cloup.option("--device", "device_name", type=str, default=defaultDeviceName, help=f'Device to use (e.g. "cuda:0", "cpu"). Defaults to "cuda" if available, otherwise "cpu". Current default: "{defaultDeviceName}".', show_default=False)
def binary(
    ctx:          cloup.Context,
    remove_flow:  bool,
    device_name:  str
):
    target:   str   = ctx.obj["target"]
    offsetXZ: ivec2 = ctx.obj["offset"]
    size:     ivec3 = ctx.obj["size"]
    bottomY:  int   = ctx.obj["bottom_y"]

    device = torch.device(device_name)
    ctx.obj["device"] = device

    ctx.obj["embeddingSize"] = 1 # Binary

    def runWithModel(model: Callable[[torch.Tensor], torch.Tensor]):
        if target == "gdpc":
            terrain, genBox, editor = getTerrainBinaryGdpc(size, bottomY, remove_flow)
        else:
            terrain, genBox = getTerrainBinaryAmulet(target, offsetXZ, size, bottomY, remove_flow)
        terrainTensor = (torch.from_numpy(terrain).float() * 2 - 1).unsqueeze(0).to(device)

        outputTensor = model(terrainTensor)

        eprint("Preparing output for placement...")
        output: np.ndarray = outputTensor[0].cpu().numpy() > 0
        differenceMask = output != terrain

        if target == "gdpc":
            placeOutputBinaryGdpc(output, differenceMask, genBox, editor)
        else:
            placeOutputBinaryAmulet(output, differenceMask, genBox, target)

    ctx.obj["runWithModel"] = runWithModel


@cloup.group()
@cloup.pass_context
@cloup.option_group(
    "Dataset and model settings",
    cloup.option("--dataset-dir", type=cloup.Path(exists=True, file_okay=False), required=False, help="Path to dataset."),
    cloup.option("--blocks-path", type=cloup.Path(exists=True, dir_okay=False), required=False, help="Path to blocks file."),
    cloup.option("--palette-path", type=cloup.Path(exists=True, dir_okay=False), required=False, help="Path to palette file."),
    cloup.option("--embeddings-path", type=cloup.Path(exists=True, dir_okay=False), required=True, help="Path to embeddings file."),
)
@cloup.constraint(cloup.constraints.require_one, ["dataset_dir", "palette_path"])
@cloup.constraint(cloup.constraints.If("freq_weight_exp", then=cloup.constraints.require_one), ["dataset_dir", "blocks_path"])
@cloup.option_group(
    "Pre- and post-processing",
    cloup.option("--prune-auto-states/--no-prune-auto-states", default=True, help="Use \"prune-auto-states\" preprocessing."),
    cloup.option("--remove-flow/--no-remove-flow", default=True, help="Use \"remove-flow\" preprocessing."),
    cloup.option("--1-16-palette/--no-1-16-palette", "fix_1_16_palette", default=True, help="Correct for a 1.16 palette"),
)
@cloup.option_group(
    "Output interpretation",
    cloup.option("--freq-weight-exp", type=float, help="Exponent for block frequency weighting. If not specified, no weighting will be used."),
    cloup.option("--max-block-index", type=int, help="Maximum block index to use. If not specified, all blocks will be used."),
)
@cloup.option("--device", "device_name", type=str, default=defaultDeviceName, help=f'Device to use (e.g. "cuda:0", "cpu"). Defaults to "cuda" if available, otherwise "cpu". Current default: "{defaultDeviceName}".', show_default=False)
def embedding(
    ctx:               cloup.Context,
    dataset_dir:       Optional[str],
    blocks_path:       Optional[str],
    palette_path:      Optional[str],
    embeddings_path:   str,
    prune_auto_states: bool,
    remove_flow:       bool,
    fix_1_16_palette:  bool,
    freq_weight_exp:   Optional[float],
    max_block_index:   Optional[int],
    device_name:       str
):
    target:   str   = ctx.obj["target"]
    offsetXZ: ivec2 = ctx.obj["offset"]
    size:     ivec3 = ctx.obj["size"]
    bottomY:  int   = ctx.obj["bottom_y"]

    if dataset_dir is not None:
        blocks_path  = f"{dataset_dir}/blocks.npy"
        palette_path = f"{dataset_dir}/palette.json"

    device = torch.device(device_name)
    ctx.obj["device"] = device

    eprint("Loading palette...")
    palette = loadPalette(palette_path)
    if max_block_index is None:
        max_block_index = len(palette)-1

    eprint("Loading embeddings...")
    embeddings = np.load(embeddings_path)
    assert embeddings.ndim == 2
    embeddingSize = embeddings.shape[1]

    ctx.obj["embeddingSize"] = embeddingSize

    def runWithModel(model: Callable[[torch.Tensor], torch.Tensor]):
        if freq_weight_exp is not None:
            eprint("Counting block frequencies...")
            blockArray = np.load(blocks_path, mmap_mode="r")
            blockTypeCounts = countBlockTypeCounts(
                blockArray,
                len(palette),
                progressCallback=tqdmProgressCallback(file=sys.stderr, dynamic_ncols=True)
            )
            blockTypeCounts = blockTypeCounts[:max_block_index+1]
            blockTypeWeights = (np.sum(blockTypeCounts) / blockTypeCounts) ** freq_weight_exp

        if target == "gdpc":
            terrain, genBox, editor = getTerrainEmbeddingGdpc(size, bottomY, palette, embeddings, prune_auto_states, remove_flow, fix_1_16_palette)
        else:
            terrain, genBox = getTerrainEmbeddingAmulet(target, offsetXZ, size, bottomY, palette, embeddings, prune_auto_states, remove_flow, fix_1_16_palette)
        terrainTensor = torch.from_numpy(terrain).unsqueeze(0).to(device)

        outputTensor = model(terrainTensor)

        eprint("Preparing output for placement...")
        output = outputTensor.squeeze(0).cpu().numpy()
        outputIndices = np.zeros((size.y, size.x, size.z), dtype=int)
        for y in tqdm(range(size.y)): # Batch by y to not exceed memory
            differences = embeddings[None, None, :max_block_index+1, :] - np.moveaxis(output, 0, -1)[y, :, :, None, :]
            distances = np.linalg.norm(differences, axis=-1)
            if freq_weight_exp is not None:
                distances *= blockTypeWeights
            outputIndices[y, :, :] = np.argmin(distances, axis=-1)

        if target == "gdpc":
            placeOutputEmbeddingGdpc(outputIndices, palette, genBox, editor, fix_1_16_palette)
        else:
            placeOutputEmbeddingAmulet(outputIndices, palette, genBox, target, fix_1_16_palette)

    ctx.obj["runWithModel"] = runWithModel


@cloup.command()
@cloup.pass_context
@cloup.option("--weights-path", type=cloup.Path(exists=True, dir_okay=False), required=True, help="Path to generator weights.")
def vox2vox(
    ctx:          cloup.Context,
    weights_path: str
):
    device        = ctx.obj["device"]
    embeddingSize = ctx.obj["embeddingSize"]

    eprint("Loading model...")
    model = Vox2voxGenerator(embeddingSize, embeddingSize).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    def modelCallback(terrainTensor: torch.Tensor):
        eprint("Applying model...")
        with torch.no_grad():
            return model(terrainTensor) # pylint: disable=not-callable

    ctx.obj["runWithModel"](modelCallback)


@cloup.command()
@cloup.pass_context
@cloup.option_group(
    "Model hyperparameters",
    cloup.option("--scales", "scale_factor", type=str,   default="0.25,0.5,0.75,1.0", help="Scales, from coarse to fine."),
)
@cloup.option_group(
    "Model weights",
    cloup.option("--scales-dir",    type=cloup.Path(exists=True, file_okay=False), help="Dir containing scale output dirs."),
    cloup.option("--weights-path", type=cloup.Path(exists=True, dir_okay=False), multiple=True, help="Path to generator weights. Must be passed as many times as there are scales, from course to fine."),
    constraint=cloup.constraints.RequireExactly(1)
)
def vox2vox_multiscale(
    ctx:          cloup.Context,
    scale_factor: str,
    scales_dir:   Optional[str],
    weights_path: Optional[List[str]],
):
    device        = ctx.obj["device"]
    embeddingSize = ctx.obj["embeddingSize"]

    eprint("Loading models...")

    scaleFactors = [float(scale) for scale in scale_factor.split(",")]

    if len(scaleFactors) == 0:
        eprint(ERROR_PREFIX + "There must be at least one scale")
        sys.exit(1)

    if scales_dir is not None:
        weights_path = []
        for scale in range(len(scaleFactors)):
            checkpointDir = f"{scales_dir}/scale-{scale}/checkpoints"
            if not os.path.isdir(checkpointDir):
                eprint(ERROR_PREFIX + f"Checkpoint dir {checkpointDir} is missing")
                sys.exit(1)
            lastCheckpoint = max(int(epochDir.split("-")[-1]) for epochDir in os.listdir(checkpointDir))
            weights_path.append(f"{checkpointDir}/epoch-{lastCheckpoint}/generator.pt")

    if len(weights_path) != len(scaleFactors):
        eprint(ERROR_PREFIX + f"There must be as many weight paths as there are scales ({len(scaleFactors)})")
        sys.exit(1)

    models = (
        [Vox2voxGenerator(embeddingSize,     embeddingSize).to(device)] +
        [Vox2voxGenerator(embeddingSize * 2, embeddingSize).to(device) for _ in range(len(scaleFactors) - 1)]
    )
    for scale in range(len(scaleFactors)):
        models[scale].load_state_dict(torch.load(weights_path[scale], map_location=device))

    def modelCallback(terrainTensor: torch.Tensor):
        originalSize = terrainTensor.shape[-3:]
        scaleSizes = [
            [int(scaleFactor * originalSize[axis]) for axis in range(3)]
            for scaleFactor in scaleFactors
        ]

        with torch.no_grad():
            eprint("Applying model for scale 0...")
            downscaledTerrain = torch.nn.functional.interpolate(terrainTensor, size=scaleSizes[0], mode="trilinear", align_corners=False)
            output = models[0](downscaledTerrain)
            for scale in range(1, len(scaleFactors)):
                eprint(f"Applying model for scale {scale}...")
                downscaledTerrain  = torch.nn.functional.interpolate(terrainTensor, size=scaleSizes[scale], mode="trilinear", align_corners=False)
                upscaledPrevOutput = torch.nn.functional.interpolate(output,        size=scaleSizes[scale], mode="trilinear", align_corners=False)
                refinement = models[scale](torch.concatenate([downscaledTerrain, upscaledPrevOutput], dim=1))
                output = upscaledPrevOutput + refinement

        return output

    ctx.obj["runWithModel"](modelCallback)


def main():
    cli.add_command(binary)
    cli.add_command(embedding)
    binary.add_command(vox2vox)
    embedding.add_command(vox2vox)
    embedding.add_command(vox2vox_multiscale)
    cli(obj={}) # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    main()
