#!/usr/bin/env python3

import sys

import cloup
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt

from lib.util import eprint
from lib.constants import ERROR_PREFIX
from lib.palette_tools import Palette, blockTupleToString, loadPalette


def parse_only_arg(only: str, palette: Palette) -> np.ndarray:
    def parse_entry(entry: str):
        if entry.isnumeric():
            return int(entry)
        if entry.startswith("-") and entry[1:].isnumeric():
            return len(palette) + int(entry)
        for i, block in enumerate(palette):
            if block[0] == entry: # TODO: block state support?
                return i
        raise ValueError()

    if ":" in only:
        start_entry, end_entry = only.split(":")
        start_index = 0            if start_entry == "" else parse_entry(start_entry)
        end_index   = len(palette) if end_entry   == "" else parse_entry(end_entry)
        if start_index > end_index:
            raise ValueError()
        return np.arange(start_index, end_index)

    else:
        return np.array([parse_entry(entry) for entry in only.split(",")])


@cloup.group(context_settings={"show_default": True})
@cloup.option("--embedding-path", type=cloup.Path(exists=True), required=True)
@cloup.option("--palette-path",   type=cloup.Path(exists=True), required=True)
@cloup.option("--only",           type=str)
@cloup.pass_context
def cli(ctx: cloup.Context, embedding_path: str, palette_path: str, only: str):
    """Visualize block embeddings"""

    embeddings = np.load(embedding_path)
    palette    = loadPalette(palette_path)

    if len(embeddings) != len(palette):
        eprint(ERROR_PREFIX + f"Number of embeddings ({len(embeddings)}) and palette entries ({len(palette)}) do not match.")
        sys.exit(1)

    if only is not None:
        indices = parse_only_arg(only, palette)
        embeddings = embeddings[indices]
        palette    = [palette[i] for i in indices]

    ctx.obj["embeddings"] = embeddings
    ctx.obj["palette"]    = palette


@cli.command()
@cloup.option("--labels/--no-labels", "draw_labels", default=False)
@cloup.pass_context
def distance_matrix(ctx: cloup.Context, draw_labels: bool):
    """Show confusion matrix"""

    embeddings: np.ndarray = ctx.obj["embeddings"]
    palette:    Palette    = ctx.obj["palette"]

    distances = np.linalg.norm(embeddings[:,None] - embeddings[None,:], axis=2)
    labels    = np.array([blockTupleToString(block) for block in palette])

    plt.rcParams.update({'font.size': 6})
    fig, ax = plt.subplots(figsize = (1*len(palette), 1*len(palette)) if draw_labels else None)
    display = ConfusionMatrixDisplay(distances, display_labels=labels)
    display.plot(ax=ax, cmap="Blues", include_values=False, xticks_rotation="vertical")
    display.ax_.set_xlabel(None)
    display.ax_.set_ylabel(None)
    display.ax_.tick_params(axis=u'both', which=u'both',length=0)
    if not draw_labels:
        display.ax_.set_xticks([])
        display.ax_.set_yticks([])
        plt.tight_layout()
    plt.show()


@cli.command()
@cloup.option("--n-neighbors", type=int,   default=5)
@cloup.option("--min-dist",    type=float, default=0.3)
@cloup.option("--metric",      type=str,   default="euclidean")
@cloup.option("--plotter",     type=cloup.Choice(["umap", "matplotlib"]), default="matplotlib")
@cloup.pass_context
def umap_2d(ctx: cloup.Context, n_neighbors: int, min_dist: float, metric: str, plotter: str):
    "Show 2D UMAP dimensionality reduction plot"

    embeddings: np.ndarray = ctx.obj["embeddings"]
    palette:    Palette    = ctx.obj["palette"]

    import umap # pylint: disable=import-outside-toplevel

    eprint("Computing UMAP reduction...")
    mapper = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric).fit(embeddings)
    eprint("Done.")

    if plotter == "umap":
        import umap.plot # pylint: disable=import-outside-toplevel

        hover_data = pd.DataFrame({"block": [blockTupleToString(block) for block in palette]})

        fig = umap.plot.interactive(mapper, hover_data=hover_data, point_size=8)
        fig.height = 800
        fig.width  = 800
        fig.sizing_mode = "scale_height"
        umap.plot.show(fig)

    elif plotter == "matplotlib":
        reduced_embeddings = mapper.transform(embeddings)

        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["cmr10"],
            "font.size": 12,
            "text.usetex": True,
            "axes.formatter.use_mathtext": True,
        })

        fig, ax = plt.subplots(figsize=(6,6))

        ax.scatter(reduced_embeddings[:,0], reduced_embeddings[:,1], s=20, c="#187bcd")

        xlength = ax.get_xlim()[1] - ax.get_xlim()[0]
        ylength = ax.get_ylim()[1] - ax.get_ylim()[0]

        for i, block in enumerate(palette):
            text = ax.annotate(blockTupleToString(block), (reduced_embeddings[i,0] + 0.01*xlength, reduced_embeddings[i,1] + 0.01*ylength))
            text.set_rotation(0)

        ax.set_xlim(right=ax.get_xlim()[1] + 0.15*xlength) # Add extra space for the captions

        ax.set_aspect(1./ax.get_data_ratio(), adjustable='box')
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.show()


def main():
    cli(obj={}) # pylint: disable=no-value-for-parameter


if __name__ == '__main__':
    main()
