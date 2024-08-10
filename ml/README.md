# Machine learning code

[Link to main readme](../README.md)

This directory contains our dataset preprocessing tool and our machine learning
code.

## Setup

The only setup needed is to install Python >= 3.7 and install the dependencies
listed in `requirements.txt` (we recommend to use a virtual environment). The
dependencies can be installed from the command line with following command:
```
python3 -m pip install -r requirements.txt
```

## Running the programs

The `src/` directory contains various executable programs:

- `src/preprocess_datasets.py`:\
  The dataset preprocessing tool.
- `src/train.py`:\
  Trains a terrain-adaptive model on a "stacked" dataset (which can be created
  from the datasets of our dataset generation system by using our dataset
  preprocessing tool).
- `src/train_block2vec.py`:\
  Generates block2vec embeddings for a stacked dataset, which are needed to
  train a categorial model with `src/train.py`.
- `src/run.py`:\
  Applies a trained model to new initial terrain.
- `src/visualize_embeddings.py`:\
  Visualizes block2vec embeddings using various diagrams.

All of the above programs provide fairly detailed documentation when you pass
the `--help` flag. For example, `src/preprocess_datasets.py --help`. You can
get more info about a subcommand by putting `--help` after the subcommand:
`src/preprocess_datasets.py stack --help`. Do note that the documentation
assumes that you have read and understood our paper, and in some cases, the
Master's thesis that is cited therein.
