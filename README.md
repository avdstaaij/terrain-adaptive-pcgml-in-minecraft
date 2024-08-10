# Terrain-adaptive PCG in Minecraft

This is the code repository for *Terrain-adaptive PCG in Minecraft* by
Arthur van der Staaij, Mike Preuss and Christoph Salge, presented at IEEE
Conference on Games 2024.

Citation info will be added here once the proceedings become available.

If you're looking for our datasets, those are available
[here](https://zenodo.org/records/11507023).


## Structure

The repository is comprised of two parts:

1. The *data generation system* (`data-gen/` directory).\
   [Link to data-gen readme](/data-gen/README.md)

   This is an automated system that manages a Minecraft server, pregenerates the
   natural terrain, executes (settlement) generator algorithms, and stores the
   resulting before- and after-samples in an efficient format.

   This is the code that you'll need if you intend to generate new
   terrain-adaptive Minecraft datasets.


2. the *machine learning code* (`ml/` directory.)\
   [Link to ml readme](/ml/README.md)

   Here, you'll find our data preprocessing system as well as all the pytorch
   code we used to train models on our preprocessed terrain-adaptive Minecraft
   data.

   This is the code you'll need if you intend to preprocess datasets you made
   with the data generation system or if you want to view the details of our
   training methodology.
