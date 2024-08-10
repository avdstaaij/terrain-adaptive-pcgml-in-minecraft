# Dataset generation system

[Link to main readme](../README.md)

This directory contains our dataset generation system.


## Setup

Some setup needs to be performed before the system can be used.

This directory contains the script `setup.sh`, which automates steps 3-11
listed below. However, caution is advised when using this script. It
downloads and executes several files from the internet, automates the
installation of Minecraft Forge (which is discouraged because it avoids their
ads) and auto-accepts the Minecraft EULA. Do not run it unless you fully
understand what it does. We recommend to perform setup manually the first time,
and to only use the script for repeated installations.

Setup steps:

1. Install Python >= 3.7 and install the dependencies listed in
   `requirements.txt` (we recommend to use a virtual environment). The
   dependencies can be installed from the command line with following
   command:
   ```
   python3 -m pip install -r requirements.txt
   ```

2. Install the Java JRE for Java 8 or higher.

3. Create a directory named `server` in `minecraft-server`.
   The `minecraft-server` directory should then have two subdirectories:
   `resources` and `server`.

4. Download the installer for [Minecraft Forge 1.16.5](https://files.minecraftforge.net/net/minecraftforge/forge/index_1.16.5.html)
   and place it in `minecraft-server/server`.
   The dataset generation system has been tested with Forge 1.16.5-36.2.39,
   but will likely work with newer 1.16.5-something versions as well.

5. Install Minecraft Forge 1.16.5 in `minecraft-server/server` using the
   installer. This can be done from the command-line by running the following
   command from inside `minecraft-server/server` (your installer filename may
   be slightly different if you got a newer version of Forge):
   ```
   java -jar "forge-1.16.5-36.2.39-installer.jar" --installServer
   ```
   If the installation was succesful, `minecraft-server/server` should now
   contain several new files, such as `minecraft_server.1.16.5.jar`.

6. Run the Forge server once. This can be done from the command-line by running
   the following command from inside `minecraft-server/server` (again, the jar
   filename may be slightly different if you got a newer version of Forge):
   ```
   java -jar -Xmx2G -Xms2G forge-1.16.5-36.2.39.jar -nogui
   ```
   The server should automatically terminate and prompt you to accept the
   Minecraft EULA.

7. Read the Minecraft EULA and accept it by changing `eula=false` to `eula=true`
   in `minecraft-server/server/eula.txt`.

8. Copy `minecraft-server/resources/server.properties` to
   `minecraft-server/server/`.
   Overwrite `minecraft-server/server/server.properties` if it's already there.

9. Install the mod *GDMC-HTTP 0.4.2* by copying
   `minecraft-server/resources/gdmc-http/gdmchttp-0.4.2.jar` to
   `minecraft-server/server/mods`.

10. Install the mod [*Chunk Pregenerator 1.16-3.5.1*](https://www.curseforge.com/minecraft/mc-mods/chunkpregenerator/files/4087387)
    by downloading the jar file and placing it in `minecraft-server/server/mods`.

11. Finally, execute `generators/setup-generators.sh`.


## Running the program

To run the program, execute `src/generate_dataset.py`.
For a full listing of the command-line arguments and options, run
`src/generate_dataset --help`.

The most common usage pattern for the program is to create two datasets, one
with samples of natural terrain and one with samples of settlements, such that
that the settlement samples are based on the same natural terrain as the natural
terrain samples. The two datasets can then later be combined into a single
"stacked" dataset using our preprocessing tool. We will give an example of how
to create two datasets like this.

First, to create a dataset of natural terrain samples:
```
src/generate_dataset.py "/path/to/output/terrain" "none" 0 0 32 32 --build-length 4 --y-min 0 --y-max 255
```
This will create a dataset in "/path/to/output/terrain" of the "none" generator
(just natural terrain), with 32x32 samples taken that are 4x4 chunks in size,
from the square world area starting at chunk (0,0), and extending 4*32 chunks in
the X and Z directions. The samples will contain all blocks start from Y=0 up to
Y=255 (tightening these bounds will yield a more space-efficient dataset).

Second, to create a dataset of settlement samples using the same terrain:
```
src/generate_dataset.py "/path/to/output/settlements" "mikes_angels" 0 0 32 32 --build-length 4 --y-min 0 --y-max 255 --initial-world "/path/to/output/terrain/world" --initial-palette "/path/to/output/terrain/palette.json" --no-chunk-regen
```
This will create a dataset in "/path/to/output/settlements" of the
"mikes_angels" generator (to see the list of generator names, enter an invalid
one), with samples taken at the same positions as for the terrain dataset. The
`--initial-world` and `--no-chunk-regen` options cause the program to not
pregenerate terrain on this run, but instead use the existing terrain of the
passed world, which significantly improves performance. The `--initial-palette`
option makes the program initialize the dataset's block palette with that of the
terrain dataset. This ensures that a block type that already existed in the
initial terrain dataset will have the same palette index in the settlement
dataset. Our preprocessing tool can also take care of "misaligned" palettes when
stacking datasets, but using this flag makes that process more efficient.

If instead of two datasets, you only want one dataset of settlement data, you
could use a command as follows:
```
src/generate_dataset.py "/path/to/output" "mikes_angels" 0 0 32 32 --build-length 4 --y-min 0 --y-max 255
```

If you generate a large dataset, the program will periodically save a checkpoint
from which it can continue if it's interrupted or if it crashes. To continue a
generation task from the last checkpoint, simply repeat the same command you
used to start the task. The program should then detect the in-progress dataset
and ask you if you wish to continue where it left off.


## Adding new settlement generators

Adding new settlement generators requires some editing of the code.
To add a new generator, you need to add an entry to the `GENERATORS` dictionary
in `src/generate_dataset.py`. The key should be the name of the generator
(used to select this generator in the command-line interface) and the value
should be a function with the following signature (or a strictly more lenient
signature):
```python
def customGenerator(server: MinecraftServer, buildArea: Box, timeoutSeconds: int)
```
The function should run your new settlement generator on the passed server and
in the passed build area, and should abort early if `timeoutSeconds` second have
passed.

Examples of generator functions for generators based on the GDMC HTTP interface
can be found in `src/lib/generators.py`.
