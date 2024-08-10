#!/bin/bash


# WARNING: This script installs and executes files from the internet, automates the installation of
# Minecraft Forge (which is discouraged because it avoids their ads) and auto-accepts the Minecraft
# EULA. Make sure you fully understand what it does before running it.
# We recommend to perform setup manually the first time, and to only use this script for repeated
# installations. See README.md (in this directory) for manual setup steps.


set -e # Quit on error
SCRIPT_DIR="$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")"
CMD="$(basename "$0")"

####################################################################################################

MINECRAFT_SERVER_DIR="$SCRIPT_DIR/minecraft-server"

FORGE_URL="https://maven.minecraftforge.net/net/minecraftforge/forge/1.16.5-36.2.39/forge-1.16.5-36.2.39-installer.jar"
FORGE_JAR_FILENAME="forge-1.16.5-36.2.39.jar"

CHUNK_PREGENERATOR_URL="https://mediafilez.forgecdn.net/files/4087/387/Chunk+Pregenerator-1.16-3.5.1.jar"

####################################################################################################

TEMP_DIR="$(mktemp -d)"

cd "$SCRIPT_DIR" > /dev/null

# Set up Minecraft server
if [[ ! -d "$MINECRAFT_SERVER_DIR/server" ]]; then
	mkdir -p "$MINECRAFT_SERVER_DIR/server"

	pushd "$TEMP_DIR" > /dev/null
		# Download forge installer
		# NOTE: Forge discourages automating the installation process because it bypasses their ads.
		wget "$FORGE_URL" -O "forge-installer.jar"
	popd
	pushd "$MINECRAFT_SERVER_DIR/server" > /dev/null
		# Initialize forge server
		java -jar "$TEMP_DIR/forge-installer.jar" --installServer
		java -jar -Xmx2G -Xms2G "$FORGE_JAR_FILENAME" -nogui
		sed -i 's/eula=false/eula=true/' eula.txt # Accept Minecraft EULA
		cp -T "../resources/server.properties" "server.properties"

		# Install mods
		pushd "mods" >/dev/null
			# GDMC-HTTP
			cp -t "." "../../resources/gdmc-http/gdmchttp-0.4.2.jar"
			# Chunk pregenerator
			wget "$CHUNK_PREGENERATOR_URL"
		popd
	popd
fi

# Set up generators
./generators/setup-generators.sh

rm -rf "$TEMP_DIR"
