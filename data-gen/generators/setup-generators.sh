#!/bin/bash

set -e # Quit on error
cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")" >/dev/null
CMD="$(basename "$0")"

####################################################################################################

# NOTE: To get a Google Drive download link, use the template of the ones below but replace the id
# NOTE: Use "safe" names, becuase virtualenv might break otherwise.
declare -A GDMC_GENERATOR_URLS=(
	["Mikes-Angels"]="https://drive.google.com/uc?export=download&id=1aCM8Veh8xOFu3Xd42DfQO2l1yGxtjRpT"
)

####################################################################################################

shopt -s dotglob

cd "code"

mkdir -p "gdmc-2022"

pushd "gdmc-2022" >/dev/null

	# Download generators
	for team in "${!GDMC_GENERATOR_URLS[@]}"; do
		if [[ ! -d "$team" ]]; then
			mkdir -p "$team"
			pushd "$team" >/dev/null
				wget "${GDMC_GENERATOR_URLS[$team]}" -O "generator.zip"
				unzip "generator.zip"
				rm "generator.zip"
			popd >/dev/null
		fi
	done

	# Mike's Angels
	pushd "Mikes-Angels" >/dev/null
		if [[ ! -d ".venv" ]]; then
			mv "Medieval City Generator"/* .
			rmdir "Medieval City Generator"
			# Freeze requirements. Notably, astar 0.94 breaks the generator.
			echo -e "astar==0.93\ngdpc==5.0.2\nnumpy==1.19.3\nPyGLM==2.6.0\npyglm-typing==0.2.1\nrequests==2.22.0\nscikit-image==0.19.3\nscipy==1.9.3" > "requirements.txt"
		fi
	popd >/dev/null

	# Mike's Angels - Patches
	for dir in "Mikes-Angels-Wall" "Mikes-Angels-Roads" "Mikes-Angels-Roads-Wall"; do
		if [[ ! -d "$dir" ]]; then
			cp -r "Mikes-Angels" "$dir"
			pushd "$dir" >/dev/null
				patch -p0 < "../../../patches/${dir}.diff"
			popd >/dev/null
		fi
	done

popd >/dev/null

# Create Python virtualenvs
for dir in "gdmc-2022/Mikes-Angels" "gdmc-2022/Mikes-Angels-Roads" "gdmc-2022/Mikes-Angels-Wall" "gdmc-2022/Mikes-Angels-Roads-Wall" "ring" "ring-adaptive"; do
	pushd "$dir" >/dev/null
		if [[ ! -d ".venv" ]]; then
			python3 -m virtualenv -p $(which python3) ".venv"
			(source ".venv/bin/activate" && pip install -r "requirements.txt")
		fi
	popd >/dev/null
done
