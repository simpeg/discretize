#!/bin/bash
set -ex #echo on and exit if any line fails

echo "arch is $ARCH"
if [[ $ARCH == "X64" ]]; then
  MINICONDA_ARCH_LABEL="x86_64"
else
  MINICONDA_ARCH_LABEL="arm64"
fi
echo $MINICONDA_ARCH_LABEL
mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-$MINICONDA_ARCH_LABEL.sh -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
echo "##vso[task.setvariable variable=CONDA;]${HOME}/miniconda3"
