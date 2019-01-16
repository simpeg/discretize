#!/bin/bash
#
# Originally from: https://github.com/fatiando/continuous-integration/blob/master/travis/setup-miniconda.sh
#
# Licensed under the BSD 3-Clause: https://github.com/fatiando/continuous-integration/blob/master/LICENSE.txt
#
#
# Installs Miniconda (Python 3), set's the PATH, updates conda, configures to
# use conda-forge, creates a "testing" environment and activates it, and
# installs dependencies from a requirements file (given in CONDA_REQUIREMENTS)

# To return a failure if any commands inside fail
set -e

miniconda_url="http://repo.continuum.io/miniconda"

if [ "$TRAVIS_OS_NAME" == "osx" ]; then
    miniconda_file=Miniconda3-latest-MacOSX-x86_64.sh
else
    miniconda_file=Miniconda3-latest-Linux-x86_64.sh
fi

conda_prefix=$HOME/miniconda

# Download and install miniconda
echo ""
echo "Downloading Miniconda from $miniconda_url/$miniconda_file"
echo "========================================================================"
wget $miniconda_url/$miniconda_file -O miniconda.sh
bash miniconda.sh -b -p $conda_prefix

# Add it to the path
export PATH="$conda_prefix/bin:$PATH"

echo ""
echo "Configuring conda"
echo "========================================================================"
# Don't change the prompt or request user input
conda config --set always_yes yes --set changeps1 no

# Add conda-forge to the top of the channel list
conda config --prepend channels conda-forge
conda config --remove channels defaults
# Add an extra channel that may be required
if [[ ! -z $CONDA_EXTRA_CHANNEL ]]; then
    conda config --append channels $CONDA_EXTRA_CHANNEL
fi

# Display all configuration options for diagnosis
conda config --show

# Update conda to the latest version
echo ""
echo "Updating conda"
echo "========================================================================"
conda update --quiet conda

# Create and activate an environment for testing
echo ""
echo "Creating the 'testing' environment with python=$PYTHON"
echo "========================================================================"
conda create --quiet --name testing python=$PYTHON pip
source activate testing

# Install dependencies
echo ""
echo "Installing conda dependencies"
echo "========================================================================"
requirements_file=full-conda-requirements.txt
if [ ! -z "$CONDA_REQUIREMENTS" ]; then
    if [ -f "$CONDA_REQUIREMENTS" ]; then
        echo "Capturing dependencies from $CONDA_REQUIREMENTS"
        cat $CONDA_REQUIREMENTS >> $requirements_file
    else
        echo $CONDA_INSTALL_EXTRA | xargs -n 1 >> $requirements_file
    fi
fi
if [ ! -z "$CONDA_REQUIREMENTS_DEV" ]; then
    echo "Capturing dependencies from $CONDA_REQUIREMENTS_DEV"
    cat $CONDA_REQUIREMENTS_DEV >> $requirements_file
fi
if [ ! -z "$CONDA_INSTALL_EXTRA" ]; then
    # Use xargs to print one argument per line
    echo $CONDA_INSTALL_EXTRA | xargs -n 1 >> $requirements_file
fi
if [ -f $requirements_file ]; then
    echo "Installing collected dependencies:"
    cat $requirements_file
    conda install --quiet --file $requirements_file python=$PYTHON=
else
    echo "No requirements defined."
fi

echo ""
echo "Installing pypi dependencies"
echo "========================================================================"
if [ ! -z "$PYPI_REQUIREMENTS" ]; then
    pip install -r $PYPI_REQUIREMENTS
fi

if [ ! -z "$PYPI_REQUIREMENTS_DEV" ]; then
    pip install -r $PYPI_REQUIREMENTS_DEV
fi

# Make sure that this is the correct Python version. Sometimes conda will try to upgrade
# Python itself if a dependency doesn't support a version. We're enforcing this in the
# conda install above but it's best to check.
echo ""
echo "Check that Python really is $PYTHON"
python -c "import sys; assert sys.version_info[:2] == tuple(int(i) for i in '$PYTHON'.split('.'))"

# Workaround for https://github.com/travis-ci/travis-ci/issues/6522
# Turn off exit on failure.
set +e
