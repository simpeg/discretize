#!/bin/bash
set -ex #echo on and exit if any line fails

# TF_BUILD is set to True on azure pipelines.
is_azure=$(echo "${TF_BUILD:-false}" | tr '[:upper:]' '[:lower:]')
do_doc=$(echo "${DOC_BUILD:-false}" | tr '[:upper:]' '[:lower:]')

if ${is_azure}
then
  if ${do_doc}
  then
    .ci/setup_headless_display.sh
  fi
fi

env_name="discretize-test"

conda create -n $env_name python=$PYTHON_VERSION
conda env update --name $env_name --file .ci/environment_test.yml --prune

if ${is_azure}
then
  source activate $env_name
  pip install pytest-azurepipelines
else
  conda activate $env_name
fi

# The --vsenv config setting will prefer msvc compilers on windows.
# but will do nothing on mac and linux.
pip install --no-build-isolation --editable . --config-settings=setup-args="--vsenv"

echo "Conda Environment:"
conda list

echo "Installed discretize version:"
python -c "import discretize; print(discretize.__version__)"