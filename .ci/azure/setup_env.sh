#!/bin/bash
set -ex #echo on and exit if any line fails

# TF_BUILD is set to True on azure pipelines.
is_azure=$(echo "${TF_BUILD:-false}" | tr '[:upper:]' '[:lower:]')
do_doc=$(echo "${DOC_BUILD:-false}" | tr '[:upper:]' '[:lower:]')
is_free_threaded=$(echo "${PYTHON_FREETHREADING:-false}" | tr '[:upper:]' '[:lower:]')
is_rc=$(echo "${PYTHON_RELEASE_CANDIDATE:-false}" | tr '[:upper:]' '[:lower:]')

if ${is_azure}
then
  if ${do_doc}
  then
    .ci/setup_headless_display.sh
  fi
fi

if ${is_free_threaded}
then
  cp .ci/environment_test_bare.yml environment_test_with_pyversion.yml
  echo "  - python-freethreading="$PYTHON_VERSION >> environment_test_with_pyversion.yml
else
  cp .ci/environment_test.yml environment_test_with_pyversion.yml
  echo "  - python="$PYTHON_VERSION >> environment_test_with_pyversion.yml
fi

if ${is_rc}
then
  sed -i '/^channels:/a\  - conda-forge/label/python_rc' environment_test_with_pyversion.yml
fi
conda env create --file environment_test_with_pyversion.yml
rm environment_test_with_pyversion.yml

if ${is_azure}
then
  source activate discretize-test
  pip install pytest-azurepipelines
else
  conda activate discretize-test
fi

# The --vsenv config setting will prefer msvc compilers on windows.
# but will do nothing on mac and linux.
pip install --no-build-isolation --editable . --config-settings=setup-args="--vsenv"

echo "Conda Environment:"
conda list

echo "Installed discretize version:"
python -c "import discretize; print(discretize.__version__)"