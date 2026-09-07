#!/bin/bash
set -ex #echo on and exit if any line fails

# TF_BUILD is set to True on azure pipelines.
is_azure=$(echo "${TF_BUILD:-false}" | tr '[:upper:]' '[:lower:]')
do_doc=$(echo "${DOC_BUILD:-false}" | tr '[:upper:]' '[:lower:]')
is_free_threaded=$(echo "${PYTHON_FREETHREADING:-false}" | tr '[:upper:]' '[:lower:]')
is_rc=$(echo "${PYTHON_RELEASE_CANDIDATE:-false}" | tr '[:upper:]' '[:lower:]')
is_bare=$(echo "${ENVIRON_BARE:-false}" | tr '[:upper:]' '[:lower:]')

if [[ "$is_azure" == "true" ]]; then
  if [[ "$do_doc" == "true" ]]; then
    .ci/setup_headless_display.sh
  fi
fi

# resolve python spec: freethreaded uses a "t" suffix; is_rc is informational
# here -- PYTHON_VERSION for the rc matrix job is itself an exact rc
# specifier (e.g. "3.15.0rc2") set in test.yml, since uv (unlike
# conda-forge's python_rc label) needs an exact pin rather than a
# "latest rc" moving target.
py_spec="$PYTHON_VERSION"
if [[ "$is_free_threaded" == "true" ]]; then
  py_spec="${py_spec}t"
fi

if [[ "$is_bare" == "true" ]]; then
  extras_csv="test,build"
elif [[ "$do_doc" == "true" ]]; then
  extras_csv="test,doc,build"
else
  extras_csv="test,all,build"
fi

# pytest is ran with its import mode set to importlib from pyproject.toml's
# [tool.pytest.ini_options], so we do not need an editable install here.
# --no-build-package on these two turns a future wheel gap (numpy/scipy
# dropped cp313t wheels at 2.5.0/1.18.0, hence no 3.13t testing -- see
# dev-prototypes/uv-ci-migration-notes.md) into a clear resolution error
# instead of a silent, doomed-to-fail source build.
uv sync --python "$py_spec" --no-editable --extra ${extras_csv//,/ --extra } \
  --no-build-package numpy --no-build-package scipy \
  --config-settings=setup-args="--vsenv"

if [[ -f .venv/bin/python ]]; then
  VENV_PY="$(pwd)/.venv/bin/python"
else
  VENV_PY="$(pwd)/.venv/Scripts/python.exe"
fi

if [[ "$is_azure" == "true" ]]; then
  uv pip install --python "$VENV_PY" pytest-azurepipelines
fi

echo "Installed packages:"
uv pip list --python "$VENV_PY"

echo "Installed discretize version:"
# run from outside the repo root, so this doesn't hit the same
# discretize/-shadows-the-installed-package issue --import-mode=importlib
# fixes for pytest specifically (plain `python -c` has no such flag).
(cd / && "$VENV_PY" -c "import discretize; print(discretize.__version__)")
