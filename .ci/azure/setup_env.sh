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
  extra_flags="--extra test --extra build"
elif [[ "$do_doc" == "true" ]]; then
  extra_flags="--extra test --extra doc --extra build"
else
  extra_flags="--extra test --extra all --extra build"
fi

# numpy/scipy dropped cp313t wheels starting at 2.5.0/1.18.0 (cp314t is
# still published); without this, uv resolves the latest version anyway and
# falls back to a source build, which fails here (no OpenBLAS toolchain).
# A temp constraints file (not a pyproject.toml extra) keeps this CI-only
# workaround out of the project's own dependency declarations.
if [[ "$is_free_threaded" == "true" && "$PYTHON_VERSION" == 3.13* ]]; then
  constraints_file=$(mktemp)
  printf 'numpy<2.5\nscipy<1.18\n' > "$constraints_file"
  export UV_CONSTRAINT="$constraints_file"
fi

# pytest is ran with its import mode set to importlib from pyproject.toml's
# [tool.pytest.ini_options], so we do not need an editable install here.
# --no-build-package on these two turns any future wheel gap into a clear
# resolution error instead of a silent, doomed-to-fail source build.
uv sync --python "$py_spec" --no-editable $extra_flags \
  --no-build-package numpy --no-build-package scipy \
  --config-settings=setup-args="--vsenv"

if [[ -n "${constraints_file:-}" ]]; then
  rm -f "$constraints_file"
fi

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
