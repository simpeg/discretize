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

# Resolve+install the selected extras only, then do one explicit editable
# build of discretize below. Must stay editable (a plain install is shadowed
# by the source tree when pytest runs from the repo root) and must stay a
# two-step install with --no-build-isolation below (a single-call install
# builds in an ephemeral env that breaks the editable rebuild-on-import
# check once it's cleaned up). See dev-prototypes/uv-ci-migration-notes.md.
# --config-settings must match exactly between both calls, or meson's
# --reconfigure of the shared build dir fails on the second call.
uv sync --python "$py_spec" --no-install-project $extra_flags \
  --config-settings=setup-args="--vsenv"

if [[ -f .venv/bin/python ]]; then
  VENV_PY=.venv/bin/python
else
  VENV_PY=.venv/Scripts/python.exe
fi

uv pip install --python "$VENV_PY" --no-build-isolation --editable . \
  --config-settings=setup-args="--vsenv"

if [[ "$is_azure" == "true" ]]; then
  uv pip install --python "$VENV_PY" pytest-azurepipelines
fi

echo "Installed packages:"
uv pip list --python "$VENV_PY"

echo "Installed discretize version:"
"$VENV_PY" -c "import discretize; print(discretize.__version__)"
