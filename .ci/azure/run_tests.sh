#!/bin/bash
set -ex #echo on and exit if any line fails

# TF_BUILD is set to True on azure pipelines.
is_azure=$(echo "${TF_BUILD:-false}" | tr '[:upper:]' '[:lower:]')
do_doc=$(echo "${DOC_BUILD:-false}" | tr '[:upper:]' '[:lower:]')
do_cov=$(echo "${COVERAGE:-false}" | tr '[:upper:]' '[:lower:]')

test_args=""

source activate discretize-test

if ${do_doc}
then
  if ${is_azure}
  then
    .ci/setup_headless_display.sh
  fi
fi
if ${do_cov}
then
  echo "Testing with coverage"
  test_args="--cov --cov-config=pyproject.toml $test_args"
fi

pytest -vv $test_args

if ${do_cov}
then
  coverage xml
fi

