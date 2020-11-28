#!/bin/bash
#
# Package and upload to PyPI using twine.

# To return a failure if any commands inside fail
set -e

echo ""
echo "Building source package and wheels for ${TRAVIS_TAG}"
echo ""
# Build source distributions and wheels
python setup.py sdist

echo ""
echo "Packages built:"
ls dist

echo ""
echo "Deploy to PyPI using twine."
echo ""
# Upload to PyPI. Credentials are set using the TWINE_PASSWORD and
# TWINE_USERNAME env variables.
twine upload --skip-existing dist/*

# Workaround for https://github.com/travis-ci/travis-ci/issues/6522
# Turn off exit on failure.
set +e
