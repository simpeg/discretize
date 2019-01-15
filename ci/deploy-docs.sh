#!/bin/bash
#
# Package and upload to PyPI using twine. The env variables TWINE_USERNAME and
# TWINE_PASSWORD must exist with your pypi.org credentials.

cd docs
gcloud auth activate-service-account --key-file credentials/client-secret.json
gcloud config set project $GAE_PROJECT;
gcloud app deploy app.yaml --version ${TRAVIS_COMMIT} --promote;
