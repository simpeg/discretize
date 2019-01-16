#!/bin/bash


cd docs
gcloud auth activate-service-account --key-file credentials/client-secret.json
gcloud config set project $GAE_PROJECT;
gcloud app deploy app.yaml --version ${TRAVIS_COMMIT} --promote;

# Workaround for https://github.com/travis-ci/travis-ci/issues/6522
# Turn off exit on failure.
set +e
