#!/bin/bash

# unpack credentials if they are not already open
if [ ! -d "credentials" ]; then
    openssl aes-256-cbc -K $encrypted_5813a1339455_key -iv $encrypted_5813a1339455_iv -in credentials.tar.gz.enc -out credentials.tar.gz -d
    tar -xvzf credentials.tar.gz
fi


# authenticate with gcloud
cd docs

# install and setup lib
conda create -n --yes py27 python=2.7 anaconda
conda activate py27
mkdir lib
pip install -t lib/ flask
ls lib
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-228.0.0-linux-x86_64.tar.gz
tar zxvf google-cloud-sdk
pip install google-compute-engine

# deploy
gcloud auth activate-service-account --key-file credentials/client-secret.json
gcloud config set project $GAE_PROJECT
gcloud app deploy app.yaml --version ${TRAVIS_COMMIT} --promote

# deactivate python 2.7 environment
conda deactivate

# Workaround for https://github.com/travis-ci/travis-ci/issues/6522
# Turn off exit on failure.
set +e
