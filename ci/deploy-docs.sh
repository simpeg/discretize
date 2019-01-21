#!/bin/bash

# To return a failure if any commands inside fail
set -e

# unpack credentials if they are not already open
if [ ! -d "credentials" ]; then
    openssl aes-256-cbc -K $encrypted_5813a1339455_key -iv $encrypted_5813a1339455_iv -in credentials.tar.gz.enc -out credentials.tar.gz -d
    tar -xvzf credentials.tar.gz
fi

# add conda activate to the shell
# echo ". $HOME/miniconda/etc/profile.d/conda.sh" >> ~/.bashrc
# source ~/.bashrc

# authenticate with gcloud
cd docs
echo "Starting deploy of the docs"

# install and setup lib
conda create -n py27 python=2.7
$HOME/miniconda/etc/profile.d/conda.sh activate py27
mkdir lib
pip install -t lib/ flask
ls lib
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-228.0.0-linux-x86_64.tar.gz
tar zxvf google-cloud-sdk
pip install google-compute-engine

# deploy
gcloud auth activate-service-account --key-file credentials/client-secret.json
echo "Successfully authenticated"
gcloud config set project $GAE_PROJECT
gcloud app deploy app.yaml --version ${TRAVIS_COMMIT} --promote
echo "Done deploying docs"

# deactivate python 2.7 environment
$HOME/miniconda/etc/profile.d/conda.sh deactivate
cd ../

# Workaround for https://github.com/travis-ci/travis-ci/issues/6522
# Turn off exit on failure.
set +e
