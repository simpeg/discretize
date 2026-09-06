#!/bin/bash
set -ex #echo on and exit if any line fails

uv sync --python 3.13 --no-install-project --extra style
