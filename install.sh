#!/usr/bin/env bash

set -ex

ENV_NAME=venv

conda create -y -n "$ENV_NAME" \
    tomosipo \
    astra-toolbox \
    numpy \
    scikit-image \
    pytorch \
    -c default \
    -c astra-toolbox \
    -c nvidia \
    -c aahendriksen \
    -c pytorch

conda install -y -n "$ENV_NAME" \
    tifffile \
    matplotlib \
    pydicom \
    vedo \
    -c default \
    -c conda-forge

conda run -n "$ENV_NAME" pip install gecatsim

conda run -n "$ENV_NAME" pip install git+https://github.com/ahendriksen/ts_algorithms.git
