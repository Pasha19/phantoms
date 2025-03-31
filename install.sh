#!/usr/bin/env bash

set -ex

ENV_NAME=venv

conda create -y -n "$ENV_NAME" \
    python=3.12 \
    tomosipo \
    astra-toolbox \
    "numpy<2.0" \
    scikit-image \
    pytorch \
    -c default \
    -c astra-toolbox \
    -c aahendriksen \
    -c pytorch

conda install -y -n "$ENV_NAME" \
    tifffile \
    matplotlib \
    pydicom \
    vedo \
    xdesign \
    -c default \
    -c conda-forge

conda run -n "$ENV_NAME" pip install gecatsim

conda run -n "$ENV_NAME" pip install git+https://github.com/ahendriksen/ts_algorithms.git
