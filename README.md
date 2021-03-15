*in development*

## Overview

This project contains the implementation of an ultra-flexible pipeline for pre-processing. It can be used for any data science or machine learning project. However, it is specifically designed to work with the generator interface of tensorflow-keras.

## Pipeline

Any pipeline can be represented by a directed acyclic graph. Each node corresponds to an operation and can have multiple inputs and outputs. The data-flow-principle is used: Each one of the source nodes output one new data point at each time step. This data then flows in parallel through the pipeline.

## Conda

The conda environment in conda_env.yml is used. To load all packages in conda_env.yml into your current local environment run:

'conda env export > conda_env.yml'
'pip install -r requirements.txt'

To update conda_env.yml to contain the same packages as your current local environment run:

'conda env update --file conda_env.yml  --prune'
'pip freeze > requirements.txt'