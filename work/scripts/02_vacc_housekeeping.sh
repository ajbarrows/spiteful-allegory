#! /bin/bash

cd ../

# pull from Github
git pull

# update Conda environment

conda env update --name pocs --file env-nobuild.yml --prune