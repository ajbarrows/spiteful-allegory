#! /bin/bash

# sync data to cluster
bash 03_local_remote.sh

# export Conda environment
conda env export -n pocs --no-builds | grep -v "prefix" > ../env-nobuild.yml

# push remaining changes to Github
git add ..
git commit -m "Housekeeping sync"
git push

