#!/bin/bash

# Create directory structure
mkdir $1
cd $1

# Code
mkdir work
mkdir work/notebooks
mkdir work/src
mkdir -p work/out/figures
ln -s ../../../data ./work # symbolic link to parent data

# Writeup
filename='CSYS6713_ajbarrow_'

mkdir writeup

ln -s ../../../overleaf/settings ./writeup
# ln -s ../../../overleaf/figures/ ./writeup/figures
ln -s ../../../overleaf/everything.bib ./writeup
ln -s ../../../overleaf/unsrtabbrv.bib ./writeup

cp ../../overleaf/$1.tex ./writeup/$filename$1.tex
cp ../../overleaf/$1.body.tex ./writeup/$filename$1.body.tex

ln -s ../work/out/figures ./writeup/



