#!/bin/bash

# Create directory structure
mkdir $1
cd $1

# Writeup
mkdir figures

# Gather Peter's LaTeX files and directories
ln -s ../../overleaf/settings/ .
ln -s ../../../overleaf/figures/local/ ./figures
ln -s ../../overleaf/everything.bib .
ln -s ../../overleaf/unsrtabbrv.bst .

# copy LaTeX template
cp ../../overleaf/$1.tex $1.tex
cp ../../overleaf/$1.body.tex $1.body.tex

# link to /work
# ln -s ../../../work/data/08_reporting/ ./figures/
cp -S ../../work/data/08_reporting/* ./figures/