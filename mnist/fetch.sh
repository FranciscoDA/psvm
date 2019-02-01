#!/bin/bash

FILENAMES="train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte"

for fn in $FILENAMES
do
	if [ ! -e $fn.gz ]
	then
		wget http://yann.lecun.com/exdb/mnist/$fn.gz
	fi
	if [ ! -e $fn.idx ]
	then
		gunzip -c $fn.gz > $fn.idx
	fi
done
