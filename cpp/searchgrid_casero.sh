#!/bin/bash

test_size=800
for C in 0.2 0.4 0.6 0.8 1 1.2
do
	for tn in 500 1000 1500 2000 2500 5000
	do
		for gamma in 0.25 0.5 0.75 1
		do
			./cusvm --train-attributes mnist/train-images.idx3-ubyte --train-labels mnist/train-labels.idx1-ubyte --train-format idx --train-n 15000 --test-attributes mnist/t10k-images.idx3-ubyte --test-labels mnist/t10k-labels.idx1-ubyte --test-format idx --test-n $tn --normalize-zero-one --cost $C --kernel rbf --gamma $gamma
		done
		for p in 2 4 6 9
		do
			for c in 0 1
			do
				./cusvm --train-attributes mnist/train-images.idx3-ubyte --train-labels mnist/train-labels.idx1-ubyte --train-format idx --train-n 15000 --test-attributes mnist/t10k-images.idx3-ubyte --test-labels mnist/t10k-labels.idx1-ubyte --test-format idx --test-n $tn --normalize-zero-one --cost $C --kernel poly --power $p --constant $c
			done
		done
		./cusvm --train-attributes mnist/train-images.idx3-ubyte --train-labels mnist/train-labels.idx1-ubyte --train-format idx --train-n 15000 --test-attributes mnist/t10k-images.idx3-ubyte --test-labels mnist/t10k-labels.idx1-ubyte --test-format idx --test-n $tn --normalize-zero-one --cost $C --kernel linear
	done
done
