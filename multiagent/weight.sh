#!/bin/bash

for ((a=1;a<=1;a++)); do 
	for ((b=-3;b<0;b++)); do
		for ((c=1;c<=3;c++)); do 
			for ((d=1;d<=4;d++)); do
				for ((e=-5;e<0;e++)); do 
					for ((f=-5;f<0;f++)); do
						python pacman.py -p ExpectimaxAgent -l smallClassic  -a evalFn=better,depth=3,a=$a,b=$b,c=$c,d=$d,e=$e,f=$f --frameTime 0 -q
						python pacman.py -p ExpectimaxAgent -l smallClassic  -a evalFn=better,depth=3,a=$a,b=$b,c=$c,d=$d,e=$e,f=$f --frameTime 0 -q
						python pacman.py -p ExpectimaxAgent -l smallClassic  -a evalFn=better,depth=3,a=$a,b=$b,c=$c,d=$d,e=$e,f=$f --frameTime 0 -q
						echo $a,$b,$c,$d,$e,$f					
					done 
				done
			done 
		done
	done 
done