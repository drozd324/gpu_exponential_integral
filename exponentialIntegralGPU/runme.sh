#!/bin/bash

make

FILE_NAME="out"
> $FILE_NAME

for (( i=1; i<3; i++ )); do

	FLAGS="-t -e -B $((32*i))" 

	echo "./main $FLAGS -n 10 -m 10 >> out"
	echo "./main $FLAGS -n 10 -m 10 >> out" >> out
	./main $FLAGS -n 10 -m 10 >> out

	echo "./main $FLAGS -n 5000 -m 5000 >> out"
	echo "./main $FLAGS -n 5000 -m 5000 >> out" >> out
	./main $FLAGS -n 5000 -m 5000 >> out

	echo "./main $FLAGS -n 8192 -m 8192 >> out "
	echo "./main $FLAGS -n 8192 -m 8192 >> out " >> out
	./main $FLAGS -n 8192 -m 8192 >> out 
#	
#	echo "./main $FLAGS -n 16384 -m 16384 >> out"
#	echo "./main $FLAGS -n 16384 -m 16384 >> out" >> out
#	./main $FLAGS -n 16384 -m 16384 >> out
#	
#	echo "./main $FLAGS -n 20000 -m 20000 >> out"
#	echo "./main $FLAGS -n 20000 -m 20000 >> out" >> out
#	./main $FLAGS -n 20000 -m 20000 >> out
#	
#
done

