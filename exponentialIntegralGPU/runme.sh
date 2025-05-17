#!/bin/bash

FILE_NAME="out"
> $FILE_NAME


BlOCK_SIZE=()
SIZES=4

for ((i=0; i<$NUM_POWERS; i++)); do
	BLOCK_SIZE+=($((2*i)))
done

TEMP=$(./task2 -m ${POWERS[$j]} -n ${POWERS[$j]} -p $NUM_ITER -x ${POWERS[$i]} -y ${POWERS[$i]} -s)

for (( i=0; i<; i++ )); do

	FLAGS="-t -e -B ${i}" 
	./main $FLAGS -n 5000 -m 5000 >> out
	./main $FLAGS -n 8192 -m 8192 >> out 
	#./main $FLAGS -n 16384 -m 16384 >> out
	#./main $FLAGS -n 20000 -m 20000 >> out

done

