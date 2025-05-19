#!/bin/bash

make

SIZES=(5000 8192 16384 20000)
FILE_NAME="out.csv"
echo "n,numberOfSamples,time_cpu_float,time_cpu_double,block_size,time_gpu_float,time_gpu_double,diff_float,diff_double,spdup_float,spdup_double,cpu,gpu" > $FILE_NAME

for (( j=0; j<4; j++ )); do
	FLAGS_CPU="-g -r" 
	echo "./main $FLAGS_CPU -n ${SIZES[$j]} -m ${SIZES[$j]} >> $FILE_NAME"
	./main $FLAGS_CPU -n ${SIZES[$j]} -m ${SIZES[$j]} >> $FILE_NAME

	for (( i=1; i<33; i++ )); do

		FLAGS_GPU="-c -r -B $i " 
		echo "./main $FLAGS_GPU -n ${SIZES[$j]} -m ${SIZES[$j]} >> $FILE_NAME"
		./main $FLAGS_GPU -n ${SIZES[$j]} -m ${SIZES[$j]} >> $FILE_NAME
	done
done

