#!/bin/bash

nvcc -o expint_cuda main.cpp expint_gpu.cu -lcuda -lcudart
