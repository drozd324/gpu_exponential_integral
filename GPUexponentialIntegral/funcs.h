#include <iostream>
#include <limits>       // std::numeric_limits
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

__constant__ static const double eulerConstant = 0.5772156649015329;

template <typename real>
__host__ __device__ inline real biggestNum(){
	return 0;
}

template <>
__host__ __device__ inline float biggestNum<float>(){
	return 3.40282e+38; 
}

template <>
__host__ __device__ inline double biggestNum<double>(){
	return 1.79769e+308;
}


template <typename real>
__host__ __device__ inline real exponentialIntegral(int n, real x, int maxIterations);

template <typename real>
real max_diff(std::vector<std::vector<real>>& a, real* b, int m, int n);

template <typename real>
void exponentialIntegral_grid_CPU(std::vector<std::vector<real>>& results, int n,
		double a, double b, int maxIterations, int numberOfSamples);

template <typename real>
__global__ void exponentialIntegral_grid_GPU(real* results, int n,
		double a, double b, int maxIterations, int numberOfSamples);


template <typename real>
__global__ void exponentialIntegral_grid_GPU_dynamic(real* results, int n,
		double a, double b, int maxIterations, int numberOfSamples, dim3 nBlock, dim3 nGrid);
