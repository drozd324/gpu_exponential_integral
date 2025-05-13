#include <time.h>
#include <iostream>
#include <limits>       // std::numeric_limits
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#ifdef __cplusplus

template <typename real>
__global__ real exponentialIntegral_grid_GPU(real* results, const int n,
		double a, double b, int maxIterations, unsigned int numberOfSamples);

template <typename real>
real exponentialIntegral_grid_CPU(std::vector<std::vector<real>> results, const int n,
		double a, double b, int maxIterations, unsigned int numberOfSamples);

template <typename real>
real exponentialIntegral(const int n, const real x, int maxIterations);

#endif
