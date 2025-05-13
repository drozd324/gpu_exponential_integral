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
//
//template <typename real>
//real exponentialIntegral(const int n, const real x, int maxIterations);
//
//template <typename real>
//__global__ void exponentialIntegral_grid_GPU(real* results, const int n,
//		double a, double b, int maxIterations, unsigned int numberOfSamples);
//
//template <typename real>
//void exponentialIntegral_grid_CPU(std::vector<std::vector<real>>& results, const int n,
//		real a, real b, int maxIterations, unsigned int numberOfSamples);
//

//===================================================================================================//

template <typename real>
__host__ __device__ real exponentialIntegral(const int n, const real x, int maxIterations){
	static const real eulerConstant = 0.5772156649015329;
	real epsilon = 1e-30;
	real bigReal = std::numeric_limits<real>::max();
	int i, ii, nm1 = n - 1;
	real a, b, c, d, del, fact, h, psi, ans = 0.0;

	if (n<0.0 || x<0.0 || (x==0.0 && ((n==0) || (n==1))) ){
		printf("Bad arguments were passed to the exponentialIntegral function call\n");
		return -1.0;
	}

	if (n == 0){
		ans = exp(-x) / x;
	} else{
		if (x > 1.0){
			b = x + n;
			c = bigReal;
			d = 1.0 / b;
			h = d;
			for (i=1; i<=maxIterations; i++){
				a = - i*(nm1 + i);
				b += 2.0;
				d = 1.0 / (a*d + b);
				c = b + a / c;
				del = c*d;
				h *= del;
				if (fabs(del - 1.0) <= epsilon){
					ans = h*exp(-x);
					return ans;
				}
			}
			ans = h*exp(-x);
			return ans;
		} else { // Evaluate series
			ans = (nm1 != 0 ? 1.0 / nm1 : - log(x) - eulerConstant);	// First term
			fact = 1.0;
			for (i=1; i<=maxIterations; i++) {
				fact *= - x / i;
				if (i != nm1){
					del = - fact / (i - nm1);
				} else {
					psi = - eulerConstant;
					for (ii=1; ii<=nm1; ii++) {
						psi += 1.0 / ii;
					}
					del = fact*(-log(x) + psi);
				}
				ans += del;
				if (fabs(del) < fabs(ans)*epsilon)
					return ans;
			}
			return ans;
		}
	}
	return ans;
}



template <typename real>
void exponentialIntegral_grid_CPU(std::vector<std::vector<real>>& results, const int n,
		double a, double b, int maxIterations, unsigned int numberOfSamples){

	real x;
	real division = (b - a) / ((real)(numberOfSamples));

	for (int ui=1; ui<=n; ui++) {
		for (int uj=1; uj<=numberOfSamples; uj++) {
			x = a + uj*division;
			results[ui-1][uj-1]  = exponentialIntegral<real>(ui, x, maxIterations);
		}
	}
}


//===================================================================================================//
//
template <typename real>
__global__ void exponentialIntegral_grid_GPU(real* results, const int n,
		double a, double b, int maxIterations, unsigned int numberOfSamples){

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	real x;
	real division = (b - a) / ((real)(numberOfSamples));
	
	if (idx<n && idy<numberOfSamples){
		x = a + (idy+1)*division;
		results[idx*n + idy] = exponentialIntegral<real>(idx+1, x, maxIterations);
	}
}


#endif
