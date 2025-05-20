#include "funcs.h"

template <typename real>
__host__ __device__ inline real exponentialIntegral(int n, real x, int maxIterations){
	real epsilon = 1e-30;
	//real bigReal = std::numeric_limits<real>::max();
	real bigReal = biggestNum<real>();
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
real max_diff(std::vector<std::vector<real>>& a, real* b, int m, int n){
	real diff;
	real top_diff = 0;

	for (int i=0; i<m; i++){
		for (int j=0; j<n; j++){
			diff = fabs(a[i][j] - b[i*n + j]);
			if (top_diff < diff){
				top_diff = diff;
			}
		}
	}
	return top_diff;
}

template <typename real>
void exponentialIntegral_grid_CPU(std::vector<std::vector<real>>& results, int n,
		double a, double b, int maxIterations, int numberOfSamples){

	real x;
	real division = (b - a) / numberOfSamples;

	for (int ui=1; ui<=n; ui++) {
		for (int uj=1; uj<=numberOfSamples; uj++) {
			x = a + uj*division;
			results[ui-1][uj-1]  = exponentialIntegral<real>(ui, x, maxIterations);
		}
	}
}


template <typename real>
__global__ void exponentialIntegral_grid_GPU(real* results, int n,
		double a, double b, int maxIterations, int numberOfSamples){

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	real x;
	real division = (b - a) / numberOfSamples;
	
	if (idx<n && idy<numberOfSamples){
		x = a + (idy+1)*division;
		results[idx*numberOfSamples + idy]  = exponentialIntegral<real>(idx+1, x, maxIterations);
	}
}


template <typename real>
__global__ void exponentialIntegral_grid_GPU_dynamic(real* results, int n,
		double a, double b, int maxIterations, int numberOfSamples, dim3 nBlock, dim3 nGrid){

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	
	if (idx==0 && idy==0){
		// very dynamic parallel
		exponentialIntegral_grid_GPU<real><<<nGrid, nBlock>>>(results, n, a, b, maxIterations, numberOfSamples);
	}

}


template float max_diff(std::vector<std::vector<float>>& a, float* b, int m, int n);
template double max_diff(std::vector<std::vector<double>>& a, double* b, int m, int n);

template void exponentialIntegral_grid_CPU<float>(std::vector<std::vector<float>>&, int, double, double, int, int);
template void exponentialIntegral_grid_CPU<double>(std::vector<std::vector<double>>&, int, double, double, int, int);

template __global__ void exponentialIntegral_grid_GPU<float>(float*, int, double, double, int, int);
template __global__ void exponentialIntegral_grid_GPU<double>(double*, int, double, double, int, int);

template  __global__ void exponentialIntegral_grid_GPU_dynamic<float>(float* results, int n, double a, double b, int maxIterations, int numberOfSamples, dim3 nBlock, dim3 nGrid);
template  __global__ void exponentialIntegral_grid_GPU_dynamic<double>(double* results, int n, double a, double b, int maxIterations, int numberOfSamples, dim3 nBlock, dim3 nGrid);

