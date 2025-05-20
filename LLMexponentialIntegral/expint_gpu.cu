#include <stdio.h>
#include <math.h>
#include <limits>

__device__ float exponentialIntegralFloatDevice(int n, float x, int maxIterations) {
    const float eulerConstant = 0.5772156649015329f;
    float epsilon = 1.E-30f;
    float bigfloat = 1e30f;
    int i, ii, nm1 = n - 1;
    float a, b, c, d, del, fact, h, psi, ans = 0.0f;

    if (n == 0) {
        return expf(-x) / x;
    } else {
        if (x > 1.0f) {
            b = x + n;
            c = bigfloat;
            d = 1.0f / b;
            h = d;
            for (i = 1; i <= maxIterations; i++) {
                a = -i * (nm1 + i);
                b += 2.0f;
                d = 1.0f / (a * d + b);
                c = b + a / c;
                del = c * d;
                h *= del;
                if (fabsf(del - 1.0f) <= epsilon) {
                    return h * expf(-x);
                }
            }
            return h * expf(-x);
        } else {
            ans = (nm1 != 0 ? 1.0f / nm1 : -logf(x) - eulerConstant);
            fact = 1.0f;
            for (i = 1; i <= maxIterations; i++) {
                fact *= -x / i;
                if (i != nm1) {
                    del = -fact / (i - nm1);
                } else {
                    psi = -eulerConstant;
                    for (ii = 1; ii <= nm1; ii++) psi += 1.0f / ii;
                    del = fact * (-logf(x) + psi);
                }
                ans += del;
                if (fabsf(del) < fabsf(ans) * epsilon) return ans;
            }
            return ans;
        }
    }
}

__global__ void computeExpIntKernel(float* results, int n, int numberOfSamples, float a, float b, int maxIterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * numberOfSamples;

    if (tid < total) {
        int order = tid / numberOfSamples + 1;
        int sample = tid % numberOfSamples + 1;
        float x = a + sample * ((b - a) / (float)numberOfSamples);
        results[tid] = exponentialIntegralFloatDevice(order, x, maxIterations);
    }
}


