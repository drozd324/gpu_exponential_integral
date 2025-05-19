#include <time.h>
#include <iostream>
#include <limits>       
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <fstream>

#include "funcs.h"

using namespace std;

void outputResultsCpu(const std::vector<std::vector<float>>& resultsFloatCpu, const std::vector<std::vector<double>>& resultsDoubleCpu);
void outputResultsGpu(float* resultsFloatGpu, double* resultsDoubleGpu);
int	parseArguments(int argc, char **argv);
void printUsage(void);

int maxIterations;
bool verbose, timing, cpu, gpu, error, csv;
int n, numberOfSamples;
double a, b; // The interval that we are going to use

int block_size = 32;

int main(int argc, char *argv[]){
	cpu = true;
	gpu = true;
	verbose = false;
	timing = false;
	error = false;
	csv = false;
	// n is the maximum order of the exponential integral that we are going to test
	// numberOfSamples is the number of samples in the interval [0,10] that we are going to calculate
	n = 10;
	numberOfSamples = 10;
	a = 0.0;
	b = 10.0;
	maxIterations = 2000000000;
	struct timeval expoStart, expoEnd;

	parseArguments(argc, argv);

	if (verbose){
		cout << "n=" << n << endl;
		cout << "numberOfSamples=" << numberOfSamples << endl;
		cout << "a=" << a << endl;
		cout << "b=" << b << endl;
		cout << "timing=" << timing << endl;
		cout << "verbose=" << verbose << endl;
	}

	// Sanity checks
	if (a >= b){
		cout << "Incorrect interval ("<<a<<","<<b<<") has been stated!" << endl;
		return 0;
	}
	if (n <= 0){
		cout << "Incorrect orders ("<<n<<") have been stated!" << endl;
		return 0;
	}
	if (numberOfSamples <= 0){
		cout << "Incorrect number of samples ("<<numberOfSamples<<") have been stated!" << endl;
		return 0;
	}

	std::vector<std::vector<float >> resultsFloatCpu;
	std::vector<std::vector<double>> resultsDoubleCpu;
	double timeTotalCpuFloat = 0.0;
	double timeTotalCpuDouble = 0.0;

	float*  resultsFloatGpu;
	double* resultsDoubleGpu;
	double timeTotalGpuFloat  = 0.0;
	double timeTotalGpuDouble = 0.0;
		

	if (cpu){

		gettimeofday(&expoStart, NULL);
		resultsFloatCpu.resize (n, vector<float >(numberOfSamples));
		exponentialIntegral_grid_CPU<float>(resultsFloatCpu, n, a, b, maxIterations, numberOfSamples);
		gettimeofday(&expoEnd, NULL);
		timeTotalCpuFloat = ((expoEnd.tv_sec + expoEnd.tv_usec*0.000001) - (expoStart.tv_sec + expoStart.tv_usec*0.000001));

		gettimeofday(&expoStart, NULL);
		resultsDoubleCpu.resize(n, vector<double>(numberOfSamples));
		exponentialIntegral_grid_CPU<double>(resultsDoubleCpu, n, a, b, maxIterations, numberOfSamples);
		gettimeofday(&expoEnd, NULL);
		timeTotalCpuDouble = ((expoEnd.tv_sec + expoEnd.tv_usec*0.000001) - (expoStart.tv_sec + expoStart.tv_usec*0.000001));
	}

	if (gpu){
		
		if (block_size > 1024)
			printf("[ERROR] BLOCK SIZE TOO LARGE\n");

		int N = n;
		int M = numberOfSamples;
		dim3 nBlock(block_size, block_size);
		dim3 nGrid((N/nBlock.x ) + (!(N%nBlock.x) ? 0:1) , (M/nBlock.y) + (!(M%nBlock.y) ? 0:1));


		gettimeofday(&expoStart, NULL);
		float* resultsFloatGpu_d;
		cudaMalloc((void**)& resultsFloatGpu_d , n*numberOfSamples * sizeof(float));

		exponentialIntegral_grid_GPU<float><<<nGrid, nBlock>>>(resultsFloatGpu_d, n, a, b, maxIterations, numberOfSamples);

		cudaStream_t streamFloat;
		cudaStreamCreate(&streamFloat);
		resultsFloatGpu = (float*)malloc(n*numberOfSamples * sizeof(float));
		cudaMemcpyAsync(resultsFloatGpu , resultsFloatGpu_d , n*numberOfSamples * sizeof(float) , cudaMemcpyDeviceToHost, streamFloat);
		//cudaMemcpy(resultsFloatGpu , resultsFloatGpu_d , n*numberOfSamples * sizeof(float) , cudaMemcpyDeviceToHost);

		cudaFree(resultsFloatGpu_d);
		cudaStreamSynchronize(streamFloat);
		cudaStreamDestroy(streamFloat);
		gettimeofday(&expoEnd, NULL);
		timeTotalGpuFloat = ((expoEnd.tv_sec + expoEnd.tv_usec*0.000001) - (expoStart.tv_sec + expoStart.tv_usec*0.000001));
	

		gettimeofday(&expoStart, NULL);
		double* resultsDoubleGpu_d;
		cudaMalloc((void**)& resultsDoubleGpu_d, n*numberOfSamples * sizeof(double));

		exponentialIntegral_grid_GPU<double><<<nGrid, nBlock>>>(resultsDoubleGpu_d, n, a, b, maxIterations, numberOfSamples);

		cudaStream_t streamDouble;
		cudaStreamCreate(&streamDouble);
		resultsDoubleGpu = (double*)malloc(n*numberOfSamples * sizeof(double));
		cudaMemcpyAsync(resultsDoubleGpu, resultsDoubleGpu_d, n*numberOfSamples * sizeof(double), cudaMemcpyDeviceToHost, streamDouble);
		//cudaMemcpy(resultsDoubleGpu, resultsDoubleGpu_d, n*numberOfSamples * sizeof(double), cudaMemcpyDeviceToHost);

		cudaFree(resultsDoubleGpu_d);
		cudaStreamSynchronize(streamDouble);
		cudaStreamDestroy(streamDouble);
		gettimeofday(&expoEnd, NULL);
		timeTotalGpuDouble = ((expoEnd.tv_sec + expoEnd.tv_usec*0.000001) - (expoStart.tv_sec + expoStart.tv_usec*0.000001));
	}


	if (timing){
		if (cpu){
			printf("calculating the exponentials on the CPU in FLOATS  took: %f seconds\n", timeTotalCpuFloat);
			printf("calculating the exponentials on the CPU in DOUBLES took: %f seconds\n", timeTotalCpuDouble);
		}

		if (gpu){
			printf("calculating the exponentials on the GPU in FLOATS  took: %f seconds\n", timeTotalGpuFloat);
			printf("calculating the exponentials on the GPU in DOUBLES took: %f seconds\n", timeTotalGpuDouble);
		}
	}

	if (verbose){
		if (cpu){
			outputResultsCpu(resultsFloatCpu, resultsDoubleCpu);
		}
		if (gpu){
			outputResultsGpu(resultsFloatGpu, resultsDoubleGpu);
		}
	}

	if (error){
		if (gpu && cpu){
			std::cout << "Max absolute CPU GPU difference Float: " << max_diff(resultsFloatCpu, resultsFloatGpu, n, numberOfSamples) << "\n";
			std::cout << "Max absolute CPU GPU difference Double: " << max_diff(resultsDoubleCpu, resultsDoubleGpu, n, numberOfSamples) << "\n";
		}
	}

	if (csv){
		//"n,numberOfSamples,time_cpu_float,time_cpu_double,block_size,time_gpu_float,time_gpu_double,diff_float,diff_double,spdup_float,spdup_double,cpu,gpu" 
		int none = -9999;	

		std::cout << none << ",";
		std::cout << n << "," << numberOfSamples << ",";		
		

		if (cpu){
			std::cout << timeTotalCpuFloat << "," << timeTotalCpuDouble << ",";		
		} else {
			std::cout << none << "," << none << ",";		
		}

		if (gpu){
			std::cout << block_size*block_size << ",";
			std::cout << timeTotalGpuFloat << "," << timeTotalGpuDouble << ",";		
		} else {
			std::cout << none << ",";
			std::cout << none << "," << none << ",";		
		}

		if (gpu && cpu){
			std::cout << max_diff(resultsFloatCpu, resultsFloatGpu, n, numberOfSamples) << ",";
			std::cout << max_diff(resultsDoubleCpu, resultsDoubleGpu, n, numberOfSamples) << ",";
	
			std::cout << timeTotalCpuFloat / timeTotalGpuFloat << ",";
			std::cout << timeTotalCpuDouble / timeTotalGpuDouble << ",";
		} else {
			std::cout << none << ",";
			std::cout << none << ",";
	
			std::cout << none << ",";
			std::cout << none << ",";
		}

		std::cout << cpu << ",";
		std::cout << gpu << ",";
		
		std::cout << std::endl;
	}

	return 0;
}

void outputResultsCpu(const std::vector<std::vector<float>>& resultsFloatCpu, const std::vector<std::vector<double>>& resultsDoubleCpu){
	unsigned int ui, uj;
	double x, division = (b - a) / ((double)(numberOfSamples));

	for (ui=1; ui<=n; ui++) {
		for (uj=1; uj<=numberOfSamples; uj++) {
			x = a + uj*division;
			std::cout << "CPU==> exponentialIntegralDouble (" << ui << "," << x <<")=" << resultsDoubleCpu[ui-1][uj-1] << " ,";
			std::cout << "exponentialIntegralFloat  ("        << ui << "," << x <<")=" << resultsFloatCpu [ui-1][uj-1] << endl;
		}
	}
}

void outputResultsGpu(float* resultsFloatGpu, double* resultsDoubleGpu){
	unsigned int ui, uj;
	double x, division = (b - a) / ((double)(numberOfSamples));

	for (ui=1; ui<=n; ui++) {
		for (uj=1; uj<=numberOfSamples; uj++) {
			x = a + uj*division;
			std::cout << "GPU==> exponentialIntegralDouble (" << ui << "," << x <<")=" << resultsDoubleGpu[(ui-1)*numberOfSamples + uj-1] << " ,";
			std::cout << "exponentialIntegralFloat  ("        << ui << "," << x <<")=" << resultsFloatGpu [(ui-1)*numberOfSamples + uj-1] << endl;
		}
	}
}

int parseArguments(int argc, char *argv[]){
	int c;
	while ((c = getopt(argc, argv, "cgehri:n:m:a:b:B:tv")) != -1){
		switch (c) {
			case 'c':
				cpu = false;
			   	break;	 //Skip the CPU test
			case 'g':
				gpu = false;
			   	break;	 //Skip the CPU test
			case 'h':
				printUsage(); 
				exit(0);
			   	break;
			case 'i':
				maxIterations = atoi(optarg);
			   	break;
			case 'n':
				n = atoi(optarg);
			   	break;
			case 'm':
				numberOfSamples = atoi(optarg);
			   	break;
			case 'a':
				a = atof(optarg);
			   	break;
			case 'b':
				b = atof(optarg);
			   	break;
			case 't':
				timing = true;
			   	break;
			case 'r':
				csv = true;
			   	break;
			case 'v':
				verbose = true;
			   	break;
			case 'e':
				error = true;
			   	break;
			case 'B':
				block_size = atoi(optarg);
			   	break;
			default:
				fprintf(stderr, "Invalid option given\n");
				printUsage();
				return -1;
		}
	}
	return 0;
}

void printUsage(){
	printf("exponentialIntegral program\n");
	printf("by: Jose Mauricio Refojo <refojoj@tcd.ie>\n");
	printf("This program will calculate a number of exponential integrals\n");
	printf("usage:\n");
	printf("exponentialIntegral.out [options]\n");
	printf("      -a   value   : will set the a value of the (a,b) interval in which the samples are taken to value (default: 0.0)\n");
	printf("      -b   value   : will set the b value of the (a,b) interval in which the samples are taken to value (default: 10.0)\n");
	printf("      -c           : will skip the CPU test\n");
	printf("      -g           : will skip the GPU test\n");
	printf("      -h           : will show this usage\n");
	printf("      -i   size    : will set the number of iterations to size (default: 2000000000)\n");
	printf("      -n   size    : will set the n (the order up to which we are calculating the exponential integrals) to size (default: 10)\n");
	printf("      -m   size    : will set the number of samples taken in the (a,b) interval to size (default: 10)\n");
	printf("      -t           : will output the amount of time that it took to generate each norm (default: no)\n");
	printf("      -v           : will activate the verbose mode  (default: no)\n");
	printf("      -e           : will show the error between gpu and cpu vercions of the code (default: no)\n");
	printf("      -B   value   : will chage the block size for the cuda grid (default: 32)\n");
	printf("      -r           : will print runtime properites into a .csv style file (default: no)\n");
	printf("     \n");
}
