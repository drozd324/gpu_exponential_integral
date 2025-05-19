CC = g++
NVCC = nvcc

CXXFLAGS  =
NVCCFLAGS = -O4 --use_fast_math --compiler-options -funroll-loops -arch=sm_75

DEBUGNV = #   -g -G --target-processes

EXECS = main

all: $(EXECS)

main: main.cu funcs.o
	$(NVCC)  -o $@ $^ $(DEBUGNV) $(NVCCFLAGS)

funcs.o: funcs.cu
	$(NVCC) -c $< $(DEBUGNV) $(NVCCFLAGS) 

clean:
	rm -f *.o $(EXECS)
