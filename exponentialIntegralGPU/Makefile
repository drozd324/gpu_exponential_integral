CC = g++
NVCC = nvcc

CXXFLAGS  = -O4 $(DEBUG)    # for g++
NVCCFLAGS = -O4# -funroll-loops --use_fast_math --compiler-options  -arch=sm_75

DEBUGNV = #   -g -G --target-processes

EXECS = main

all: $(EXECS)

main: main.cu funcs.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(DEBUGNV)

funcs.o: funcs.cu
	$(NVCC) $(NVCCFLAGS) -c $< $(DEBUGNV)

clean:
	rm -f *.o $(EXECS)
