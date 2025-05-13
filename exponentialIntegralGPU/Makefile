CC = g++
NVCC = nvcc

CXXFLAGS  = -O4 $(DEBUG)    # for g++
NVCCFLAGS = -O4# -funroll-loops --use_fast_math --compiler-options  -arch=sm_75

DEBUG   = # -g -Wextra -Wall
DEBUGNV =  # -g -G --target-processes

EXECS = main

all: $(EXECS)

main: main.cu funcs.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(DEBUG) $(DEBUGNV)

funcs.o: funcs.cu
	$(NVCC) $(NVCCFLAGS) -c $< $(DEBUG) $(DEBUGNV)

clean:
	rm -f *.o $(EXECS)




#task1: task1.c task1_funcs.o
#	        $(CC) $(NVCCFLAGS) -o $@ $^ $(DEBUG) $(DEBUGNV)
#
#task1_funcs.o: task1_funcs.c
#	        $(CC) $(NVCCFLAGS) -c $< $(DEBUG) $(DEBUGNV)
#
#task2: task2.cu task2_funcs.o task1_funcs.o
#	        $(NVCC) $(NVCCFLAGS) -o $@ $^ $(DEBUG) $(DEBUGNV)
#
#task2_funcs.o: task2_funcs.cu
#	        $(NVCC) $(NVCCFLAGS) -c $< $(DEBUG) $(DEBUGNV)
#


#main: $(OBJECTS)
#	$(NVCC) $(OBJECTS) -o $(TARGET) 
#
#add_vectors.o: add_vectors.cu
#	$(NVCC)  add_vectors.cu -c $(NVCCFLAGS) -I$(INCPATH)
#
