
debug		:= 0
include		../Makefile.inc


all: clean smooth2

darker: darker.cpp darker.cu
	$(NVCC) -c darker.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/darkerCUDA darker.cpp darker.o ../src/Timer.cpp $(INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./darker.o

histogram: histogram.cpp histogram.cu
	$(NVCC) -c histogram.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/histogramCUDA histogram.cpp histogram.o ../src/Timer.cpp $(INCLUDES) $(CUDA_INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./histogram.o

histogram2: histogram2.cpp histogram2.cu
	$(NVCC) -c histogram2.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/histogram2CUDA histogram2.cpp histogram2.o ../src/Timer.cpp $(INCLUDES) $(CUDA_INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./histogram2.o

smooth: smooth.cpp smooth.cu
	$(NVCC) -c smooth.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/smoothCUDA smooth.cpp smooth.o ../src/Timer.cpp $(INCLUDES) $(CUDA_INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./smooth.o

smooth2: smooth2.cpp smooth2.cu
	$(NVCC) -c smooth2.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/smooth2CUDA smooth2.cpp smooth2.o ../src/Timer.cpp $(INCLUDES) $(CUDA_INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./smooth2.o

clean:
	rm -f $(PROJ_BASE)/bin/darkerCUDA
	rm -f $(PROJ_BASE)/bin/histogramCUDA
	rm -f $(PROJ_BASE)/bin/histogram2CUDA
	rm -f $(PROJ_BASE)/bin/smoothCUDA

