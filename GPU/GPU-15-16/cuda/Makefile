
debug		:= 0
include		../Makefile.inc


all: clean darker_fused histogram2_

darker: darker.cpp darker.cu
	$(NVCC) -c darker.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/darkerCUDA darker.cpp darker.o ../src/Timer.cpp $(INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./darker.o

darker2: darker2.cpp darker2.cu
	$(NVCC) -c darker2.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/darker2CUDA darker2.cpp darker2.o ../src/Timer.cpp $(INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./darker2.o

darker2_: darker2_.cpp darker2_.cu
	$(NVCC) -c darker2_.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/darker2_CUDA darker2_.cpp darker2_.o ../src/Timer.cpp $(INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./darker2_.o

darker_fused: darker_fused.cpp darker_fused.cu
	$(NVCC) -c darker_fused.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/darker_fusedCUDA darker_fused.cpp darker_fused.o ../src/Timer.cpp $(INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./darker_fused.o

histogram: histogram.cpp histogram.cu
	$(NVCC) -c histogram.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/histogramCUDA histogram.cpp histogram.o ../src/Timer.cpp $(INCLUDES) $(CUDA_INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./histogram.o

histogram2: histogram2.cpp histogram2.cu
	$(NVCC) -c histogram2.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/histogram2CUDA histogram2.cpp histogram2.o ../src/Timer.cpp $(INCLUDES) $(CUDA_INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./histogram2.o

histogram2_: histogram2_.cpp histogram2_.cu
	$(NVCC) -c histogram2_.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/histogram2_CUDA histogram2_.cpp histogram2_.o ../src/Timer.cpp $(INCLUDES) $(CUDA_INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./histogram2_.o

smooth: smooth.cpp smooth.cu
	$(NVCC) -c smooth.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/smoothCUDA smooth.cpp smooth.o ../src/Timer.cpp $(INCLUDES) $(CUDA_INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./smooth.o

clean:
	rm -f $(PROJ_BASE)/bin/darkerCUDA
	rm -f $(PROJ_BASE)/bin/histogramCUDA
	rm -f $(PROJ_BASE)/bin/smoothCUDA
	rm -f $(PROJ_BASE)/bin/darker2CUDA
	rm -f $(PROJ_BASE)/bin/histogram2CUDA
	rm -f $(PROJ_BASE)/bin/darker_fusedCUDA

