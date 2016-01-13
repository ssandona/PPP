
debug		:= 0
include		../Makefile.inc


all: darker_uni_more

darker_square: darker_square.cpp darker_square.cu
	$(NVCC) -c darker_square.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/darker_squareCUDA darker_square.cpp darker_square.o ../src/Timer.cpp $(INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./darker_square.o

darker: darker.cpp darker.cu
	$(NVCC) -c darker.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/darkerCUDA darker.cpp darker.o ../src/Timer.cpp $(INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./darker.o

darker2: darker2.cpp darker2.cu
	$(NVCC) -c darker2.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/darker2CUDA darker2.cpp darker2.o ../src/Timer.cpp $(INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./darker2.o

darker_square_more: darker_square_more.cpp darker_square_more.cu
	$(NVCC) -c darker_square_more.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/darker_square_moreCUDA darker_square_more.cpp darker_square_more.o ../src/Timer.cpp $(INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./darker_square_more.o

darker_uni: darker_uni.cpp darker_uni.cu
	$(NVCC) -c darker_uni.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/darker_uniCUDA darker_uni.cpp darker_uni.o ../src/Timer.cpp $(INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./darker_uni.o

darker_uni_more: darker_uni_more.cpp darker_uni_more.cu
	$(NVCC) -c darker_uni_more.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/darker_uni_moreCUDA darker_uni_more.cpp darker_uni_more.o ../src/Timer.cpp $(INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./darker_uni_more.o

darker_uni_more2: darker_uni_more2.cpp darker_uni_more2.cu
	$(NVCC) -c darker_uni_more2.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/darker_uni_more2CUDA darker_uni_more2.cpp darker_uni_more2.o ../src/Timer.cpp $(INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./darker_uni_more2.o

darker_grid: darker_grid.cpp darker_grid.cu
	$(NVCC) -c darker_grid.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/darker_gridCUDA darker_grid.cpp darker_grid.o ../src/Timer.cpp $(INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./darker_grid.o

histogram: histogram.cpp histogram.cu
	$(NVCC) -c histogram.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/histogramCUDA histogram.cpp histogram.o ../src/Timer.cpp $(INCLUDES) $(CUDA_INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./histogram.o

histogram_warp2: histogram_warp2.cpp histogram_warp2.cu
	$(NVCC) -c histogram_warp2.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/histogram_warp2CUDA histogram_warp2.cpp histogram_warp2.o ../src/Timer.cpp $(INCLUDES) $(CUDA_INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./histogram_warp2.o

histogram_square: histogram_square.cpp histogram_square.cu
	$(NVCC) -c histogram_square.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/histogram_squareCUDA histogram_square.cpp histogram_square.o ../src/Timer.cpp $(INCLUDES) $(CUDA_INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./histogram_square.o

histogram_square_more: histogram_square_more.cpp histogram_square_more.cu
	$(NVCC) -c histogram_square_more.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/histogram_square_moreCUDA histogram_square_more.cpp histogram_square_more.o ../src/Timer.cpp $(INCLUDES) $(CUDA_INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./histogram_square_more.o

histogram_square_more2: histogram_square_more2.cpp histogram_square_more2.cu
	$(NVCC) -c histogram_square_more2.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/histogram_square_more2CUDA histogram_square_more2.cpp histogram_square_more2.o ../src/Timer.cpp $(INCLUDES) $(CUDA_INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./histogram_square_more2.o

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

smooth2: smooth2.cpp smooth2.cu
	$(NVCC) -c smooth2.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/smooth2CUDA smooth2.cpp smooth2.o ../src/Timer.cpp $(INCLUDES) $(CUDA_INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./smooth2.o

smooth3: smooth3.cpp smooth3.cu
	$(NVCC) -c smooth3.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/smooth3CUDA smooth3.cpp smooth3.o ../src/Timer.cpp $(INCLUDES) $(CUDA_INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./smooth3.o

clean:
	rm -f $(PROJ_BASE)/bin/darker_squareCUDA
	rm -f $(PROJ_BASE)/bin/darker_uniCUDA
	rm -f $(PROJ_BASE)/bin/darker_uni_moreCUDA
	rm -f $(PROJ_BASE)/bin/darker_gridCUDA
	rm -f $(PROJ_BASE)/bin/histogramCUDA
	rm -f $(PROJ_BASE)/bin/smoothCUDA
	rm -f $(PROJ_BASE)/bin/darker2CUDA
	rm -f $(PROJ_BASE)/bin/histogram2CUDA
	rm -f $(PROJ_BASE)/bin/smooth2CUDA

