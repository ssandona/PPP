
debug		:= 0
include		../Makefile.inc


all:  darker_grid_32

darker_square: darker_square.cpp
	$(NVCC) -c darker_square.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/darker_squareCUDA darker_square.cpp darker_square.o ../src/Timer.cpp $(INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./darker_square.o

darker: darker.cpp darker.cu
	$(NVCC) -c darker.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/darkerCUDA darker.cpp darker.o ../src/Timer.cpp $(INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./darker.o

darker_try: darker_try.cpp darker_try.cu
	$(NVCC) -lineinfo -c darker_try.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/darker_tryCUDA darker_try.cpp darker_try.o ../src/Timer.cpp $(INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./darker_try.o


darker_fixed: darker_fixed.cpp darker_fixed.cu
	$(NVCC) -c darker_fixed.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/darker_fixedCUDA darker_fixed.cpp darker_fixed.o ../src/Timer.cpp $(INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./darker_fixed.o

darker3: darker3.cpp darker3.cu
	$(NVCC) -c darker3.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/darker3CUDA darker3.cpp darker3.o ../src/Timer.cpp $(INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./darker3.o

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

darker_grid_32: darker_grid_32.cpp darker_grid_32.cu
	$(NVCC) -c darker_grid_32.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/darker_grid_32CUDA darker_grid_32.cpp darker_grid_32.o ../src/Timer.cpp $(INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./darker_grid_32.o

histogram: histogram.cpp histogram.cu
	$(NVCC) -c histogram.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/histogramCUDA histogram.cpp histogram.o ../src/Timer.cpp $(INCLUDES) $(CUDA_INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./histogram.o

histogram_try: histogram_try.cpp histogram_try.cu
	$(NVCC) -c histogram_try.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/histogram_tryCUDA histogram_try.cpp histogram_try.o ../src/Timer.cpp $(INCLUDES) $(CUDA_INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./histogram_try.o

histogram_try2: histogram_try2.cpp histogram_try2.cu
	$(NVCC) -c histogram_try2.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/histogram_try2CUDA histogram_try2.cpp histogram_try2.o ../src/Timer.cpp $(INCLUDES) $(CUDA_INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./histogram_try2.o

histogram_fixed: histogram_fixed.cpp histogram_fixed.cu
	$(NVCC) -c histogram_fixed.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/histogram_fixedCUDA histogram_fixed.cpp histogram_fixed.o ../src/Timer.cpp $(INCLUDES) $(CUDA_INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./histogram_fixed.o

histogram_fixed_square: histogram_fixed_square.cpp histogram_fixed_square.cu
	$(NVCC) -c histogram_fixed_square.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/histogram_fixed_squareCUDA histogram_fixed_square.cpp histogram_fixed_square.o ../src/Timer.cpp $(INCLUDES) $(CUDA_INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./histogram_fixed_square.o

histogram2: histogram2.cpp histogram2.cu
	$(NVCC) -c histogram2.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/histogram2CUDA histogram2.cpp histogram2.o ../src/Timer.cpp $(INCLUDES) $(CUDA_INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./histogram2.o

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

histogram_uni: histogram_uni.cpp histogram_uni.cu
	$(NVCC) -c histogram_uni.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/histogram_uniCUDA histogram_uni.cpp histogram_uni.o ../src/Timer.cpp $(INCLUDES) $(CUDA_INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./histogram_uni.o

histogram_uni_more: histogram_uni_more.cpp histogram_uni_more.cu
	$(NVCC) -c histogram_uni_more.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/histogram_uni_moreCUDA histogram_uni_more.cpp histogram_uni_more.o ../src/Timer.cpp $(INCLUDES) $(CUDA_INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./histogram_uni_more.o

smooth: smooth.cpp smooth.cu
	$(NVCC) -c smooth.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/smoothCUDA smooth.cpp smooth.o ../src/Timer.cpp $(INCLUDES) $(CUDA_INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./smooth.o

smooth_more: smooth_more.cpp smooth_more.cu
	$(NVCC) -c smooth_more.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/smooth_moreCUDA smooth_more.cpp smooth_more.o ../src/Timer.cpp $(INCLUDES) $(CUDA_INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./smooth_more.o

smooth_more2: smooth_more2.cpp smooth_more2.cu
	$(NVCC) -c smooth_more2.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/smooth_more2CUDA smooth_more2.cpp smooth_more2.o ../src/Timer.cpp $(INCLUDES) $(CUDA_INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./smooth_more2.o

smooth2: smooth2.cpp smooth2.cu
	$(NVCC) -c smooth2.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/smooth2CUDA smooth2.cpp smooth2.o ../src/Timer.cpp $(INCLUDES) $(CUDA_INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./smooth2.o

smooth2_bigger: smooth2_bigger.cpp smooth2_bigger.cu
	$(NVCC) -c smooth2_bigger.cu $(INCLUDES) $(NVCCFLAGS)
	$(LINKER) -o $(PROJ_BASE)/bin/smooth2_biggerCUDA smooth2_bigger.cpp smooth2_bigger.o ../src/Timer.cpp $(INCLUDES) $(CUDA_INCLUDES) $(LIBS) $(CUDA_LIBS) $(CFLAGS) $(CUDA_LDFLAGS)
	rm -f ./smooth2_bigger.o


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

