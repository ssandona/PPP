
#include <Timer.hpp>
#include <iostream>
#include <iomanip>

using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

const unsigned int nrThreads = 256;
unsigned int PIXELS_THREAD = 10;

__global__ void kernel(const int width, const int height, const unsigned char * inputImage, unsigned char * outputDarkGrayImage) {
	unsigned int i;
	//loop over different pixels assigned per thread
	for(i = ((blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x) + threadIdx.x; i < width * height; i += (gridDim.x * blockDim.x) * (gridDim.y * blockDim.y)) {
		float grayPix = 0.0f;
		float r = static_cast< float >(inputImage[i]);
		float g = static_cast< float >(inputImage[(width * height) + i]);
		float b = static_cast< float >(inputImage[(2 * width * height) + i]);

		grayPix = ((0.3f * r) + (0.59f * g) + (0.11f * b));
		grayPix = (grayPix * 0.6f) + 0.5f;

		outputDarkGrayImage[i] = static_cast< unsigned char >(grayPix);
	}
}	

void darkGray(const int width, const int height, const unsigned char * inputImage, unsigned char * darkGrayImage) {
	cudaError_t devRetVal = cudaSuccess;
	//Timers. 
	NSTimer globalTimer("GlobalTimer", false, false);
	NSTimer kernelTimer = NSTimer("KernelTimer", false, false);
	NSTimer memoryTimer("MemoryTimer", false, false);
	//Device memory pointers.
	unsigned char* devInputImage = 0;
	unsigned char* devDarkGrayImage = 0;
	//Start of the computation.
	globalTimer.start();
	// Allocate CUDA memory
	if ( (devRetVal = cudaMalloc(reinterpret_cast< void ** >(&devInputImage), (3 * width * height) * sizeof(unsigned char))) != cudaSuccess ) {
		cerr << "Impossible to allocate device memory for input matrix." << endl;
		return;
	}
	if ( (devRetVal = cudaMalloc(reinterpret_cast< void ** >(&devDarkGrayImage), (width * height) * sizeof(unsigned char))) != cudaSuccess ) {
		cerr << "Impossible to allocate device memory for output matrix" << endl;
		return;
	}
	// Copy input to device
	memoryTimer.start();
	if ( (devRetVal = cudaMemcpy(devInputImage, reinterpret_cast< const void * >(inputImage), (3 * width * height) * sizeof(unsigned char), cudaMemcpyHostToDevice)) != cudaSuccess ) {
		cerr << "Impossible to copy the input matrix to device." << endl;
		return;
	}
	memoryTimer.stop();
	//Kernel
	unsigned int grid_width = static_cast< unsigned int >(ceil(width / static_cast< float >(nrThreads)));
    unsigned int grid_height = static_cast< unsigned int >(ceil(height / static_cast< float >(PIXELS_THREAD)));
    // Execute the kernel
    dim3 gridSize(grid_width, grid_height);
    dim3 blockSize(nrThreads);
    cout<<"grid size: "<<grid_width<<"x"<<grid_height<<" -> threads doing nothing -> "<<(grid_width*height*nrThreads)-width*height<<endl;
	kernelTimer.start();
	kernel<<< gridSize, blockSize >>>(width, height, devInputImage, devDarkGrayImage);
	cudaDeviceSynchronize();
	kernelTimer.stop();
	// Check if the kernel returned an error
	if ( (devRetVal = cudaGetLastError()) != cudaSuccess ) {
		cerr << "Uh, the kernel had some kind of issue: " << cudaGetErrorString(devRetVal) << endl;
		return;
	}
	// Copy the output back to host
	memoryTimer.start();
	if ( (devRetVal = cudaMemcpy(reinterpret_cast< void * >(darkGrayImage), devDarkGrayImage, (width * height) * sizeof(unsigned char), cudaMemcpyDeviceToHost)) != cudaSuccess ) {
		cerr << "Impossible to copy output to host." << endl;
		return;
	}
	memoryTimer.stop();
	globalTimer.stop();
	// Print the timers
	long GFLOPS = ((long)width * (long)height) * (long)(4 + 3);
	long GB = ((long)width * (long)height) * (long)(3 + 1) * (float)sizeof(unsigned char);
	cout << fixed << setprecision(6);
	cout << endl;
	cout << "Total (s): \t" << globalTimer.getElapsed() << endl;
	cout << "Kernel (s): \t" << kernelTimer.getElapsed() << endl;
	cout << "Memory (s): \t" << memoryTimer.getElapsed() << endl;
	cout << endl;
	cout << setprecision(3);
	cout << "GFLOP/s: \t" << (float)GFLOPS /  (1000000000.0f * kernelTimer.getElapsed()) << endl;
	cout << "GB/s: \t\t" << (float)GB / (kernelTimer.getElapsed() * 1000000000.0f) << endl;
	cout << endl;
}
