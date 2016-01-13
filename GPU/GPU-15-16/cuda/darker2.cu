
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

__global__ void kernel(const int width, const int height, const unsigned char * inputImage, unsigned char * outputDarkGrayImage) {
	unsigned int item = (blockIdx.x * blockDim.x + threadIdx.x)+(blockIdx.y * width);

	/*unsigned int i = ;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int globalIdx=(blockIdx.x * blockDim.x + threadIdx.x)+(threadIdx.y * width);*/

	if ( item < width * height ) {
		float grayPix = 0.0f;
		float r = static_cast< float >(inputImage[item]);
		float g = static_cast< float >(inputImage[(width * height) + item]);
		float b = static_cast< float >(inputImage[(2 * width * height) + item]);

		grayPix = ((0.3f * r) + (0.59f * g) + (0.11f * b));
		grayPix = (grayPix * 0.6f) + 0.5f;

		outputDarkGrayImage[item] = static_cast< unsigned char >(grayPix);
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
    //unsigned int grid_height = static_cast< unsigned int >(ceil(height / static_cast< float >(B_HEIGHT)));
    // Execute the kernel
    dim3 gridSize(grid_width, height);
    dim3 blockSize(nrThreads);
    cout<<"grid size: "<<grid_width<<"x"<<height<<" -> threads doing nothing -> "<<(width*height-grid_width*height*nrThreads)<<endl;
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
	long Gflops = ((long)width * (long)height) * (long)(4 + 3);
	long GB = ((long)width * (long)height) * (long)(3 + 1) * (float)sizeof(unsigned char);
	cout << fixed << setprecision(6);
	cout << endl;
	cout << "Total (s): \t" << globalTimer.getElapsed() << endl;
	cout << "Kernel (s): \t" << kernelTimer.getElapsed() << endl;
	cout << "Memory (s): \t" << memoryTimer.getElapsed() << endl;
	cout << endl;
	cout << setprecision(3);
	cout << "GFLOP/s: \t" << (float)Gflops /  (1000000000.0f * kernelTimer.getElapsed()) << endl;
	cout << "GB/s: \t\t" << (float)GB / (kernelTimer.getElapsed() * 1000000000.0f) << endl;
	cout << endl;
}
