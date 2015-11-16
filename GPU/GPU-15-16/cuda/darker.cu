
#include <iostream>
#include <Timer.hpp>
#include <cmath>
#include <iomanip>
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;
using LOFAR::NSTimer;

/*const unsigned int DIM = 16000000;
const unsigned int nrThreads = 256;*/
const unsigned int B_WIDTH = 32;
const unsigned int B_HEIGHT = 16;

__global__ void darkGray(unsigned int height, unsigned int width, unsigned char *inputImage, unsigned char *outputImage) {
	int x = blockIdx.y * blockDim.y + threadIdx.y;
	int y = blockIdx.x * blockDim.x + threadIdx.x;
	if(x >= height || y >= width) return;

	float grayPix = 0.0f;
	float r = static_cast< float >(inputImage[(y * width) + x]);
	float g = static_cast< float >(inputImage[(width * height) + (y * width) + x]);
	float b = static_cast< float >(inputImage[(2 * width * height) + (y * width) + x]);

	grayPix = ((0.3f * r) + (0.59f * g) + (0.11f * b));
	grayPix = (grayPix * 0.6f) + 0.5f;

	outputImage[(y * width) + x] = static_cast< unsigned char >(grayPix);
}



int main(void) {
	cudaError_t devRetVal = cudaSuccess;
	CImg< unsigned char > inputImage;
	unsigned char * devInputImage;
	CImg< unsigned char > darkGrayImage;
	unsigned char * devDarkGrayImage;
	unsigned char * outputImage;
	int height;
	int width;
	int* devHeight;
	int* devWidth;
	int pixel_numbers;
	
	NSTimer globalTimer("GlobalTimer", false, false);
	NSTimer kernelTimer("KernelTimer", false, false);
	NSTimer memoryTimer("MemoryTimer", false, false);


	if ( argc != 2 ) {
		cerr << "Usage: " << argv[0] << " <filename>" << endl;
		return 1;
	}

	// Load the input image
	CImg< unsigned char > inputImage = CImg< unsigned char >(argv[1]);
	if ( inputImage.spectrum() != 3 ) {
		cerr << "The input must be a color image." << endl;
		return 1;
	}

	imageSize=inputImage.width() * inputImage.height();

	// Start of the computation
	globalTimer.start();

	// Convert the input image to grayscale and make it darker
	outputImage = malloc(pixel_numbers * sizeof(unsigned char));

	// Allocate CUDA memory
	if ( (devRetVal = cudaMalloc(reinterpret_cast< void ** >(&devInputImage), pixel_numbers * sizeof(unsigned char))) != cudaSuccess ) {
		cerr << "Impossible to allocate device memory for inputImage." << endl;
		return 1;
	}
	if ( (devRetVal = cudaMalloc(reinterpret_cast< void ** >(&devDarkGrayImage), pixel_numbers * sizeof(unsigned char))) != cudaSuccess ) {
		cerr << "Impossible to allocate device memory for darkGrayImage." << endl;
		return 1;
	}

	// Copy input to device
	memoryTimer.start();
	if ( (devRetVal = cudaMemcpy(devInputImage, reinterpret_cast< void * >(inputImage.data()), pixel_numbers * sizeof(unsigned char), cudaMemcpyHostToDevice)) != cudaSuccess ) {
		cerr << "Impossible to copy devA to device." << endl;
		return 1;
	}
	memoryTimer.stop();


	int grid_width=inputImage.width()%B_WIDTH==0?inputImage.width()/B_WIDTH:inputImage.width()/B_WIDTH+1;
	int grid_height=inputImage.width()%B_HEIGHT==0?inputImage.height()/B_HEIGHT:inputImage.height()/B_HEIGHT+1;

	// Execute the kernel
	dim3 gridSize(tatic_cast< unsigned int >(ceil(inputImage.height() / static_cast< float >(B_HEIGHT))), static_cast< unsigned int >(ceil(inputImage.width() / static_cast< float >(B_WIDTH))));
	dim3 blockSize(B_WIDTH*B_HEIGHT);

	kernelTimer.start();
	darkGray<<< gridSize, blockSize >>>(grid_height, grid_width, devInputImage, devDarkGrayImage);
	cudaDeviceSynchronize();
	kernelTimer.stop();

	// Check if the kernel returned an error
	if ( (devRetVal = cudaGetLastError()) != cudaSuccess ) {
		cerr << "Uh, the kernel had some kind of issue: " << cudaGetErrorString(devRetVal) << endl;
		return 1;
	}

	// Copy the output back to host
	memoryTimer.start();
	if ( (devRetVal = cudaMemcpy(reinterpret_cast< void * >(outputImage), devDarkGrayImage, pixel_numbers * sizeof(unsigned char), cudaMemcpyDeviceToHost)) != cudaSuccess ) {
		cerr << "Impossible to copy devC to host." << endl;
		return 1;
	}
	memoryTimer.stop();
	darkGrayImage._data=outputImage;
	//CImg<float> darkGrayImage(matrix,width,height,1,1,true); 

	// End of the computation
	globalTimer.stop();

	// Print the timers
	cout << fixed << setprecision(6);
	cout << endl;
	cout << "Total (s): \t" << globalTimer.getElapsed() << endl;
	cout << "Kernel (s): \t" << kernelTimer.getElapsed() << endl;
	cout << "Memory (s): \t" << memoryTimer.getElapsed() << endl;
	cout << endl;
	cout << setprecision(3);
	cout << "GFLOP/s: \t" << (DIM / kernelTimer.getElapsed()) / 1000000000.0 << endl;
	cout << "GB/s: \t\t" << ((12 * DIM) / kernelTimer.getElapsed()) / 1000000000.0 << endl;
	cout << endl;

	// Save output
	darkGrayImage.save(("./" + string(argv[1]) + ".dark.seq.bmp").c_str());

	cudaFree(devInputImage);
	cudaFree(devDarkGrayImage);
	free(outputImage);

	return 0;
}