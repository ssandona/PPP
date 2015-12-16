#include <Timer.hpp>
#include <iostream>
#include <iomanip>
#include <CImg.h>
#include <string>
#include <cmath>

using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;
using cimg_library::CImg;
using std::string;

/*const unsigned int DIM = 16000000;
const unsigned int nrThreads = 256;*/
const unsigned int B_WIDTH = 16;
const unsigned int B_HEIGHT = 16;
consf int PIXELS_THREAD

__global__ void darkGrayKernel(const int width, const int height, const unsigned char *inputImage, unsigned char *darkGrayImage) {
    /*unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;*/

    //M[i,j]
    /*unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;*/

    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int globalIdx = j + (blockDim.x * gridDim.x * i);
    int k;

    for(k = 0; k < PIXELS_THREAD; k++) {
        if(globalIdx >= width * height) return;
        //if(j >= width || i >= height) return;
        float grayPix = 0.0f;
        float r = static_cast< float >(inputImage[globalIdx]);
        float g = static_cast< float >(inputImage[(width * height) + globalIdx]);
        float b = static_cast< float >(inputImage[(2 * width * height) + globalIdx]);
        grayPix = ((0.3f * r) + (0.59f * g) + (0.11f * b));
        grayPix = (grayPix * 0.6f) + 0.5f;
        //}
        darkGrayImage[globalIdx] = static_cast< unsigned char >(grayPix);
        globalIdx += (gridDim.x * blockDim.x) * (gridDim.y * blockDim.y);
    }



    int darkGray(const int width, const int height, const unsigned char *inputImage, unsigned char *darkGrayImage) {
        //cout << "FUNC\n";
        cudaError_t devRetVal = cudaSuccess;
        unsigned char *devInputImage = 0;
        unsigned char *devDarkGrayImage = 0;
        int pixel_numbers;

        NSTimer globalTimer("GlobalTimer", false, false);
        NSTimer kernelTimer("KernelTimer", false, false);
        NSTimer memoryTimer("MemoryTimer", false, false);

        int i, j;
        /*for(i = 0; i < width * height; i++) {
            cout << inputImage;
        }*/

        //cout << "FUNC1\n";
        pixel_numbers = width * height;

        // Start of the computation
        globalTimer.start();
        // Convert the input image to grayscale and make it darker
        //*outputImage = new unsigned char[pixel_numbers];

        //cout << "FUNC2\n";
        // Allocate CUDA memory
        if ( (devRetVal = cudaMalloc(reinterpret_cast< void ** >(&devInputImage), pixel_numbers * 3 * sizeof(unsigned char))) != cudaSuccess ) {
            cerr << "Impossible to allocate device memory for inputImage." << endl;
            return 1;
        }
        if ( (devRetVal = cudaMalloc(reinterpret_cast< void ** >(&devDarkGrayImage), pixel_numbers * sizeof(unsigned char))) != cudaSuccess ) {
            cerr << "Impossible to allocate device memory for darkGrayImage." << endl;
            return 1;
        }
        //cout << "FUNC3\n";

        // Copy input to device
        memoryTimer.start();
        if ( (devRetVal = cudaMemcpy(devInputImage, (void *)(inputImage), pixel_numbers * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice)) != cudaSuccess ) {
            cerr << "Impossible to copy inputImage to device." << endl;
            return 1;
        }

        /*if ( (devRetVal = cudaMemcpy(devDarkGrayImage, reinterpret_cast< void *>(*outputImage), pixel_numbers * sizeof(unsigned char), cudaMemcpyHostToDevice)) != cudaSuccess ) {
            cerr << "Impossible to copy outputImage to device." << endl;
            return 1;
        }*/
        memoryTimer.stop();

        //cout << "FUNC4\n";
        //int grid_width = width % B_WIDTH == 0 ? width / B_WIDTH : width / B_WIDTH + 1;
        //int grid_height = height % B_HEIGHT == 0 ? height / B_HEIGHT : height / B_HEIGHT + 1;

        //cout << "Image size (w,h): (" << width << ", " << height << ")\n";
        //cout << "Grid size (w,h): (" << grid_width << ", " << grid_height << ")\n";

        //unsigned int grid_width=static_cast< unsigned int >(ceil(width / static_cast< float >(B_WIDTH)));
        //unsigned int grid_height=static_cast< unsigned int >(ceil(height / static_cast< float >(B_HEIGHT)));
        // Execute the kernel
        unsigned int grid_size = static_cast< unsigned int >(ceil(sqrt(ceil(width * height / PIXELS_THREAD) / (float)THREAD_NUMBER)));

        dim3 gridSize(grid_size, grid_size);
        dim3 blockSize(B_WIDTH, B_HEIGHT);

        kernelTimer.start();
        //cout << "FUNC5\n";
        darkGrayKernel <<< gridSize, blockSize >>>(width, height, devInputImage, devDarkGrayImage);
        cudaDeviceSynchronize();
        kernelTimer.stop();
        //cout << "FUNC6\n";
        // Check if the kernel returned an error
        if ( (devRetVal = cudaGetLastError()) != cudaSuccess ) {
            cerr << "Uh, the kernel had some kind of issue: " << cudaGetErrorString(devRetVal) << endl;
            return 1;
        }
        //cout << "FUNC7\n";
        // Copy the output back to host
        memoryTimer.start();
        if ( (devRetVal = cudaMemcpy(reinterpret_cast< void *>(darkGrayImage), devDarkGrayImage, pixel_numbers * sizeof(unsigned char), cudaMemcpyDeviceToHost)) != cudaSuccess ) {
            cerr << "Impossible to copy devC to host." << endl;
            return 1;
        }
        memoryTimer.stop();

        globalTimer.stop();
        //cout << "FUNC8\n";
        //darkGrayImage._data = outputImage;
        // Time GFLOP/s GB/s
        cout << fixed << setprecision(6) << kernelTimer.getElapsed() << setprecision(3) << " " << (static_cast< long long unsigned int >(width) * height * 7) / 1000000000.0 / kernelTimer.getElapsed() << " " << (static_cast< long long unsigned int >(width) * height * (4 * sizeof(unsigned char))) / 1000000000.0 / kernelTimer.getElapsed() << endl;


        // Print the timers
        cout << "Total (s): \t" << globalTimer.getElapsed() << endl;
        cout << "Kernel (s): \t" << kernelTimer.getElapsed() << endl;
        cout << "Memory (s): \t" << memoryTimer.getElapsed() << endl;
        cout << endl;

        cudaFree(devInputImage);
        cudaFree(devDarkGrayImage);
        return 0;
    }