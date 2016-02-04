
#include <Timer.hpp>
#include <iostream>
#include <iomanip>

using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;


__constant__ float filter[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 2.0f, 3.0f, 2.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};


__global__ void triangularSmoothDKernel(const int width, const int height, const int spectrum, unsigned char *inputImage, unsigned char *smoothImage) {
    //indexes of the first assigned pixel
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    //loop over different pixels associated per thread
    while(j < width && i < height) {

        for ( int z = 0; z < spectrum; z++ ) {
            unsigned int filterItem = 0;
            float filterSum = 0.0f;
            float smoothPix = 0.0f;

            for (int fy = i - 2; fy < i + 3; fy++ ) {
                if ( fy < 0 ) {
                    filterItem += 5;
                    continue;
                } else if ( fy == height ) {
                    break;
                }

                for ( int fx = j - 2; fx < j + 3; fx++ ) {
                    if ( (fx < 0) || (fx >= width) ) {
                        filterItem++;
                        continue;
                    }

                    smoothPix += static_cast< float >(inputImage[(z * width * height) + (fy * width) + fx]) * filter[filterItem];
                    filterSum += filter[filterItem];
                    filterItem++;
                }
            }

            smoothPix /= filterSum;
            smoothImage[(z * width * height) + (i * width) + j] = static_cast< unsigned char >(smoothPix + 0.5f);
        }
        i+=(gridDim.y * blockDim.y);
    }
}


int triangularSmooth(const int width, const int height, const int spectrum, unsigned char *inputImage, unsigned char *smoothImage) {
    cudaError_t devRetVal = cudaSuccess;
    unsigned char *devInputImage = 0;
    unsigned char *devSmoothImage = 0;

    unsigned int B_WIDTH = 32;
    unsigned int B_HEIGHT = 16;
    unsigned int grid_height=15;

    int pixel_numbers;

    NSTimer globalTimer("GlobalTimer", false, false);
    NSTimer kernelTimer("KernelTimer", false, false);
    NSTimer memoryTimer("MemoryTimer", false, false);


    pixel_numbers = width * height;

    // Start of the computation
    globalTimer.start();

    // Allocate CUDA memory
    if ( (devRetVal = cudaMalloc(reinterpret_cast< void ** >(&devInputImage), pixel_numbers * 3 * sizeof(unsigned char))) != cudaSuccess ) {
        cerr << "Impossible to allocate device memory for inputImage." << endl;
        return 1;
    }
    if ( (devRetVal = cudaMalloc(reinterpret_cast< void ** >(&devSmoothImage), pixel_numbers * 3 * sizeof(unsigned char))) != cudaSuccess ) {
        cerr << "Impossible to allocate device memory for darkGrayImage." << endl;
        return 1;
    }

    // Copy input to device
    memoryTimer.start();
    if ( (devRetVal = cudaMemcpy(devInputImage, (void *)(inputImage), pixel_numbers * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice)) != cudaSuccess ) {
        cerr << "Impossible to copy inputImage to device." << endl;
        return 1;
    }

    memoryTimer.stop();

    // Grid width calculation (dynamic)
    unsigned int grid_width = static_cast< unsigned int >(ceil(width / static_cast< float >(B_WIDTH)));
    
    // Execute the kernel
    dim3 gridSize(grid_width, grid_height);
    dim3 blockSize(B_WIDTH, B_HEIGHT);
    kernelTimer.start();
    triangularSmoothDKernel <<< gridSize, blockSize >>>(width, height, spectrum, devInputImage, devSmoothImage);
    cudaDeviceSynchronize();
    kernelTimer.stop();

    // Check if the kernel returned an error
    if ( (devRetVal = cudaGetLastError()) != cudaSuccess ) {
        cerr << "Uh, the kernel had some kind of issue: " << cudaGetErrorString(devRetVal) << endl;
        return 1;
    }

    // Copy the output back to host
    memoryTimer.start();
    if ( (devRetVal = cudaMemcpy(reinterpret_cast< void *>(smoothImage), devSmoothImage, pixel_numbers * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost)) != cudaSuccess ) {
        cerr << "Impossible to copy devC to host." << endl;
        return 1;
    }
    memoryTimer.stop();

    globalTimer.stop();

    // Time GFLOP/s GB/s
    long GFLOPS = ((long)width * (long)height) * (long)(4*25*3+3);
    long GB = ((long)width * (long)height) * (long)(1*25*3+3) * (float)sizeof(unsigned char);
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

    cudaFree(devInputImage);
    cudaFree(devSmoothImage);
    return 0;
}
