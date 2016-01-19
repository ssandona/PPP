
#include <Timer.hpp>
#include <iostream>
#include <iomanip>

using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

const unsigned int B_WIDTH = 16;
const unsigned int B_HEIGHT = 16;

__constant__ float filter[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 2.0f, 3.0f, 2.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};


__global__ void triangularSmoothDKernel(const int width, const int height, const int spectrum, unsigned char *inputImage, unsigned char *smoothImage) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    //index inside the block (from 0 to 255)
    int inBlockIdx = threadIdx.x + (blockDim.x * threadIdx.y);

    //the block is 16x16, but to apply the filter over these 256 pixels, also some
    //external pixels are needed. The filter is 5x5 so we need also the 2 px border of
    //the 16x16 portion.
    __shared__ unsigned char localImagePortion[20 * 20 * 3];

    //coordinates of the top left pixel for the localImagePortion
    int topLeftPxI = (blockIdx.y * blockDim.y) - 2;
    int topLeftPxJ = (blockIdx.x * blockDim.x) - 2;

    //coordinates of the first pixel to copy into the localImagePortion
    int pxAI = topLeftPxI + (inBlockIdx / 20);
    int pxAJ = topLeftPxJ + (inBlockIdx % 20);
    int pxA = pxAJ + (width * pxAI);

    //coordinates of the pixel inside the localImagePortion to which this thread has to apply
    //the filter
    int inLocalPortionI = threadIdx.y + 2;
    int inLocalPortionJ = threadIdx.x + 2;

    //coordinates of the localImagePortion in which copy the pixel
    int imageIdxI = pxAI - topLeftPxI;
    int imageIdxJ = pxAJ - topLeftPxJ;
    int imageIdx = imageIdxJ + (20 * imageIdxI);


    //if the first pixel to copy is not out of the image, copy it into the localImagePortion
    if(pxAI >= 0 && pxAI < height && pxAJ >= 0 && pxAJ < width) {
        localImagePortion[imageIdx] = inputImage[pxA];
        localImagePortion[imageIdx + 20 * 20] = inputImage[pxA + (width * height)];
        localImagePortion[imageIdx + 2 * 20 * 20] = inputImage[pxA + 2 * (width * height)];
    }

    //displacement to calculate the second pixel to add to the localImagePortion
    int newInBlockIdx = inBlockIdx + 16 * 16;

    //coordinates of the second pixel to copy into the localImagePortion
    pxAI = topLeftPxI + (newInBlockIdx / 20);
    pxAJ = topLeftPxJ + (newInBlockIdx % 20);
    pxA = pxAJ + (width * pxAI);

    //coordinates of the localImagePortion in which copy the pixel
    imageIdxI = pxAI - topLeftPxI;
    imageIdxJ = pxAJ - topLeftPxJ;
    imageIdx = imageIdxJ + (20 * imageIdxI);

    //if the second pixel to copy is not out of the image, copy it into the localImagePortion
    if(pxAI >= 0 && pxAI < height && pxAJ >= 0 && pxAJ < width && imageIdx < 20 * 20) {
        localImagePortion[imageIdx] = inputImage[pxA];
        localImagePortion[imageIdx + 20 * 20] = inputImage[pxA + (width * height)];
        localImagePortion[imageIdx + 2 * 20 * 20] = inputImage[pxA + 2 * (width * height)];
    }

    __syncthreads();

    while(j < width && i < height) {
        //same code as the sequential, but with indexes of the localImagePortion
        for ( int z = 0; z < spectrum; z++ ) {
            unsigned int filterItem = 0;
            float filterSum = 0.0f;
            float smoothPix = 0.0f;

            for (int fy = i - 2, localFy = inLocalPortionI - 2 ; fy < i + 3; fy++, localFy++) {
                if ( fy < 0 ) {
                    filterItem += 5;
                    continue;
                } else if ( fy == height ) {
                    break;
                }

                for ( int fx = j - 2, localFx = inLocalPortionJ - 2; fx < j + 3; fx++, localFx++) {
                    if ( (fx < 0) || (fx >= width) ) {
                        filterItem++;
                        continue;
                    }

                    smoothPix += static_cast< float >(localImagePortion[(z * 20 * 20) + (localFy * 20) + localFx]) * filter[filterItem];
                    filterSum += filter[filterItem];
                    filterItem++;
                }
            }

            smoothPix /= filterSum;
            smoothImage[(z * width * height) + (i * width) + j] = static_cast< unsigned char >(smoothPix + 0.5f);
        }
        i += (gridDim.y * blockDim.y);
    }
}


int triangularSmooth(const int width, const int height, const int spectrum, unsigned char *inputImage, unsigned char *smoothImage, int pixelThreads) {
    cudaError_t devRetVal = cudaSuccess;
    unsigned char *devInputImage = 0;
    unsigned char *devSmoothImage = 0;

    int pixel_numbers;

    NSTimer globalTimer("GlobalTimer", false, false);
    NSTimer kernelTimer("KernelTimer", false, false);
    NSTimer memoryTimer("MemoryTimer", false, false);


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

    unsigned int grid_width = static_cast< unsigned int >(ceil(width / static_cast< float >(B_WIDTH)));
    unsigned int grid_height = static_cast< unsigned int >(ceil(ceil(height / static_cast< float >(B_HEIGHT)) / static_cast< float >(pixelThreads)));
    // Execute the kernel
    dim3 gridSize(grid_width, grid_height);
    dim3 blockSize(B_WIDTH, B_HEIGHT);

    kernelTimer.start();
    triangularSmoothDKernel <<< gridSize, blockSize >>>(width, height, spectrum, devInputImage, devSmoothImage);
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
    if ( (devRetVal = cudaMemcpy(reinterpret_cast< void *>(smoothImage), devSmoothImage, pixel_numbers * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost)) != cudaSuccess ) {
        cerr << "Impossible to copy devC to host." << endl;
        return 1;
    }
    memoryTimer.stop();

    globalTimer.stop();
    //cout << "FUNC8\n";
    //darkGrayImage._data = outputImage;
    // Time GFLOP/s GB/s
    cout << fixed << setprecision(6) << kernelTimer.getElapsed() << setprecision(3) << " " << (static_cast< long long unsigned int >(width) * height * 6) / 1000000000.0 / kernelTimer.getElapsed() << " " << (static_cast< long long unsigned int >(width) * height * ((4 * sizeof(unsigned char)) + (1 * sizeof(unsigned int)))) / 1000000000.0 / kernelTimer.getElapsed() << endl;


    // Print the timers
    cout << "Total (s): \t" << globalTimer.getElapsed() << endl;
    cout << "Kernel (s): \t" << kernelTimer.getElapsed() << endl;
    cout << "Memory (s): \t" << memoryTimer.getElapsed() << endl;
    cout << endl;

    cudaFree(devInputImage);
    cudaFree(devSmoothImage);
    return 0;
}