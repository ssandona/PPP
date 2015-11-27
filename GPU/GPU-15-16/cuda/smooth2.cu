
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

    if(j >= width || i >= height) return;

    unsigned int inBlockIdx = threadIdx.x + (blockDim.x * threadIdx.y);
    //unsigned int globalIdx = j + (width * i);

    __shared__ unsigned char localImagePortion[20 * 20 * 3];

    //int topLeftPx=(globalIdx-2*width-2);
    int topLeftPxI = (blockIdx.y * blockDim.y) - 2;
    int topLeftPxJ = (blockIdx.x * blockDim.x) - 2;

    int pxAI = topLeftPxI + (inBlockIdx / 20);
    int pxAJ = topLeftPxJ + (inBlockIdx % 20);
    int pxA = pxAJ + (width * pxAI);

    int inLocalPortionI = threadIdx.y + 2;
    int inLocalPortionJ = threadIdx.x + 2;

    int imageIdxI = pxAI - topLeftPxI;
    int imageIdxJ = pxAJ - topLeftPxJ;
    int imageIdx = imageIdxJ + (20 * imageIdxI);


    //from ai,aj to inside (20,20)

    if(pxAI >= 0 && pxAI < height && pxAJ >= 0 && pxAJ < width) {
        //localImagePortion[imageIdx] = inputImage[pxA];
        //localImagePortion[imageIdx + 20 * 20] = inputImage[pxA + (B_WIDTH * B_HEIGHT)];
        localImagePortion[imageIdx + 2 * 20 * 20] = inputImage[pxA + 2 * (B_WIDTH * B_HEIGHT)];
    }

    int newInBlockIdx = inBlockIdx + 16 * 16;

    pxAI = topLeftPxI + (newInBlockIdx / 20);
    pxAJ = topLeftPxJ + (newInBlockIdx % 20);
    pxA = pxAJ + (width * pxAI);

    imageIdxI = pxAI - topLeftPxI;
    imageIdxJ = pxAJ - topLeftPxJ;
    imageIdx = imageIdxJ + (20 * imageIdxI);

    if(pxAI >= 0 && pxAI < height && pxAJ >= 0 && pxAJ < width && imageIdx<20*20) {
        //localImagePortion[imageIdx] = inputImage[pxA];
        //localImagePortion[imageIdx + 20 * 20] = inputImage[pxA + (B_WIDTH * B_HEIGHT)];
        localImagePortion[imageIdx + 2 * 20 * 20] = inputImage[pxA + 2 * (B_WIDTH * B_HEIGHT)];
    }

    __syncthreads();


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
        //smoothImage[(z * width * height) + (i * width) + j] = static_cast< unsigned char >(smoothPix + 0.5f);

        smoothImage[(z * width * height) + (i * width) + j] = localImagePortion[(z * 20 * 20) + (inLocalPortionI * 20) + inLocalPortionJ];
        if(z ==1) {
            smoothImage[(z * width * height) + (i * width) + j] = static_cast< unsigned char >(0.0f);

        }
    }
}


int triangularSmooth(const int width, const int height, const int spectrum, unsigned char *inputImage, unsigned char *smoothImage) {
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
    unsigned int grid_height = static_cast< unsigned int >(ceil(height / static_cast< float >(B_HEIGHT)));
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