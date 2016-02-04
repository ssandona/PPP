#include <Timer.hpp>
#include <iostream>
#include <iomanip>

using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

const int HISTOGRAM_SIZE = 256;
const unsigned int B_WIDTH = 32;
const unsigned int B_HEIGHT = 8;
const int WARPS = 8;
const unsigned int WARP_SIZE = 32;
const int grid_height = 60;
const int grid_width = 45;


__global__ void histogram1DKernel(const int width, const int height, const unsigned char *inputImage, unsigned char *grayImage, unsigned int *histogram) {
    int k;
    unsigned int i;
    unsigned char last_gray_px;
    int cont = 0;

    unsigned int inBlockIdx = threadIdx.x;
    unsigned int warpid = inBlockIdx / WARP_SIZE;

    __shared__ unsigned int localHistogram[WARPS * HISTOGRAM_SIZE];

    for(k = 0; k < WARPS; k++) {
        localHistogram[k * 256 + inBlockIdx] = 0;
    }

    __syncthreads();


    for(i = ((blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x) + threadIdx.x; i < width * height; i += (gridDim.x * blockDim.x) * (gridDim.y * blockDim.y)) {
        float grayPix = 0.0f;
        float r = static_cast< float >(inputImage[i]);
        float g = static_cast< float >(inputImage[(width * height) + i]);
        float b = static_cast< float >(inputImage[(2 * width * height) + i]);

        grayPix = ((0.3f * r) + (0.59f * g) + (0.11f * b)) + 0.5f;
        //}
        grayImage[i] = static_cast< unsigned char >(grayPix);
        if(cont==0 || grayPix == last_gray_px){
        	cont+=1;
        }
        else {
            atomicAdd((unsigned int *)&localHistogram[warpid * 256 + static_cast< unsigned int >(last_gray_px)], cont);
        	cont=1;
        }
        last_gray_px=grayPix;
    }
    atomicAdd((unsigned int *)&localHistogram[warpid * 256 + static_cast< unsigned int >(last_gray_px)], cont);

    __syncthreads();


    int s = 0;
    for(k = 0; k < WARPS; k++) {
        s += localHistogram[k * 256 + inBlockIdx];
    }

    atomicAdd((unsigned int *)&histogram[inBlockIdx], s);

}



int histogram1D(const int width, const int height, const unsigned char *inputImage, unsigned char *grayImage, unsigned int *histogram) {
    cudaError_t devRetVal = cudaSuccess;
    unsigned char *devInputImage = 0;
    unsigned char *devGrayImage = 0;
    unsigned int *devHistogram = 0;

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
    if ( (devRetVal = cudaMalloc(reinterpret_cast< void ** >(&devGrayImage), pixel_numbers * sizeof(unsigned char))) != cudaSuccess ) {
        cerr << "Impossible to allocate device memory for darkGrayImage." << endl;
        return 1;
    }

    if ( (devRetVal = cudaMalloc(reinterpret_cast< void ** >(&devHistogram), HISTOGRAM_SIZE * sizeof(unsigned int))) != cudaSuccess ) {
        cerr << "Impossible to allocate device memory for histogram." << endl;
        return 1;
    }




    // Copy input to device
    memoryTimer.start();
    if ( (devRetVal = cudaMemcpy(devInputImage, (void *)(inputImage), pixel_numbers * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice)) != cudaSuccess ) {
        cerr << "Impossible to copy inputImage to device." << endl;
        return 1;
    }

    if ( (devRetVal = cudaMemcpy(devHistogram, (void *)(histogram), HISTOGRAM_SIZE * sizeof(unsigned int), cudaMemcpyHostToDevice)) != cudaSuccess ) {
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

    // Execute the kernel
    dim3 gridSize(grid_width, grid_height);
    dim3 blockSize(256);

    kernelTimer.start();
    //cout << "FUNC5\n";
    histogram1DKernel <<< gridSize, blockSize >>>(width, height, devInputImage, devGrayImage, devHistogram);
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
    if ( (devRetVal = cudaMemcpy(reinterpret_cast< void *>(grayImage), devGrayImage, pixel_numbers * sizeof(unsigned char), cudaMemcpyDeviceToHost)) != cudaSuccess ) {
        cerr << "Impossible to copy devC to host." << endl;
        return 1;
    }
    if ( (devRetVal = cudaMemcpy(reinterpret_cast< void *>(histogram), devHistogram, HISTOGRAM_SIZE * sizeof(unsigned int), cudaMemcpyDeviceToHost)) != cudaSuccess ) {
        cerr << "Impossible to copy devC to host." << endl;
        return 1;
    }
    memoryTimer.stop();

    globalTimer.stop();
    //cout << "FUNC8\n";
    //darkGrayImage._data = outputImage;
    // Time GFLOP/s GB/s
    long Gflops = ((long)width * (long)height) * (long)(3 + 3 + 1) + (grid_height * grid_width) * (HISTOGRAM_SIZE + (HISTOGRAM_SIZE * WARPS));
    long GB = ((long)width * (long)height) * (long)(3 + 1) * (float)sizeof(unsigned char) + (grid_height * grid_width) * (HISTOGRAM_SIZE * (float)sizeof(unsigned int));
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

    cudaFree(devInputImage);
    cudaFree(devGrayImage);
    return 0;
}
