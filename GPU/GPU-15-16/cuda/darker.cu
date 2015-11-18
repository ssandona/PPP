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

__global__ void darkGrayKernel(unsigned int width, unsigned int height, unsigned char *inputImage, unsigned char *outputImage) {
    /*unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;*/

    //M[i,j]
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    
    if(j >= width || i >= height) return;

    float grayPix = 0.0f;
    float r = static_cast< float >(inputImage[(i * width) + j]);
    float g = static_cast< float >(inputImage[(width * height) + (i * width) + j]);
    float b = static_cast< float >(inputImage[(2 * width * height) + (i * width) + j]);

    grayPix = ((0.3f * r) + (0.59f * g) + (0.11f * b));
    grayPix = (grayPix * 0.6f) + 0.5f;

    if(blockIdx.x==0 && blockIdx.y==0){
        outputImage[(i * width) + j] = static_cast< unsigned char >(grayPix);
    }
    else {
        outputImage[(i * width) + j])=inputImage[(i * width) + j]);
        outputImage[(width * height) + (i * width) + j]=inputImage[(width * height) + (i * width) + j]
        outputImage[(2 * width * height) + (i * width) + j]=inputImage[(2 * width * height) + (i * width) + j];
    }
}



int darkGray(const int width, const int height, unsigned char *inputImage, unsigned char **outputImage) {
    cout << "FUNC\n";
    cudaError_t devRetVal = cudaSuccess;
    unsigned char *devInputImage=0;
    unsigned char *devDarkGrayImage=0;
    int pixel_numbers;

    NSTimer globalTimer("GlobalTimer", false, false);
    NSTimer kernelTimer("KernelTimer", false, false);
    NSTimer memoryTimer("MemoryTimer", false, false);

    int i,j;
    for(i=0;i<width*height;i++){
        cout << inputImage;
    }

    cout << "FUNC1\n";
    pixel_numbers=width * height;

    // Start of the computation
    globalTimer.start();
    // Convert the input image to grayscale and make it darker
    //*outputImage = new unsigned char[pixel_numbers];

    cout << "FUNC2\n";
    // Allocate CUDA memory
    if ( (devRetVal = cudaMalloc(reinterpret_cast< void ** >(&devInputImage), pixel_numbers * 3 * sizeof(unsigned char))) != cudaSuccess ) {
        cerr << "Impossible to allocate device memory for inputImage." << endl;
        return 1;
    }
    if ( (devRetVal = cudaMalloc(reinterpret_cast< void ** >(&devDarkGrayImage), pixel_numbers * 3 * sizeof(unsigned char))) != cudaSuccess ) {
        cerr << "Impossible to allocate device memory for darkGrayImage." << endl;
        return 1;
    }
    cout << "FUNC3\n";

    // Copy input to device
    memoryTimer.start();
    if ( (devRetVal = cudaMemcpy(devInputImage, reinterpret_cast< void *>(inputImage), pixel_numbers * sizeof(unsigned char), cudaMemcpyHostToDevice)) != cudaSuccess ) {
        cerr << "Impossible to copy inputImage to device." << endl;
        return 1;
    }

    /*if ( (devRetVal = cudaMemcpy(devDarkGrayImage, reinterpret_cast< void *>(*outputImage), pixel_numbers * sizeof(unsigned char), cudaMemcpyHostToDevice)) != cudaSuccess ) {
        cerr << "Impossible to copy outputImage to device." << endl;
        return 1;
    }*/
    memoryTimer.stop();

    cout << "FUNC4\n";
    int grid_width = width % B_WIDTH == 0 ? width / B_WIDTH : width / B_WIDTH + 1;
    int grid_height = height % B_HEIGHT == 0 ? height / B_HEIGHT : height / B_HEIGHT + 1;

    cout << "Image size (w,h): (" << width << ", " << height << ")\n";
    cout << "Grid size (w,h): (" << grid_width << ", " << grid_height << ")\n";

    // Execute the kernel
    dim3 gridSize(grid_width, grid_height,1);
    dim3 blockSize(B_WIDTH,B_HEIGHT,1);
    kernelTimer.start();
    cout << "FUNC5\n";
    darkGrayKernel <<< gridSize, blockSize >>>(width, height, devInputImage, devDarkGrayImage);
    cudaDeviceSynchronize();
    kernelTimer.stop();
    cout << "FUNC6\n";
    // Check if the kernel returned an error
    if ( (devRetVal = cudaGetLastError()) != cudaSuccess ) {
        cerr << "Uh, the kernel had some kind of issue: " << cudaGetErrorString(devRetVal) << endl;
        return 1;
    }
    cout << "FUNC7\n";
    // Copy the output back to host
    memoryTimer.start();
    if ( (devRetVal = cudaMemcpy(reinterpret_cast< void *>(*outputImage), devDarkGrayImage, pixel_numbers * sizeof(unsigned char), cudaMemcpyDeviceToHost)) != cudaSuccess ) {
        cerr << "Impossible to copy devC to host." << endl;
        return 1;
    }
    memoryTimer.stop();
    cout << "FUNC8\n";
    //darkGrayImage._data = outputImage;
    // Time GFLOP/s GB/s
    cout << fixed << setprecision(6) << kernelTimer.getElapsed() << setprecision(3) << " " << (static_cast< long long unsigned int >(width) * height * 7) / 1000000000.0 / kernelTimer.getElapsed() << " " << (static_cast< long long unsigned int >(width) * height * (4 * sizeof(unsigned char))) / 1000000000.0 / kernelTimer.getElapsed() << endl;
     

    // Print the timers
    cout << "Total (s): \t" << globalTimer.getElapsed() << endl;
    cout << "Kernel (s): \t" << kernelTimer.getElapsed() << endl;
    cout << "Memory (s): \t" << memoryTimer.getElapsed() << endl;
    cout << endl;
    globalTimer.stop();

    // Save output
    //darkGrayImage.save(("./" + string(argv[1]) + ".dark.seq.bmp").c_str());

    cudaFree(devInputImage);
    cudaFree(devDarkGrayImage);
    return 0;
}
/*
int main(int argc, char *argv[]) {
    cudaError_t devRetVal = cudaSuccess;
    CImg< unsigned char > inputImage;
    unsigned char *devInputImage;
    CImg< unsigned char > darkGrayImage;
    unsigned char *devDarkGrayImage;
    unsigned char *outputImage;
    int pixel_numbers;

    NSTimer globalTimer("GlobalTimer", false, false);
    NSTimer kernelTimer("KernelTimer", false, false);
    NSTimer memoryTimer("MemoryTimer", false, false);


    if ( argc != 2 ) {
        cerr << "Usage: " << argv[0] << " <filename>" << endl;
        return 1;
    }

    // Load the input image
    inputImage = CImg< unsigned char >(argv[1]);
    if ( inputImage.spectrum() != 3 ) {
        cerr << "The input must be a color image." << endl;
        return 1;
    }

    pixel_numbers = width() * height();

    // Start of the computation
    globalTimer.start();

    // Convert the input image to grayscale and make it darker
    outputImage = new unsigned char[pixel_numbers];

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


    int grid_width = width() % B_WIDTH == 0 ? width() / B_WIDTH : width() / B_WIDTH + 1;
    int grid_height = width() % B_HEIGHT == 0 ? height() / B_HEIGHT : height() / B_HEIGHT + 1;

    // Execute the kernel
    dim3 gridSize(static_cast< unsigned int >(ceil(height() / static_cast< float >(B_HEIGHT))), static_cast< unsigned int >(ceil(width() / static_cast< float >(B_WIDTH))));
    dim3 blockSize(B_WIDTH * B_HEIGHT);

    kernelTimer.start();
    darkGray <<< gridSize, blockSize >>>(grid_height, grid_width, devInputImage, devDarkGrayImage);
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
    darkGrayImage._data = outputImage;
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
    cout << "GFLOP/s: \t" << (pixel_numbers / kernelTimer.getElapsed()) / 1000000000.0 << endl;
    cout << "GB/s: \t\t" << ((12 * pixel_numbers) / kernelTimer.getElapsed()) / 1000000000.0 << endl;
    cout << endl;

    // Save output
    darkGrayImage.save(("./" + string(argv[1]) + ".dark.seq.bmp").c_str());

    cudaFree(devInputImage);
    cudaFree(devDarkGrayImage);
    free(outputImage);

    return 0;
}
*/