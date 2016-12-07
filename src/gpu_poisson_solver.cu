#include <stdio.h>
#include <math.h>
#ifdef _WIN64
#include <cufft.h>

#define PI_FLOAT 3.14159265358979323846264338327f

#define OUTPUT
//#define OUTPUT_GF
#define OUTPUT_CHARGE
//#define OUTPUT_CHARGE_FFT
//#define OUTPUT_CHARGE_FFT_GF
//#define OUTPUT_POTENTIAL

void displayDeviceProperties(cudaDeviceProp* pDeviceProp);

__global__ void createGreensFunc(cufftReal* greensfunc, unsigned int Nx, unsigned int Ny, unsigned int Nz, float h) {
    unsigned int tmp;
    unsigned int coord[3];
    
    for(int i = blockDim.x*blockIdx.x+threadIdx.x; i < Nz * Ny * (Nx/2+1); i += gridDim.x*blockDim.x) {
        coord[0] = i % (Nx/2+1);
        tmp = i / (Nx/2+1);
        coord[1] = tmp % Ny;
        coord[2] = tmp / Ny;
        
        /* Setting 0th fourier mode to 0.0 enforces charge neutrality (effectively
           adds homogeneous counter charge). This is necessary, since the equation
           otherwise has no solution in periodic boundaries (an infinite amount of
           charge would create an infinite potential). */
        if(i == 0)
            greensfunc[i] = 0.0f;
        else
            greensfunc[i] = -0.5f * h * h / (cos(2.0f*PI_FLOAT*coord[0]/(cufftReal)Nx) +
                    cos(2.0f*PI_FLOAT*coord[1]/(cufftReal)Ny) + cos(2.0f*PI_FLOAT*coord[2]/(cufftReal)Nz) - 3.0f);
    }
}

__global__ void multiplyGreensFunc(cufftComplex* data, cufftReal* greensfunc, unsigned int N) {
    for(int i = blockDim.x*blockIdx.x+threadIdx.x; i < N; i += gridDim.x*blockDim.x) {
        data[i].x *= greensfunc[i];
        data[i].y *= greensfunc[i];
    }
}

void displayDeviceProperties(cudaDeviceProp* pDeviceProp)
{
    if(!pDeviceProp)
        return;

    printf("\nDevice Name \t – %s ", pDeviceProp->name);
    printf("\n**************************************");
    printf("\nTotal Global Memory\t\t -%d KB", (int) pDeviceProp->totalGlobalMem/1024);
    printf("\nShared memory available per block \t – %d KB", (int) pDeviceProp->sharedMemPerBlock/1024);
    printf("\nNumber of registers per thread block \t – %d", pDeviceProp->regsPerBlock);
    printf("\nWarp size in threads \t – %d", pDeviceProp->warpSize);
    printf("\nMemory Pitch \t – %d bytes", (int) pDeviceProp->memPitch);
    printf("\nMaximum threads per block \t – %d", pDeviceProp->maxThreadsPerBlock);
    printf("\nMaximum Thread Dimension (block) \t – %d %d %d", pDeviceProp->maxThreadsDim[0], pDeviceProp->maxThreadsDim[1], pDeviceProp->maxThreadsDim[2]);
    printf("\nMaximum Thread Dimension (grid) \t – %d %d %d", pDeviceProp->maxGridSize[0], pDeviceProp->maxGridSize[1], pDeviceProp->maxGridSize[2]);
    printf("\nTotal constant memory \t – %d bytes", (int) pDeviceProp->totalConstMem);
    printf("\nCUDA ver \t – %d.%d", pDeviceProp->major, pDeviceProp->minor);
    printf("\nClock rate \t – %d KHz", pDeviceProp->clockRate);
    printf("\nTexture Alignment \t – %d bytes", (int) pDeviceProp->textureAlignment);
    printf("\nDevice Overlap \t – %s", pDeviceProp-> deviceOverlap?"Allowed":"Not Allowed");
    printf("\nNumber of Multi processors \t – %d\n", pDeviceProp->multiProcessorCount);
}
#endif