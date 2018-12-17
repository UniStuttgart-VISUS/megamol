//
// PotentialCalculator.cu
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Apr 23, 2013
//     Author: scharnkn
//

#include "gpu_poisson_solver.cu" // Note: by Georg Rempfer (georg@icp.uni-stuttgart.de)
#include "cuenergy.cu"
#include "cuda_error_check.h"
#ifdef _WIN64
#include "cufft.h"

#define CUERR { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
  printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
  return err; }}

typedef unsigned int uint;


extern "C"
cudaError_t SolvePoissonEq(float gridSpacing, uint3 gridSize, float *charges,
        float *potential_D, float *potential) {

    unsigned int Nx = gridSize.x;
    unsigned int Ny = gridSize.y;
    unsigned int Nz = gridSize.z;
    float h = gridSpacing; // Gridspacing

    printf("Calculating electrostatic potential on a %d*%d*%d grid with spacing %f\n", Nx, Ny, Nz, h);

    /* timing */
    float time = 0.0, time_tmp;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cufftHandle plan_fft;
    cufftHandle plan_ifft;
    cufftComplex* data_dev;
    cufftComplex* data_host;
    cufftReal* data_real_host;
    cufftReal* greensfunc_dev;
    cufftReal* greensfunc_host;

    cudaMalloc((void**) &data_dev, sizeof(cufftComplex)*Nz*Ny*(Nx/2+1));

    if(cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "ERROR: Failed to allocate\n");
        return cudaGetLastError();
    }

    cudaMalloc((void**) &greensfunc_dev, sizeof(cufftReal)*Nz*Ny*(Nx/2+1));

    if(cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "ERROR: Failed to allocate\n");
        return cudaGetLastError();
    }

    cudaMallocHost((void**) &data_host, sizeof(cufftComplex)*Nz*Ny*(Nx/2+1));

    if(cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "ERROR: Failed to allocate\n");
        return cudaGetLastError();
    }

    data_real_host = (cufftReal*) data_host;

    cudaMallocHost((void**) &greensfunc_host, sizeof(cufftReal)*Nz*Ny*(Nx/2+1));

    if(cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "ERROR: Failed to allocate\n");
        return cudaGetLastError();
    }

    /* greens function */
    printf("Creating greens function in device memory\n");

    createGreensFunc <<< 14, 32*32 >>> (greensfunc_dev, Nx, Ny, Nz, h);

    /* charge density */
    printf("Writing charge density in host memory\n");

	for (int z = 0; z < (int)Nz; ++z) {
		for (int y = 0; y < (int)Ny; ++y) {
			for (int x = 0; x < (int)Nx; ++x) {
//                if((x-Nx/2)*(x-Nx/2) + (y-Ny/2)*(y-Ny/2) + (z-Nz/2)*(z-Nz/2) <= 5*5/(h*h)) //homogeneously chargeed sphere of radius 5
//                    data_real_host[Ny*Nx*z+Nx*y+x] = h*h*h;
//                else
//                    data_real_host[Ny*Nx*z+Nx*y+x] = 0.0;
                data_real_host[Ny*Nx*z+Nx*y+x] = charges[Ny*Nx*z+Nx*y+x];
            }
        }
    }

    printf("Copying charge density to device\n");

    cudaMemcpy(data_dev, data_host, sizeof(cufftComplex)*Nz*Ny*(Nx/2+1), cudaMemcpyHostToDevice);

    if(cudaGetLastError() != cudaSuccess) {
        fprintf(stderr, "ERROR: Failed to copy\n");
        return cudaGetLastError();
    }

    /* create 3D FFT plans */
    printf("Setting up FFT and iFFT plans\n");

    /* Notice how the directions x and z are exchanged. This is because for R2C
       transforms, cuda only stores half the results in the 3rd direction. At
       the same time cuda expects the fastest running index to be the one with
       only half the values stored, which effectively forces one to make the 3rd
       index (usually z) the fastest running one. I find this rather uncommon
       and want x to be the festest running index and z the slowest running, so
       I chose to exchange the two in the fourier transforms. */
    if(cufftPlan3d(&plan_fft, Nz, Ny, Nx, CUFFT_R2C) != CUFFT_SUCCESS) {
        fprintf(stderr, "ERROR: Unable to create fft plan\n");
        return cudaGetLastError();
    }

    /*if(cufftSetCompatibilityMode(plan_fft, CUFFT_COMPATIBILITY_NATIVE) != CUFFT_SUCCESS) {
        fprintf(stderr, "ERROR: Unable to set fft compatibility mode to native\n");
        return cudaGetLastError();
    }*/

    if(cufftPlan3d(&plan_ifft, Nz, Ny, Nx, CUFFT_C2R) != CUFFT_SUCCESS) {
        fprintf(stderr, "ERROR: Unable to create ifft plan\n");
        return cudaGetLastError();
    }

    /*if(cufftSetCompatibilityMode(plan_ifft, CUFFT_COMPATIBILITY_NATIVE) != CUFFT_SUCCESS) {
        fprintf(stderr, "ERROR: Unable to set ifft compatibility mode to native\n");
        return cudaGetLastError();
    }*/

    /* FFT in place */
    printf("Executing FFT in place\n");

    cudaEventRecord(start, 0);

    if(cufftExecR2C(plan_fft, (cufftReal*) data_dev, data_dev) != CUFFT_SUCCESS) {
        fprintf(stderr, "ERROR: Unable to execute FFT plan\n");
        return cudaGetLastError();
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_tmp, start, stop);
    printf("Execution time: %f ms\n", time_tmp);
    time += time_tmp;

    if(cudaThreadSynchronize() != cudaSuccess) {
        fprintf(stderr, "ERROR: Failed to synchronize\n");
        return cudaGetLastError();
    }

    /* multiplying with greens function */
    printf("Executing multiplication with greens function in place\n");

    cudaEventRecord(start, 0);

    //18-fold occupation seems to be optimal for the GT520 and 32-fold for the C2050
    multiplyGreensFunc <<<14,32*32>>> (data_dev,
            greensfunc_dev, Nz*Ny*(Nx/2+1));

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_tmp, start, stop);
    printf("Execution time: %f ms\n", time_tmp);
    time += time_tmp;

    if(cudaThreadSynchronize() != cudaSuccess) {
        fprintf(stderr, "ERROR: Failed to synchronize\n");
        return cudaGetLastError();
    }


    /* inverse FFT in place */
    printf("Executing iFFT in place\n");

    cudaEventRecord(start, 0);

    if(cufftExecC2R(plan_ifft, data_dev, (cufftReal*) data_dev) != CUFFT_SUCCESS) {
        fprintf(stderr, "ERROR: Unable to execute iFFT plan\n");
        return cudaGetLastError();
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_tmp, start, stop);
    printf("Execution time: %f ms\n", time_tmp);
    time += time_tmp;

    if(cudaThreadSynchronize() != cudaSuccess) {
        fprintf(stderr, "ERROR: Failed to synchronize\n");
        return cudaGetLastError();
    }

    /* retrieving result from device */
    cudaMemcpy(data_host, data_dev, sizeof(cufftComplex)*Nz*Ny*(Nx/2+1), cudaMemcpyDeviceToHost);

    /* Output result to the according host array */

	for (int z = 0; z < (int)Nz; z++) {
		for (int y = 0; y < (int)Ny; y++)
			for (int x = 0; x < (int)Nx; x++)
               potential[Ny*Nx*z+Nx*y+x]  = data_real_host[Ny*Nx*z+Nx*y+x]/(Nx*Ny*Nz);
    }

    /* cleanup */
    printf("Cleanup\n");

    cufftDestroy(plan_fft);
    cufftDestroy(plan_ifft);

    cudaFree(data_dev);
    cudaFree(greensfunc_dev);
    cudaFree(data_real_host); // TODO This causes invalif device pointer error, why?
    cudaFree(greensfunc_host);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Net device execution time: %f ms\n", time);

    return cudaGetLastError();
}

extern "C"
cudaError_t DirectCoulombSummation(float *atomData, uint atomCount,
        float *potential, uint3 gridSize, float gridspacing) {

    // TODO Why is this so incredibely slow??

    float *doutput = NULL;
    dim3 volsize, Gsz, Bsz;
    //float copytotal, runtotal, mastertotal, hostcopytotal;
    const char *statestr = "|/-\\.";
    int state=0;

    printf("CUDA accelerated coulombic potential microbenchmark V4.0\n");
    printf("John E. Stone <johns@ks.uiuc.edu>\n");
    printf("and Chris Rodrigues\n");
    printf("http://www.ks.uiuc.edu/Research/gpu/\n");
    printf("--------------------------------------------------------\n");
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("Detected %d CUDA accelerators:\n", deviceCount);
    int dev;
    for (dev=0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        printf("  [%d]: '%s'  Clock: %.1f GHz  Mem: %dMB  Rev: %d.%d\n",
                dev, deviceProp.name,
                deviceProp.clockRate / 1000000.0f, deviceProp.totalGlobalMem / (1024*1024),
                deviceProp.major, deviceProp.minor);
    }

    int cudadev = 0; // Use first cuda device
    //      if (argc == 2) {
    //        sscanf(argv[1], "%d", &cudadev);
    //        if (cudadev < 0 || cudadev >= deviceCount) {
    //          cudadev = 0;
    //        }
    //      }
    printf("  Single-threaded single-GPU test run.\n");
    printf("  Opening CUDA device %d...\n", cudadev);
    cudaSetDevice(cudadev);
    CUERR // check and clear any existing errors TODO

    // number of atoms to simulate
    int atomcount = 100000;

    // setup energy grid size
    // XXX this is a large test case to clearly illustrate that even while
    //     the CUDA kernel is running entirely on the GPU, the CUDA runtime
    //     library is soaking up the entire host CPU for some reason.
    volsize = gridSize;

    // setup CUDA grid and block sizes
    // XXX we have to make a trade-off between the number of threads per
    //     block and the resulting padding size we'll end up with since
    //     each thread will do several consecutive grid cells in this version,
    //     we're using up some of our available parallelism to reduce overhead.
    Bsz.x = BLOCKSIZEX;
    Bsz.y = BLOCKSIZEY;
    Bsz.z = 1;
    Gsz.x = max(1, volsize.x / (Bsz.x * UNROLLX));
    Gsz.y = max(1, volsize.y / (Bsz.y * UNROLLY));
    Gsz.z = volsize.z;

//    printf("Run the kernel, gridSize %u %u %u, blockSize %u %u %u\n", (Bsz.x * UNROLLX),
//            Gsz.y, Gsz.z, Bsz.x, Bsz.y, Bsz.z);

    // allocate and initialize the GPU output array
    int volmemsz = sizeof(float) * volsize.x * volsize.y * volsize.z;
    printf("Allocating %.2fMB of memory for output buffer...\n", volmemsz / (1024.0 * 1024.0));

    cudaMalloc((void**)&doutput, volmemsz);
    CUERR // check and clear any existing errors TODO
    cudaMemset(doutput, 0, volmemsz);
    CUERR // check and clear any existing errors TODO

    for (uint z = 0; z < volsize.z; ++z) {
        printf("starting run for slab %u...\n", z);

        int iterations=0;
        int atomstart;

        for (atomstart = 0; atomstart < atomcount; atomstart += MAXATOMS) {

            iterations++;
            int runatoms;
            int atomsremaining = atomcount - atomstart;
            if (atomsremaining > MAXATOMS)
                runatoms = MAXATOMS;
            else
                runatoms = atomsremaining;

            printf("%c\r", statestr[state]);
            fflush(stdout);
            state = (state+1) & 3;

            // copy the atoms to the GPU

            if (copyatomstoconstbuf(atomData + 4*atomstart, runatoms, z*gridspacing))
                return cudaGetLastError();

            CUERR // check and clear any existing errors

            // RUN the kernel...
            cenergy <<< Gsz, Bsz >>> (runatoms, gridspacing, doutput);

            // TODO
            CUERR // check and clear any existing errors
        }
        cudaThreadSynchronize();
    }

    // Copy the GPU output data back to the host and use/store it..
    cudaMemcpy(potential, doutput, volmemsz,  cudaMemcpyDeviceToHost);



    //TODO
    CUERR // check and clear any existing errors

#if 1
    int x, y;
    for (y=0; y<16; y++) {
        for (x=0; x<16; x++) {
            int addr = y * volsize.x + x;
            printf("out[%d]: %f\n", addr, potential[addr]);
        }
    }
#endif

//    printf("Final calculation required %d iterations of %d atoms\n", iterations, MAXATOMS);
//    printf("Copy time: %f seconds, %f per iteration\n", copytotal, copytotal / (float) iterations);
//    printf("Kernel time: %f seconds, %f per iteration\n", runtotal, runtotal / (float) iterations);
//    printf("Total time: %f seconds\n", mastertotal);
//    printf("Kernel invocation rate: %f iterations per second\n", iterations / mastertotal);
//    printf("GPU to host copy bandwidth: %gMB/sec, %f seconds total\n",
//            (volmemsz / (1024.0 * 1024.0)) / hostcopytotal, hostcopytotal);

//    double atomevalssec = ((double) volsize.x * volsize.y * volsize.z * atomcount) / (mastertotal * 1000000000.0);
//    printf("Efficiency metric, %g billion atom evals per second\n", atomevalssec);
//
//    /* 59/8 FLOPS per atom eval */
//    printf("FP performance: %g GFLOPS\n", atomevalssec * (59.0/8.0));

    cudaFree(doutput);

    return cudaGetLastError();
}



#endif // _WIN64
