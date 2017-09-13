#ifndef CUDAQUICKSES_H
#define CUDAQUICKSES_H

#include <cuda.h>
#include <cuda_runtime.h>

class CUDAQuickSES {
public:
	CUDAQuickSES();  ///< constructor
    ~CUDAQuickSES(); ///< destructor

private:
	int cudadevice;            ///< CUDA device index
	int cudacomputemajor;      ///< CUDA compute capability major version

};

#endif // CUDAQUICKSES_H
