#include "CUDAQuickSES.h"


CUDAQuickSES::CUDAQuickSES() {
	// Query GPU device attributes so we can launch the best kernel type
	cudaDeviceProp deviceProp;
	memset(&deviceProp, 0, sizeof(cudaDeviceProp));

	if (cudaGetDevice(&cudadevice) != cudaSuccess) {
		// XXX do something more useful here...
	}

	if (cudaGetDeviceProperties(&deviceProp, cudadevice) != cudaSuccess) {
		// XXX do something more useful here...
	}

	cudacomputemajor = deviceProp.major;
}


CUDAQuickSES::~CUDAQuickSES() {

}
