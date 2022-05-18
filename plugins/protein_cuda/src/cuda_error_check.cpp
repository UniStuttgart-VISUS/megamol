//
// cuda_error_check.cpp
//
// Copyright (C) 2020 by University of Stuttgart (VISUS).
// All rights reserved.
//

#include "cuda_error_check.h"
#include "mmcore/utility/log/Log.h"


bool checkForCudaError(const char* file, const int line) {
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "cudaSafeCall() failed at %s:%i : %s", file, line, cudaGetErrorString(err));
        return false;
    }
#endif
    return true;
}

bool checkForCudaErrorSync(const char* file, const int line) {
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaDeviceSynchronize();
    if (cudaSuccess != err) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "cudaSafeCall() failed with sync at %s:%i : %s", file, line, cudaGetErrorString(err));
        return false;
    }
#endif
    return true;
}

bool cudaSafeCall(cudaError err, const char* file, const int line) {
#ifdef CUDA_ERROR_CHECK
    if (cudaSuccess != err) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "cudaSafeCall() failed at %s:%i : %s", file, line, cudaGetErrorString(err));
        return false;
    }
#endif
    return true;
}
