//
// cuda_error_check.h
//
// Copyright (C) 2013, 2020 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Apr 04, 2013
//     Author: scharnkn
//

#ifndef MMPROTEINCUDAPLUGIN_CUDA_ERROR_CHECK_H_INCLUDED
#define MMPROTEINCUDAPLUGIN_CUDA_ERROR_CHECK_H_INCLUDED

#include <cuda_runtime.h>

#define CUDA_ERROR_CHECK // Toggle CUDA error checking

#define CudaSafeCall(err) cudaSafeCall(err, __FILE__, __LINE__)
#define CheckForCudaError() checkForCudaError(__FILE__, __LINE__)
#define CheckForCudaErrorSync() checkForCudaErrorSync(__FILE__, __LINE__)

/**
 * Utility function that retrieves the last CUDA error and prints an
 * error message if it is not cudaSuccess.
 *
 * @param file The file in which the failure took place
 * @param line The line at which the failure took place
 * @return 'True' if the last error is cudaSuccess, 'false' otherwise
 */
bool checkForCudaError(const char* file, const int line);

/**
 * Utility function that retrieves the last CUDA error and prints an
 * error message if it is not cudaSuccess.
 *
 * @param file The file in which the failure took place
 * @param line The line at which the failure took place
 * @return 'True' if the last error is cudaSuccess, 'false' otherwise
 */
bool checkForCudaErrorSync(const char* file, const int line);

/**
 * Exits and prints an error message if a called method does return a CUDA
 * error.
 *
 * @param err  The cuda related error
 * @param file The file in which the failure took place
 * @param line The line at which the failure took place
 * @return 'True' if the last error is cudaSuccess, 'false' otherwise
 */
bool cudaSafeCall(cudaError err, const char* file, const int line);

#endif // MMPROTEINCUDAPLUGIN_CUDA_ERROR_CHECK_H_INCLUDED
