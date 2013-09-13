//
// PotentialVolumeRenderer_thrust_sort_code.cu
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Jun 5, 2013
//     Author: scharnkn
//

#include "ComparativeSurfacePotentialRenderer.cuh"
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/device_ptr.h>
#include "cuda_helper.h"

extern "C"
cudaError_t ComputePrefixSumExclusiveScan(uint *flagArray_D, uint *offsArray_D,
        uint cnt) {

    thrust::exclusive_scan(
            thrust::device_ptr<uint>(flagArray_D),
            thrust::device_ptr<uint>(flagArray_D + cnt + 1),
            thrust::device_ptr<uint>(offsArray_D));

    return cudaGetLastError();
}

extern "C"
cudaError_t AccumulateFloat(float &res, float *begin_D, uint cnt) {

    res = thrust::reduce(
            thrust::device_ptr<float>(begin_D),
            thrust::device_ptr<float>(begin_D + cnt));

    return cudaGetLastError();
}

extern "C"
cudaError_t ReduceToMax(float &res, float *begin_D, uint cnt, float init) {

    res = thrust::reduce(
            thrust::device_ptr<float>(begin_D),
            thrust::device_ptr<float>(begin_D + cnt), init,
            thrust::maximum<float>());

    return cudaGetLastError();
}

