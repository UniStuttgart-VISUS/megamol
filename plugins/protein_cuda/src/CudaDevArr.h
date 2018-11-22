//
// CudaDevArr.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Apr 14, 2013
//     Author: scharnkn
//

#ifndef MMPROTEINCUDAPLUGIN_CUDADEVARR_H_INCLUDED
#define MMPROTEINCUDAPLUGIN_CUDADEVARR_H_INCLUDED

#include "cuda_runtime.h"
#include "cuda_error_check.h"

namespace megamol {
namespace protein_cuda {

template<class T>
class CudaDevArr {

public:

    /** Ctor */
    CudaDevArr() : size(0), count(0), pt_D(NULL) {}

    /** Dtor */
    ~CudaDevArr() {
    }

    /**
     * Copies the content of the array to a host array. Does not allocate host
     * memory.
     */
    inline cudaError_t CopyToHost(T *hostArr) {
        CudaSafeCall(cudaMemcpy(hostArr, this->pt_D, sizeof(T)*this->count,
                cudaMemcpyDeviceToHost));
        return cudaGetLastError();
    }

    /**
     * returns the element at a given index. The data is copied to a host
     * variable.
     */
    T GetAt(size_t idx) {
        ASSERT(idx < this->count);
        T el;
        CudaSafeCall(cudaMemcpy(&el, this->pt_D + idx, sizeof(T), cudaMemcpyDeviceToHost));
        return el;
    }

    /**
     * Answers the number of elements in the array.
     *
     * @return The number of elements
     */
    inline size_t GetCount() const {
        return this->count;
    }

    /**
     * Answers the number of elements in the array.
     *
     * @return The number of elements
     */
    inline size_t GetSize() const {
        return this->size;
    }

    /**
     * Returns the actual (non const) device pointer.
     *
     * @return The pointer to the device memory.
     */
    inline T *Peek() {
        return this->pt_D;
    }

    /**
     * Returns the actual (const) device pointer.
     *
     * @return The pointer to the device memory.
     */
    inline const T *PeekConst() const {
        return this->pt_D;
    }

    /**
     * Deallocates the memory of the array and inits size with zero.
     *
     * @return 'cudaSuccess' on success, the respective error value otherwise
     */
    inline cudaError_t Release() {
        if (this->pt_D != NULL) {
            CudaSafeCall(cudaFree((void*)(this->pt_D)));
        }
        this->size = 0;
        this->count = 0;
        this->pt_D = NULL;
        return cudaGetLastError();
    }

    /**
     * Inits every byte in the array to the value of 'c'.
     *
     * @param c The byte value the array is initialized with
     * @return 'cudaSuccess' on success, the respective error value otherwise
     */
    inline cudaError_t Set(unsigned char c) {
        CudaSafeCall(cudaMemset(this->pt_D, c, sizeof(T)*this->size));
        return cudaGetLastError();
    }

    /**
     * If necessary (re)allocates the device memory to meet the desired new size
     * of the array.
     *
     * @param sizeNew The desired amount of elements.
     * @return 'cudaSuccess' on success, the respective error value otherwise
     */
    inline cudaError_t Validate(size_t sizeNew) {

        if((this->pt_D == NULL)||(sizeNew > this->size)) {
            this->Release();
            CudaSafeCall(cudaMalloc((void**)&this->pt_D, sizeof(T)*sizeNew));
//            printf("Allocated at %p\n", &this->pt_D);
            this->size = sizeNew;
        }
        this->count = sizeNew;
        return cudaGetLastError();
    }

private:

    /// The amount of allocated memory in sizeof(T)
    size_t size;

    /// The number of elements
    size_t count;

    /// The pointer to the device memory
    T *pt_D;
};

} // namespace protein_cuda
} // namespace megamol

#endif // MMPROTEINCUDAPLUGIN_CUDADEVARR_H_INCLUDED
