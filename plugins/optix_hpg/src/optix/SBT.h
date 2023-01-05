#pragma once

#include "optix.h"

#include "optix/Utils.h"

namespace megamol::optix_hpg {

template<typename T>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SBTRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SBTNullRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

class MMOptixSBT {
public:
    MMOptixSBT() {
        std::memset(&sbt_, 0, sizeof(sbt_));
    }

    ~MMOptixSBT() {
        CUDA_CHECK_ERROR(cuMemFree(sbt_.raygenRecord));
        CUDA_CHECK_ERROR(cuMemFree(sbt_.exceptionRecord));
        CUDA_CHECK_ERROR(cuMemFree(sbt_.missRecordBase));
        CUDA_CHECK_ERROR(cuMemFree(sbt_.hitgroupRecordBase));
        CUDA_CHECK_ERROR(cuMemFree(sbt_.callablesRecordBase));
    }

    void SetSBT(void const* raygen, std::size_t raygen_size, void const* exception, std::size_t exception_size,
        void const* miss, unsigned int miss_stride, unsigned int miss_count, void const* hitgroup,
        unsigned int hitgroup_stride, unsigned int hitgroup_count, void const* callables, unsigned int callables_stride,
        unsigned int callables_count, CUstream stream) {
        if (raygen != nullptr) {
            if (raygen_size != raygen_record_size_) {
                CUDA_CHECK_ERROR(cuMemFree(sbt_.raygenRecord));
                CUDA_CHECK_ERROR(cuMemAlloc(&sbt_.raygenRecord, raygen_size));
                raygen_record_size_ = raygen_size;
            }
            CUDA_CHECK_ERROR(cuMemcpyHtoDAsync(sbt_.raygenRecord, raygen, raygen_size, stream));
        }

        if (exception != nullptr) {
            if (exception_size != exception_record_size_) {
                CUDA_CHECK_ERROR(cuMemFree(sbt_.exceptionRecord));
                CUDA_CHECK_ERROR(cuMemAlloc(&sbt_.exceptionRecord, exception_size));
                exception_record_size_ = exception_size;
            }
            CUDA_CHECK_ERROR(cuMemcpyHtoDAsync(sbt_.exceptionRecord, exception, exception_size, stream));
        }

        if (miss != nullptr) {
            auto miss_size = miss_count * miss_stride;
            if (miss_size != miss_record_size_) {
                CUDA_CHECK_ERROR(cuMemFree(sbt_.missRecordBase));
                CUDA_CHECK_ERROR(cuMemAlloc(&sbt_.missRecordBase, miss_size));
                miss_record_size_ = miss_size;
            }
            CUDA_CHECK_ERROR(cuMemcpyHtoDAsync(sbt_.missRecordBase, miss, miss_size, stream));
            sbt_.missRecordCount = miss_count;
            sbt_.missRecordStrideInBytes = miss_stride;
        }

        if (hitgroup != nullptr) {
            auto hitgroup_size = hitgroup_count * hitgroup_stride;
            if (hitgroup_size != hitgroup_record_size_) {
                CUDA_CHECK_ERROR(cuMemFree(sbt_.hitgroupRecordBase));
                CUDA_CHECK_ERROR(cuMemAlloc(&sbt_.hitgroupRecordBase, hitgroup_size));
                hitgroup_record_size_ = hitgroup_size;
            }
            CUDA_CHECK_ERROR(cuMemcpyHtoDAsync(sbt_.hitgroupRecordBase, hitgroup, hitgroup_size, stream));
            sbt_.hitgroupRecordCount = hitgroup_count;
            sbt_.hitgroupRecordStrideInBytes = hitgroup_stride;
        }

        if (callables != nullptr) {
            auto callables_size = callables_count * callables_stride;
            if (callables_size != callables_record_size_) {
                CUDA_CHECK_ERROR(cuMemFree(sbt_.callablesRecordBase));
                CUDA_CHECK_ERROR(cuMemAlloc(&sbt_.callablesRecordBase, callables_size));
                callables_record_size_ = callables_size;
            }
            CUDA_CHECK_ERROR(cuMemcpyHtoDAsync(sbt_.callablesRecordBase, callables, callables_size, stream));
            sbt_.callablesRecordCount = callables_count;
            sbt_.callablesRecordStrideInBytes = callables_stride;
        }
    }

    operator OptixShaderBindingTable const*() const {
        return &sbt_;
    }

private:
    OptixShaderBindingTable sbt_;

    std::size_t raygen_record_size_;

    std::size_t exception_record_size_;

    std::size_t miss_record_size_;

    std::size_t hitgroup_record_size_;

    std::size_t callables_record_size_;
};

} // namespace megamol::optix_hpg
