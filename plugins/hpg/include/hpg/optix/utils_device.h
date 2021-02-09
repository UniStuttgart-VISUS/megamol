#pragma once

#include "cuda.h"
#include "optix.h"

#include "utils_host.h"

#include "glm/glm.hpp"

#ifndef __CUDACC__
#error "CUDA device-only include"
#endif

#define MM_OPTIX_RAYGEN_KERNEL(name) extern "C" __global__ void __raygen__##name

#define MM_OPTIX_INTERSECTION_KERNEL(name) extern "C" __global__ void __intersection__##name

#define MM_OPTIX_ANYHIT_KERNEL(name) extern "C" __global__ void __anyhit__##name

#define MM_OPTIX_CLOSESTHIT_KERNEL(name) extern "C" __global__ void __closesthit__##name

#define MM_OPTIX_MISS_KERNEL(name) extern "C" __global__ void __miss__##name

#define MM_OPTIX_DIRECT_CALLABLE_KERNEL(name) extern "C" __global__ void __direct_callable__##name

#define MM_OPTIX_CONTINUATION_CALLABLE_KERNEL(name) extern "C" __global__ void __continuation_callable__##name

#define MM_OPTIX_EXCEPTION_KERNEL(name) extern "C" __global__ void __exception__##name

#define MM_OPTIX_BOUNDS_KERNEL(name)                                                                         \
    inline __device__ void __bounds__##name(void const* geo_data, box3f& bounds, unsigned int const primID); \
                                                                                                             \
    extern "C" __global__ void __boundsKernel__##name(                                                       \
        void const* geo_data, box3f* boundsArray, unsigned int const numPrims) {                             \
        unsigned int blockIndex = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;  \
        unsigned int primID = threadIdx.x + blockDim.x * threadIdx.y + blockDim.x * blockDim.y * blockIndex; \
        if (primID < numPrims) {                                                                             \
            __bounds__##name(geo_data, boundsArray[primID], primID);                                         \
        }                                                                                                    \
    }                                                                                                        \
                                                                                                             \
    inline __device__ void __bounds__##name

namespace megamol {
namespace hpg {
    namespace optix {
        namespace device {
            typedef struct Ray {
                __device__ Ray(float3 const& org, float3 const& dir, float tmin, float tmax)
                        : origin(org.x, org.y, org.z), direction(dir.x, dir.y, dir.z), tmin(tmin), tmax(tmax) {}
                __device__ Ray(glm::vec3 const& org, glm::vec3 const& dir, float tmin, float tmax)
                        : origin(org), direction(dir), tmin(tmin), tmax(tmax) {}

                glm::vec3 origin;
                glm::vec3 direction;
                float tmin;
                float tmax;
            } Ray;


            template<typename T>
            inline __device__ T const& getProgramData() {
                return *(T const*) optixGetSbtDataPointer();
            }


            inline __device__ void* unpackPointer(unsigned int i0, unsigned int i1) {
                unsigned long long const uptr = static_cast<unsigned long long>(i0) << 32 | i1;
                return reinterpret_cast<void*>(uptr);
            }


            inline __device__ void packPointer(void* ptr, unsigned int& i0, unsigned int& i1) {
                unsigned long long const uptr = reinterpret_cast<unsigned long long>(ptr);
                i0 = uptr >> 32;
                i1 = uptr & 0x00000000ffffffff;
            }


            inline __device__ void* getPerRayDataPointer() {
                unsigned int const u0 = optixGetPayload_0();
                unsigned int const u1 = optixGetPayload_1();
                return unpackPointer(u0, u1);
            }


            template<typename T>
            inline __device__ T& getPerRayData() {
                return *(T*) getPerRayDataPointer();
            }
        } // namespace device
    }     // namespace optix
} // namespace hpg
} // namespace megamol
