#pragma once

#include "cuda.h"
#include "optix.h"

#include "utils_host.h"

#include "glm/glm.hpp"

#ifndef __CUDACC__
#error "CUDA device-only include"
#endif

#define MMO_INV_PI 0.318309886183f
#define MMO_PI     3.141592653589f
#define MMO_PI_2   9.869604401084f
#define MMO_PI_4  97.409091033904f

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


            // pbrt
            inline __device__ glm::vec2 ConcentricSampleDisk(glm::vec2 const& u) {
                auto const uOffset = 2.f * u - glm::vec2(1.f);
                if (uOffset.x == 0.f && uOffset.y == 0.f) {
                    return glm::vec2(0.f);
                }
                float theta, r;
                if (fabsf(uOffset.x) > fabsf(uOffset.y)) {
                    r = uOffset.x;
                    theta = MMO_PI_4 * (uOffset.y / uOffset.x);
                } else {
                    r = uOffset.y;
                    theta = MMO_PI_2 - MMO_PI_4 * (uOffset.x / uOffset.y);
                }
                return r * glm::vec2(cosf(theta), sinf(theta));
            }

            inline __device__ glm::vec3 CosineSampleHemisphere(glm::vec2 const& u) {
                auto const d = ConcentricSampleDisk(u);
                float const z = sqrtf(fmaxf(0.f, 1.f - d.x * d.x - d.y * d.y));
                return glm::vec3(d.x, d.y, z);
            }

            inline __device__ float CosineHemispherePdf(float cosTheta) {
                return cosTheta * MMO_INV_PI;
            }

            // optix sdk
            /** Faceforward
             * Returns N if dot(i, nref) > 0; else -N;
             * Typical usage is N = faceforward(N, -ray.dir, N);
             * Note that this is opposite of what faceforward does in Cg and GLSL */
            inline __device__ glm::vec3 faceforward(const glm::vec3& n, const glm::vec3& i, const glm::vec3& nref) {
                return n * copysignf(1.0f, dot(i, nref));
            }

            inline __device__ glm::vec3 reflect(const glm::vec3& i, const glm::vec3& n) {
                return i - 2.0f * n * dot(n, i);
            }

            inline __device__ float luminance(const glm::vec3& rgb) {
                const glm::vec3 ntsc_luminance = glm::vec3(0.30f, 0.59f, 0.11f);
                return dot(rgb, ntsc_luminance);
            }
        } // namespace device
    }     // namespace optix
} // namespace hpg
} // namespace megamol
