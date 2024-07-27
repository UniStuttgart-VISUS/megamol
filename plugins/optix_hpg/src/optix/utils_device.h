#pragma once

#include <cuda.h>
#include <glm/glm.hpp>
#include <optix.h>

#include "perraydata.h"
#include "random.h"
#include "utils_host.h"

#ifndef __CUDACC__
#error "CUDA device-only include"
#endif

#define MMO_INV_PI 0.318309886183f
#define MMO_PI 3.141592653589f
#define MMO_PI_2 9.869604401084f
#define MMO_PI_4 97.409091033904f

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
namespace optix_hpg {
namespace device {
typedef struct Ray {
    __device__ Ray(float3 const& org, float3 const& dir, float tmin, float tmax)
            : origin(org.x, org.y, org.z)
            , direction(dir.x, dir.y, dir.z)
            , tmin(tmin)
            , tmax(tmax) {}
    __device__ Ray(glm::vec3 const& org, glm::vec3 const& dir, float tmin, float tmax)
            : origin(org)
            , direction(dir)
            , tmin(tmin)
            , tmax(tmax) {}

    glm::vec3 origin;
    glm::vec3 direction;
    float tmin;
    float tmax;
} Ray;

// OptiX SDK

//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

//
// Modified 2021 MegaMol Dev Team
//

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

inline __device__ glm::vec3 lerp(const glm::vec3& a, const glm::vec3& b, const float t) {
    return a + t * (b - a);
}

inline __device__ glm::vec4 lerp(const glm::vec4& a, const glm::vec4& b, const float t) {
    return a + t * (b - a);
}

struct Onb {
    inline __device__ Onb(const glm::vec3& normal) {
        m_normal = normal;

        if (fabs(m_normal.x) > fabs(m_normal.z)) {
            m_binormal.x = -m_normal.y;
            m_binormal.y = m_normal.x;
            m_binormal.z = 0;
        } else {
            m_binormal.x = 0;
            m_binormal.y = -m_normal.z;
            m_binormal.z = m_normal.y;
        }

        m_binormal = normalize(m_binormal);
        m_tangent = cross(m_binormal, m_normal);
    }

    inline __device__ void inverse_transform(glm::vec3& p) const {
        p = p.x * m_tangent + p.y * m_binormal + p.z * m_normal;
    }

    glm::vec3 m_tangent;
    glm::vec3 m_binormal;
    glm::vec3 m_normal;
};

// End OptiX SDK


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


// megamol


inline __device__ void set_depth(PerRayData& prd, float depth) {
#if 1
    if (prd.countDepth) {
        prd.ray_depth = depth;
        prd.countDepth = false;
    }
#else
    prd.ray_depth = depth;
#endif
}

// OptiX SDK
// Path tracer example

//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

//
// Modified 2021 MegaMol Dev Team
//
#if 0
inline __device__ void lighting(PerRayData& prd, glm::vec3 const& geo_col, glm::vec3 const& P, glm::vec3 const& ffN) {
    if (prd.countEmitted)
        prd.emitted = geo_col * 0.2f;
    // prd.emitted = N;
    /*else
        prd.emitted = glm::vec3(0.0f);*/


    unsigned int seed = prd.seed;

    {
        const float z1 = rnd(seed);
        const float z2 = rnd(seed);

        glm::vec3 w_in;
        w_in = CosineSampleHemisphere(glm::vec2(z1, z2));
        Onb onb(ffN);
        onb.inverse_transform(w_in);
        prd.direction = w_in;
        prd.origin = P + 0.0001f * w_in;

        prd.beta *= geo_col;
        prd.countEmitted = false;
    }

    /*const float z1 = rnd(seed);
    const float z2 = rnd(seed);*/


    // Calculate properties of light sample (for area based pdf)
    const float Ldist = length(prd.lpos - P);
    const glm::vec3 L = normalize(prd.lpos - P);
    const float nDl = dot(ffN, L);
    const float LnDl = -dot(prd.ldir, L);

    float weight = 0.0f;
    if (nDl > 0.0f /* && LnDl > 0.0f*/) {
        // const bool occluded = traceOcclusion(params.handle, P, L,
        //    0.01f,        // tmin
        //    Ldist - 0.01f // tmax
        //);
        float3 Pn = make_float3(P.x, P.y, P.z);
        float3 Ln = make_float3(L.x, L.y, L.z);
        unsigned int occluded = 0;
        optixTrace(prd.world, Pn, Ln, 0.01f, Ldist - 0.01f, 0.0f, (OptixVisibilityMask) -1,
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT, 1, 2, 1, occluded);

        if (!occluded) {
            weight = nDl /* LnDl*/; // / (MMO_PI * Ldist * Ldist);
        }
    }

    prd.radiance += glm::vec3(prd.intensity) * weight;
}
#endif


} // namespace device
} // namespace optix_hpg
} // namespace megamol
