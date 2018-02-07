/*
* CUDAAdditionalTypedefs.cuh
* Copyright (C) 2009-2018 by MegaMol Team
* Alle Rechte vorbehalten.
*/
#ifndef MMMOLMAPPLG_CUDAADDITIONALTYPEDEFS_CUH_INCLUDED
#define MMMOLMAPPLG_CUDAADDITIONALTYPEDEFS_CUH_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

namespace megamol {
namespace volume_cuda {

    typedef float VolumeType;

    /**
     * A 3x4 matrix
     */
    typedef struct {
        float4 m[3];
    } float3x4;

    /**
     * A ray with origin and direction
     */
    struct Ray {
        float3 o;   // origin
        float3 d;   // direction
    };

    /**
     *	Transform vector by matrix (no translation)
     *
     *	@param M The 3x4 matrix
     *	@param v The vector to be transformed
     *	@return The transformed vector.
     */
    inline __device__ float3 mul(const float3x4 &M, const float3 &v) {
    	float3 r;
    	r.x = dot(v, make_float3(M.m[0]));
    	r.y = dot(v, make_float3(M.m[1]));
    	r.z = dot(v, make_float3(M.m[2]));
    	return r;
    }

    /**
     *	Transform vector by matrix (with translation)
     *
     *	@param M The 3x4 matrix
     *	@param v The vector to be transformed
     *	@return The transformed vector.
     */
    inline __device__ float4 mul(const float3x4 &M, const float4 &v) {
    	float4 r;
    	r.x = dot(v, M.m[0]);
    	r.y = dot(v, M.m[1]);
    	r.z = dot(v, M.m[2]);
    	r.w = 1.0f;
    	return r;
    }

    /**
     *	Converts a rgba color to a colour represented by an unsigned int
     *
     *	@param rgba The rgba colour.
     *	@return The colour as an unsigned int
     */
    inline __device__ uint rgbaFloatToInt(float4 rgba) {
        rgba.x = rgba.x < 0.0f ? 0.0f : rgba.x;
        rgba.x = rgba.x > 1.0f ? 1.0f : rgba.x;
        rgba.y = rgba.y < 0.0f ? 0.0f : rgba.y;
        rgba.y = rgba.y > 1.0f ? 1.0f : rgba.y;
        rgba.z = rgba.z < 0.0f ? 0.0f : rgba.z;
        rgba.z = rgba.z > 1.0f ? 1.0f : rgba.z;
        rgba.w = rgba.w < 0.0f ? 0.0f : rgba.w;
        rgba.w = rgba.w > 1.0f ? 1.0f : rgba.w;
    	return (uint(rgba.w * 255) << 24) | (uint(rgba.z * 255) << 16) | (uint(rgba.y * 255) << 8) | uint(rgba.x * 255);
    }
}
}

#endif