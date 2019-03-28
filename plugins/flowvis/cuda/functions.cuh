#pragma once

#include <cuda_runtime_api.h>

#include "real_type.h"

/**
* Manual bilinear interpolation of values or vectors within a voxel
*
* @param position Position at which to interpolate
*
* @return Interpolated tuple
*/
template <int dimension>
__device__
typename real_t<float, dimension>::type texture_interpolation(const cudaTextureObject_t texture, const float2 position)
{
    // Calculate lower and upper corners of the interpolated voxel
    const float2 lc = { floor(position.x - 0.5f) + 0.5f, floor(position.y - 0.5f) + 0.5f };
    const float2 uc = lc + make_float2(1.0);

    // Calculate relative position within the voxel
    const float2 a = { position.x - lc.x, position.y - lc.y };

    // Interpolate linearly
    const auto t0 = tex2D<typename real_t<float, dimension>::type>(texture, lc.x, lc.y);
    const auto t1 = tex2D<typename real_t<float, dimension>::type>(texture, uc.x, lc.y);
    const auto t2 = tex2D<typename real_t<float, dimension>::type>(texture, uc.x, uc.y);
    const auto t3 = tex2D<typename real_t<float, dimension>::type>(texture, lc.x, uc.y);

    const auto ra = (1.0f - a.x) * t0 + a.x * t1;
    const auto rb = (1.0f - a.x) * t3 + a.x * t2;

    // Interpolate linearly between previous results
    return (1.0f - a.y) * ra + a.y * rb;
}

/**
* Calculate the distance between a point and a line
*
* @param p Point
* @param lp0 First line endpoint
* @param lp1 Second line endpoint
*
* @return Distance between the point and line
*/
inline __device__
float distance_point_line(const float2 p, const float2 lp0, const float2 lp1)
{
    // Calculate lenght and line vector
    const float len = length(lp1 - lp0);
    const float2 line_vector = normalizeSafe(lp1 - lp0);

    // Test if one of the line endpoints is closest to the point
    const float d = dot(line_vector, p - lp0);

    if (d < static_cast<float>(0.0))
    {
        return length(p - lp0);
    }
    if (d > len)
    {
        return length(p - lp1);
    }

    // Project point onto line and calculate the distance
    const float2 pproj = lp0 + line_vector * d;

    return length(p - pproj);
}
