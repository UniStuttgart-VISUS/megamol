#pragma once

#include <cuda_runtime_api.h>

#include "real_type.h"

/**
* Manual trilinear interpolation of Real4 tuples within a voxel
*
* @param position Position at which to interpolate
*
* @return Interpolated tuple
*/
template <int dimension>
__device__
typename real_t<Real, dimension>::type tex3D_interp(const cudaTextureObject_t texture, const Real4 position)
{
    // Calculate lower and upper corners of the interpolated voxel
    const Real3 lc = {
        floor(position.x - static_cast<Real>(0.5)) + static_cast<Real>(0.5),
        floor(position.y - static_cast<Real>(0.5)) + static_cast<Real>(0.5),
        floor(position.z - static_cast<Real>(0.5)) + static_cast<Real>(0.5) };
    const Real3 uc = lc + make_real<Real, 3>(1.0);

    // Calculate relative position within the voxel
    const Real3 a = { position.x - lc.x, position.y - lc.y, position.z - lc.z };

    // Interpolate bilinearly
    const auto t0 = make_real<Real, dimension, float>(tex3D<typename real_t<float, dimension>::type>(texture, lc.x, lc.y, lc.z));
    const auto t1 = make_real<Real, dimension, float>(tex3D<typename real_t<float, dimension>::type>(texture, uc.x, lc.y, lc.z));
    const auto t2 = make_real<Real, dimension, float>(tex3D<typename real_t<float, dimension>::type>(texture, uc.x, uc.y, lc.z));
    const auto t3 = make_real<Real, dimension, float>(tex3D<typename real_t<float, dimension>::type>(texture, lc.x, uc.y, lc.z));

    const auto ra = (static_cast<Real>(1.0) - a.x) * t0 + a.x * t1;
    const auto rb = (static_cast<Real>(1.0) - a.x) * t3 + a.x * t2;
    const auto r1 = (static_cast<Real>(1.0) - a.y) * ra + a.y * rb;

    // Interpolate bilinearly
    const auto t4 = make_real<Real, dimension, float>(tex3D<typename real_t<float, dimension>::type>(texture, lc.x, lc.y, uc.z));
    const auto t5 = make_real<Real, dimension, float>(tex3D<typename real_t<float, dimension>::type>(texture, uc.x, lc.y, uc.z));
    const auto t6 = make_real<Real, dimension, float>(tex3D<typename real_t<float, dimension>::type>(texture, uc.x, uc.y, uc.z));
    const auto t7 = make_real<Real, dimension, float>(tex3D<typename real_t<float, dimension>::type>(texture, lc.x, uc.y, uc.z));

    const auto rc = (static_cast<Real>(1.0) - a.x) * t4 + a.x * t5;
    const auto rd = (static_cast<Real>(1.0) - a.x) * t7 + a.x * t6;
    const auto r2 = (static_cast<Real>(1.0) - a.y) * rc + a.y * rd;

    // Interpolate linearly between previous results (trilinear)
    return (static_cast<Real>(1.0) - a.z) * r1 + a.z * r2;
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
Real distance_point_line(const Real3 p, const Real3 lp0, const Real3 lp1)
{
    // Calculate lenght and line vector
    const Real len = length(lp1 - lp0);
    const Real3 line_vector = normalizeSafe(lp1 - lp0);

    // Test if one of the line endpoints is closest to the point
    const Real d = dot(line_vector, p - lp0);

    if (d < static_cast<Real>(0.0))
    {
        return length(p - lp0);
    }
    if (d > len)
    {
        return length(p - lp1);
    }

    // Project point onto line and calculate the distance
    const Real3 pproj = lp0 + line_vector * d;

    return length(p - pproj);
}

/**
* Calculate the distance between a point and a triangle
*
* @param p Point
* @param p0 First triangle corner point
* @param p1 Second triangle corner point
* @param p2 Third triangle corner point
*
* @return Distance between the point and triangle
*/
inline __device__
Real distance_point_triangle(const Real3 p, const Real3 p0, const Real3 p1, const Real3 p2)
{
    // Calculate edge vectors, the normal and the projection of the point onto the triangle
    const Real3 e0 = normalizeSafe(p1 - p0);
    const Real3 e1 = normalizeSafe(p2 - p0);
    const Real3 e2 = normalizeSafe(p2 - p1);

    const Real3 n = normalizeSafe(cross(e0, e1));

    const Real3 pproj = p - n * dot(n, p - p0);

    // Check the projected position to find out where the point lies
    const Real3 cr0 = cross(e0, pproj - p0);
    const bool inside_e0 = dot(cr0, n) > static_cast<Real>(0.0) ? true : false;

    const Real3 cr1 = cross(pproj - p0, e1);
    const bool inside_e1 = dot(cr0, n) > static_cast<Real>(0.0) ? true : false;

    const Real3 cr2 = cross(e2, pproj - p0);
    const bool inside_e2 = dot(cr0, n) > static_cast<Real>(0.0) ? true : false;

    // Calculate distance based on the relative position
    if (inside_e0 && inside_e1 && inside_e2)
    {
        return length(pproj - p);
    }
    else if (!inside_e0)
    {
        return distance_point_line(p, p0, p1);
    }
    else if (!inside_e1)
    {
        return distance_point_line(p, p0, p2);
    }
    else
    {
        return distance_point_line(p, p1, p2);
    }
}
