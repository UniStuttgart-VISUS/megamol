//
// Interpol.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Feb 07, 2013
//     Author: scharnkn
//

#ifndef MMPROTEINCALLPLUGIN_INTERPOL_H_INCLUDED
#define MMPROTEINCALLPLUGIN_INTERPOL_H_INCLUDED

#include "vislib/math/Matrix.h"

namespace megamol::protein_calls {

class Interpol {
public:
    /**
     * Executes linear interpolation based on two neighbor values and
     * a coordinate in the range [0...1].
     *
     * @param n0, n1 The two neighbor values
     * @param alpha, beta, gamma The coordinates inside the cell
     */
    template<class T>
    inline static T Lin(const T n0, const T n1, float alpha);

    /**
     * Executes bilinear interpolation based on four neighbor values and
     * a set of two-dimensional coordinates in the range [0...1][0...1].
     *
     * @param n0...n3 The four neighbor values
     * @param alpha, beta, gamma The coordinates inside the cell
     */
    template<class T>
    inline static T Bilin(const T n0, const T n1, const T n2, const T n3, float alpha, float beta);

    /**
     * Executes trilinear interpolation based on eight neighbor values and
     * a set of three-dimensional coordinates in the range
     * [0...1][0...1][0...1].
     *
     * @param n0...n7 The eight neighbor values
     * @param alpha, beta, gamma The coordinates inside the cell
     */
    template<class T>
    inline static T Trilin(const T n0, const T n1, const T n2, const T n3, const T n4, const T n5, const T n6,
        const T n7, float alpha, float beta, float gamma);

    /**
     * Executes cubic interpolation based on four neighbor values and
     * a coordinate in the range [0...1].
     *
     * @param n The four neighbour values
     * @param alpha, beta, gamma The coordinates inside the cell
     */
    template<class T>
    inline static T Cubic(T n[4], float alpha);

    /**
     * Executes bicubic interpolation based on 16 neighbor values and
     * a set of two-dimensional coordinates in the range [0...1][0...1].
     *
     * @param n The neighbor values
     * @param alpha, beta, gamma The coordinates inside the cell
     */
    template<class T>
    inline static T Bicubic(T n[4][4], float alpha, float beta);

    /**
     * Executes tricubic interpolation based on 64 neighbor values and
     * a set of three-dimensional coordinates in the range
     * [0...1][0...1][0...1].
     *
     * @param n The neighbor values
     * @param alpha, beta, gamma The coordinates inside the cell
     */
    template<class T>
    inline static T Tricubic(T n[4][4][4], float alpha, float beta, float gamma);
};


/*
 * Interpol::LinInterp
 */
template<class T>
inline T Interpol::Lin(const T n0, const T n1, float alpha) {
    return (1.0f - alpha) * n0 + alpha * n1;
}


/*
 * Interpol::BilinInterp
 */
template<class T>
inline T Interpol::Bilin(const T n0, const T n1, const T n2, const T n3, float alpha, float beta) {

    T a, b, c, d;
    a = n0;
    b = n1 - n0;
    c = n2 - n0;
    d = n3 - n1 - n2 + n0;

    return a + b * alpha + c * beta + d * alpha * beta;
}


/*
 * Interpol::TrilinInterp
 */
template<class T>
inline T Interpol::Trilin(const T n0, const T n1, const T n2, const T n3, const T n4, const T n5, const T n6,
    const T n7, float alpha, float beta, float gamma) {

    T a, b, c, d, e, f, g, h;
    a = n0;
    b = n1 - n0;
    c = n2 - n0;
    d = n3 - n1 - n2 + n0;
    e = n4 - n0;
    f = n5 - n1 - n4 + n0;
    g = n6 - n2 - n4 + n0;
    h = n7 - n3 - n5 - n6 + n1 + n2 + n4 - n0;

    return a + b * alpha + c * beta + d * alpha * beta + e * gamma + f * alpha * gamma + g * beta * gamma +
           h * alpha * beta * gamma;
}


/*
 * Interpol::CubInterp
 */
template<class T>
inline T Interpol::Cubic(T n[4], float alpha) {
    //  see http://www.paulinternet.nl/?page=bicubic
    return n[1] +
           0.5 * alpha *
               (n[2] - n[0] +
                   alpha * (2.0 * n[0] - 5.0 * n[1] + 4.0 * n[2] - n[3] + alpha * (3.0 * (n[1] - n[2]) + n[3] - n[0])));
}


/*
 * Interpol::BicubInterp
 */
template<class T>
inline T Interpol::Bicubic(T n[4][4], float alpha, float beta) {
    // see http://www.paulinternet.nl/?page=bicubic
    double arr[4];
    arr[0] = Interpol::Cubic(n[0], beta);
    arr[1] = Interpol::Cubic(n[1], beta);
    arr[2] = Interpol::Cubic(n[2], beta);
    arr[3] = Interpol::Cubic(n[3], beta);
    return Interpol::Cubic(arr, alpha);
}


/*
 * Interpol::TricubInterp
 */
template<class T>
inline T Interpol::Tricubic(T n[4][4][4], float alpha, float beta, float gamma) {
    // see http://www.paulinternet.nl/?page=bicubic
    double arr[4];
    arr[0] = Interpol::Bicubic(n[0], beta, gamma);
    arr[1] = Interpol::Bicubic(n[1], beta, gamma);
    arr[2] = Interpol::Bicubic(n[2], beta, gamma);
    arr[3] = Interpol::Bicubic(n[3], beta, gamma);
    return Interpol::Cubic(arr, alpha);
}

/**
 * Samples the field at a given position using linear interpolation.
 *
 * @param pos The position
 * @return The sampled value of the field
 */
template<typename T>
inline T SampleFieldAtPosTrilin(float pos[3], T* field, float gridOrg[3], float gridDelta[3], int gridSize[3]) {

    int c[3];
    float f[3];

    // Get id of the cell containing the given position and interpolation
    // coefficients
    f[0] = (pos[0] - gridOrg[0]) / gridDelta[0];
    f[1] = (pos[1] - gridOrg[1]) / gridDelta[1];
    f[2] = (pos[2] - gridOrg[2]) / gridDelta[2];
    c[0] = (int)(f[0]);
    c[1] = (int)(f[1]);
    c[2] = (int)(f[2]);
    f[0] = f[0] - (float)c[0]; // alpha
    f[1] = f[1] - (float)c[1]; // beta
    f[2] = f[2] - (float)c[2]; // gamma

    c[0] = std::min(std::max(c[0], int(0)), gridSize[0] - 2);
    c[1] = std::min(std::max(c[1], int(0)), gridSize[1] - 2);
    c[2] = std::min(std::max(c[2], int(0)), gridSize[2] - 2);

    // Get values at corners of current cell
    T s[8];
    s[0] = field[gridSize[0] * (gridSize[1] * (c[2] + 0) + (c[1] + 0)) + c[0] + 0];
    s[1] = field[gridSize[0] * (gridSize[1] * (c[2] + 0) + (c[1] + 0)) + c[0] + 1];
    s[2] = field[gridSize[0] * (gridSize[1] * (c[2] + 0) + (c[1] + 1)) + c[0] + 0];
    s[3] = field[gridSize[0] * (gridSize[1] * (c[2] + 0) + (c[1] + 1)) + c[0] + 1];
    s[4] = field[gridSize[0] * (gridSize[1] * (c[2] + 1) + (c[1] + 0)) + c[0] + 0];
    s[5] = field[gridSize[0] * (gridSize[1] * (c[2] + 1) + (c[1] + 0)) + c[0] + 1];
    s[6] = field[gridSize[0] * (gridSize[1] * (c[2] + 1) + (c[1] + 1)) + c[0] + 0];
    s[7] = field[gridSize[0] * (gridSize[1] * (c[2] + 1) + (c[1] + 1)) + c[0] + 1];

    // Use trilinear interpolation to sample the volume
    return Interpol::Trilin<T>(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], f[0], f[1], f[2]);
}


} // namespace megamol::protein_calls

#endif // MMPROTEINCALLPLUGIN_INTERPOL_H_INCLUDED
