//
// interpol.cuh
//
// Contains CUDA functionality to interpolate values in device memory.
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Apr 13, 2013
//     Author: scharnkn
//

#ifndef MMPROTEINCUDAPLUGIN_INTERPOL_CUH_INCLUDED
#define MMPROTEINCUDAPLUGIN_INTERPOL_CUH_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

// Shut up eclipse syntax error highlighting
#ifdef __CDT_PARSER__
#define __device__
#define __global__
#define __shared__
#define __constant__
#endif


template <typename T>
inline __device__ T InterpFieldLin_D(T v0, T v1, float alpha) {
    return v0+alpha*(v1-v0);
}


template <typename T>
inline __device__ T InterpFieldBilin_D(T v0, T v1, T v2, T v3,
        float alpha, float beta) {
    return InterpFieldLin_D<T>(InterpFieldLin_D<T>(v0, v1, alpha),
            InterpFieldLin_D<T>(v2, v3, alpha), beta);
}


template <typename T>
inline __device__ T InterpFieldTrilin_D(T v[8], float alpha, float beta,
        float gamma) {
    return InterpFieldLin_D<T>(
            InterpFieldBilin_D<T>(v[0], v[1], v[2], v[3], alpha, beta),
            InterpFieldBilin_D<T>(v[4], v[5], v[6], v[7], alpha, beta), gamma);
}


template <typename T>
inline __device__ T InterpFieldCubic_D(T n[4], float alpha) {

    return n[1] + 0.5*alpha*(n[2] - n[0] + alpha*(2.0*n[0] - 5.0*n[1] +
            4.0*n[2] - n[3] + alpha*(3.0*(n[1] - n[2]) + n[3] - n[0])));
}

//template <typename T>
//inline __device__ T InterpFieldCubicSepArgs_D(T n0, T n1, T n2, T n3, float alpha) {
//    return n1 + 0.5*alpha*(n2 - n0 + alpha*(2.0*n0 - 5.0*n1 +
//            4.0*n2 - n3 + alpha*(3.0*(n1 - n2) + n3 - n0)));
//}


template <typename T>
inline __device__ T InterpFieldCubicSepArgs_D(T n0, T n1, T n2, T n3, float alpha) {

    T res = n1 + 0.5*alpha*(n2 - n0 + alpha*(2.0*n0 - 5.0*n1 +
            4.0*n2 - n3 + alpha*(3.0*(n1 - n2) + n3 - n0)));
    return res;

//     T P = (n3 - n2) - (n0 - n1);
//     T Q = (n0 - n1)- P;
//     T R = n2 - n0;
//     T S = n1;
//     return P * alpha*alpha*alpha + Q * alpha*alpha + R * alpha + S;
}


template <typename T>
inline __device__ T InterpFieldBicubic_D(T n[4][4], float alpha, float beta) {

    T arr[4];
    arr[0] = InterpFieldCubic_D(n[0], beta);
    arr[1] = InterpFieldCubic_D(n[1], beta);
    arr[2] = InterpFieldCubic_D(n[2], beta);
    arr[3] = InterpFieldCubic_D(n[3], beta);
    return InterpFieldCubic_D(arr, alpha);
}


template <typename T>
inline __device__ T InterpFieldTricubic_D(T n[4][4][4], float alpha, float beta,
        float gamma) {

    T arr[4];
    arr[0] = InterpFieldBicubic_D(n[0], beta, gamma);
    arr[1] = InterpFieldBicubic_D(n[1], beta, gamma);
    arr[2] = InterpFieldBicubic_D(n[2], beta, gamma);
    arr[3] = InterpFieldBicubic_D(n[3], beta, gamma);
    return InterpFieldCubic_D(arr, alpha);
}

#endif // MMPROTEINCUDAPLUGIN_INTERPOL_CUH_INCLUDED
