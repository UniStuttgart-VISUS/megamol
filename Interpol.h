//
// Interpol.h
//
// Copyright (C) 2013 by University of Stuttgart (VISUS).
// All rights reserved.
//
// Created on: Feb 07, 2013
//     Author: scharnkn
//

#ifndef MMPROTEINPLUGIN_INTERPOL_H_INCLUDED
#define MMPROTEINPLUGIN_INTERPOL_H_INCLUDED

#include "vislib/Matrix.h"

namespace megamol {
namespace protein {

class Interpol {
public:

    /**
     * Executes linear interpolation based on two neighbor values and
     * a coordinate in the range [0...1].
     *
     * @param n0, n1 The two neighbor values
     * @param alpha, beta, gamma The coordinates inside the cell
     */
    template <class T>
    static T Lin(const T n0, const T n1, float alpha);

    /**
     * Executes bilinear interpolation based on four neighbor values and
     * a set of two-dimensional coordinates in the range [0...1][0...1].
     *
     * @param n0...n3 The four neighbor values
     * @param alpha, beta, gamma The coordinates inside the cell
     */
    template <class T>
    static T Bilin(const T n0, const T n1, const T n2, const T n3,
            float alpha, float beta);

    /**
     * Executes trilinear interpolation based on eight neighbor values and
     * a set of three-dimensional coordinates in the range
     * [0...1][0...1][0...1].
     *
     * @param n0...n7 The eight neighbor values
     * @param alpha, beta, gamma The coordinates inside the cell
     */
    template <class T>
    static T Trilin(const T n0, const T n1, const T n2, const T n3,
            const T n4, const T n5, const T n6, const T n7, float alpha,
            float beta, float gamma);

    /**
     * Executes cubic interpolation based on four neighbor values and
     * a coordinate in the range [0...1].
     *
     * @param n The four neighbour values
     * @param alpha, beta, gamma The coordinates inside the cell
     */
    template <class T>
    static T Cubic(T n[4], float alpha);

    /**
     * Executes bicubic interpolation based on 16 neighbor values and
     * a set of two-dimensional coordinates in the range [0...1][0...1].
     *
     * @param n The neighbor values
     * @param alpha, beta, gamma The coordinates inside the cell
     */
    template <class T>
    static T Bicubic(T n[4][4], float alpha, float beta);

    /**
     * Executes tricubic interpolation based on 64 neighbor values and
     * a set of three-dimensional coordinates in the range
     * [0...1][0...1][0...1].
     *
     * @param n The neighbor values
     * @param alpha, beta, gamma The coordinates inside the cell
     */
    template <class T>
    static T Tricubic(T n[4][4][4], float alpha, float beta,
            float gamma);
};


/*
 * Interpol::LinInterp
 */
template <class T>
T Interpol::Lin(const T n0, const T n1, float alpha) {
    return (1.0f - alpha)*n0 + alpha*n1;
}


/*
 * Interpol::BilinInterp
 */
template <class T>
T Interpol::Bilin(const T n0, const T n1, const T n2,
        const T n3, float alpha, float beta) {

    T a, b, c, d;
    a = n0;
    b = n1 - n0;
    c = n2 - n0;
    d = n3 - n1 - n2 + n0;

    return a + b*alpha + c*beta + d*alpha*beta;
}


/*
 * Interpol::TrilinInterp
 */
template <class T>
T Interpol::Trilin(const T n0, const T n1, const T n2, const T n3,
        const T n4, const T n5, const T n6, const T n7, float alpha,
        float beta, float gamma) {

    T a, b, c, d, e, f, g, h;
    a = n0;
    b = n1 - n0;
    c = n2 - n0;
    d = n3 - n1 - n2 + n0;
    e = n4 - n0;
    f = n5 - n1 - n4 + n0;
    g = n6 - n2 - n4 + n0;
    h = n7 - n3 - n5 - n6 + n1 + n2 + n4 - n0;

    return a + b*alpha + c*beta + d*alpha*beta + e*gamma + f*alpha*gamma
            + g*beta*gamma + h*alpha*beta*gamma;
}


/*
 * Interpol::CubInterp
 */
template <class T>
T Interpol::Cubic(T n[4], float alpha) {
    //  see http://www.paulinternet.nl/?page=bicubic
    return n[1] + 0.5*alpha*(n[2] - n[0] + alpha*(2.0*n[0] - 5.0*n[1] +
            4.0*n[2] - n[3] + alpha*(3.0*(n[1] - n[2]) + n[3] - n[0])));
}


/*
 * Interpol::BicubInterp
 */
template <class T>
T Interpol::Bicubic(T n[4][4], float alpha, float beta) {
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
template <class T>
T Interpol::Tricubic(T n[4][4][4], float alpha, float beta, float gamma) {
    // see http://www.paulinternet.nl/?page=bicubic
    double arr[4];
    arr[0] = Interpol::Bicubic(n[0], beta, gamma);
    arr[1] = Interpol::Bicubic(n[1], beta, gamma);
    arr[2] = Interpol::Bicubic(n[2], beta, gamma);
    arr[3] = Interpol::Bicubic(n[3], beta, gamma);
    return Interpol::Cubic(arr, alpha);
}

} // end namespace protein
} // end namespace megamol

#endif // MMPROTEINPLUGIN_INTERPOL_H_INCLUDED
