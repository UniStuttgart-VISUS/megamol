/*
 * BSpline.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_BSPLINE_H_INCLUDED
#define MEGAMOL_BSPLINE_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "vislib/Array.h"
#include "vislib/math/Matrix4.h"
#include "vislib/math/Vector.h"
#include <vector>

namespace megamol {
namespace protein {

class BSpline {
public:
    BSpline();
    virtual ~BSpline();

    // set the coordinates for the geometry matrix G
    void setG(vislib::math::Vector<float, 3> v1, vislib::math::Vector<float, 3> v2, vislib::math::Vector<float, 3> v3,
        vislib::math::Vector<float, 3> v4);

    // set the number of segments to create --> this function also sets matrix S!
    void setN(unsigned int n);

    // set the backbone atom coordinates
    void setBackbone(std::vector<vislib::math::Vector<float, 3>> bb) {
        backbone.clear();
        backbone = bb;
    };

    // get the result vector
    void getResult(std::vector<vislib::math::Vector<float, 3>>& res) {
        res = result;
    };

protected:
    vislib::math::Matrix<float, 4, vislib::math::ROW_MAJOR> S;
    vislib::math::Matrix<float, 4, vislib::math::ROW_MAJOR> B;
    vislib::math::Matrix<float, 4, vislib::math::ROW_MAJOR> G;
    vislib::math::Matrix<float, 4, vislib::math::ROW_MAJOR> M;

    unsigned int N;

    std::vector<vislib::math::Vector<float, 3>> backbone;
    std::vector<vislib::math::Vector<float, 3>> result;

public:
    // compute the spline from the given backbone coordinates
    bool computeSpline();
};

} // namespace protein
} // namespace megamol

#endif /* MEGAMOL_BSPLINE_H_INCLUDED */
