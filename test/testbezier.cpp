/*
 * testbezier.cpp
 *
 * Copyright (C) 2009 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "testbezier.h"

#include "testhelper.h"
#include "vislib/BezierCurve.h"
#include "vislib/Point.h"
#include "vislib/Vector.h"


void TestBezier(void) {
    using vislib::math::Point;
    using vislib::math::Vector;
    typedef vislib::math::BezierCurve<Point<float, 3>, 3> BezierCurve;

    BezierCurve curve;
    curve.ControlPoint(0).Set(0.0f, -1.0f, 0.0f);
    curve.ControlPoint(1).Set(0.0f, 1.0f, 0.0f);
    curve.ControlPoint(2).Set(2.0f, -1.0f, 0.0f);
    curve.ControlPoint(3).Set(2.0f, 1.0f, 0.0f);

    Point<float, 3> p1;
    curve.CalcPoint(p1, 0.0f);
    Point<float, 3> p2;
    curve.CalcPoint(p2, 0.5f);
    Point<float, 3> p3;
    curve.CalcPoint(p3, 1.0f);

    Vector<float, 3> t1;
    curve.CalcTangent(t1, 0.0f).Normalise();
    Vector<float, 3> t2;
    curve.CalcTangent(t2, 0.5f).Normalise();
    Vector<float, 3> t3;
    curve.CalcTangent(t3, 1.0f).Normalise();

    AssertEqual("Point at t=0 correct",   p1, Point<float, 3>(0.0f, -1.0f, 0.0f));
    AssertEqual("Point at t=0.5 correct", p2, Point<float, 3>(1.0f, 0.0f, 0.0f));
    AssertEqual("Point at t=1 correct",   p3, Point<float, 3>(2.0f, 1.0f, 0.0f));

    AssertEqual("Tangent at t=0 correct",   t1, Vector<float, 3>(0.0f, 1.0f, 0.0f));
    AssertEqual("Tangent at t=0.5 correct", t2, Vector<float, 3>(1.0f, 0.0f, 0.0f));
    AssertEqual("Tangent at t=1 correct",   t3, Vector<float, 3>(0.0f, 1.0f, 0.0f));

}
