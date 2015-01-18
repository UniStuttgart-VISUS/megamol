/*
 * testquaternion.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "testquaternion.h"

#define _USE_MATH_DEFINES
#include <cmath>
#include "testhelper.h"
#include "vislib/Quaternion.h"
#include "vislib/Vector.h"


void TestQuaternion(void) {
    using namespace vislib::math;
    Quaternion<double> q1;
    Quaternion<double> q2(1.0, 0.0, 0.0, 1.0);
    Quaternion<double> q3;

    AssertEqual("Quaternion dft ctor x", q1.X(), 0.0);
    AssertEqual("Quaternion dft ctor y", q1.Y(), 0.0);
    AssertEqual("Quaternion dft ctor z", q1.Z(), 0.0);
    AssertEqual("Quaternion dft ctor w", q1.W(), 1.0);

    AssertEqual("Quaternion 4 component ctor x", q2.X(), 1.0);
    AssertEqual("Quaternion 4 component ctor y", q2.Y(), 0.0);
    AssertEqual("Quaternion 4 component ctor z", q2.Z(), 0.0);
    AssertEqual("Quaternion 4 component ctor w", q2.W(), 1.0);

    q3 = q1 * q2;

    q1.Set(M_PI * 0.5, Vector<double, 3>(0.0, 1.0, 0.0)); // rotates 90° around the y-axis
    Vector<double, 3> v1 = q1 * Vector<double, 3>(1.0, 0.0f, 0.0);
    AssertEqual("Rotation 1 works", v1, Vector<double, 3>(0.0, 0.0, -1.0));

    q2.Set(M_PI * 0.25, Vector<double, 3>(1.0, 0.0, 0.0)); // rotates 45° around the x-axis
    Vector<double, 3> v2 = q2 * v1;
    AssertEqual("Rotation 2 works", v2, Vector<double, 3>(0.0, 0.5 * sqrt(2.0), -0.5 * sqrt(2.0)));

    q3 = q2 * q1;
    Vector<double, 3> v3 = q3 * Vector<double, 3>(1.0, 0.0f, 0.0);
    AssertEqual("Combined rotation works", v2, v3);

    v1.Set(1.0, -2.0, 3.0);
    v1.Normalise();
    double a = M_PI * 0.987654321;
    q1.Set(a, v1);
    double b;
    q1.AngleAndAxis(b, v2);

    AssertNearlyEqual("Angle reconstructed", a, b);
    AssertEqual("Axis reconstructed", v1, v2);

}
