/*
 * testquaternion.h
 *
 * Copyright (C) 2006 - 2007 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "testquaternion.h"

#include "testhelper.h"
#include "vislib/Quaternion.h"


void TestQuaternion(void) {
    using namespace vislib::math;
    Quaternion<float> q1;
    Quaternion<float> q2(1.0f, 0.0, 0.0, 1.0f);
    Quaternion<float> q3;

    AssertEqual("Quaternion dft ctor x", q1.X(), 0.0f);
    AssertEqual("Quaternion dft ctor y", q1.Y(), 0.0f);
    AssertEqual("Quaternion dft ctor z", q1.Z(), 0.0f);
    AssertEqual("Quaternion dft ctor w", q1.W(), 1.0f);

    AssertEqual("Quaternion 4 component ctor x", q2.X(), 1.0f);
    AssertEqual("Quaternion 4 component ctor y", q2.Y(), 0.0f);
    AssertEqual("Quaternion 4 component ctor z", q2.Z(), 0.0f);
    AssertEqual("Quaternion 4 component ctor w", q2.W(), 1.0f);

    q3 = q1 * q2;

}
