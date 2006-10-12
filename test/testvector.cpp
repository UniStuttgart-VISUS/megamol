/*
 * testvector.h  14.09.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testhelper.h"

#include "vislib/ShallowVector.h"
#include "vislib/Vector.h"


void TestVector(void) {
    typedef vislib::math::Vector<float, 2> Vector2D;
    typedef vislib::math::ShallowVector<float, 2> ShallowVector2D;

    float sv1data[2];
    float sv2data[2];

    Vector2D v1;
    ShallowVector2D sv1(sv1data);
    ShallowVector2D sv2(sv2data);
    
    ::AssertEqual("v1[0] == 0", v1[0], 0.0f);
    ::AssertEqual("v1[1] == 0", v1[1], 0.0f);
    ::AssertEqual("v1.GetX() == 0", v1.GetX(), 0.0f);
    ::AssertEqual("v1.GetY() == 0", v1.GetY(), 0.0f);
    ::AssertEqual("v1.X() == 0", v1.X(), 0.0f);
    ::AssertEqual("v1.Y() == 0", v1.Y(), 0.0f);

    ::AssertEqual("v1.Length() == 0", v1.Length(), 0.0f);
    ::AssertEqual("v1.MaxNorm() == 0", v1.MaxNorm(), 0.0f);
    ::AssertTrue("v1.IsNull()", v1.IsNull());

    sv1 = v1;
    ::AssertTrue("Deep to shallow assignment", (v1 == sv1));
    ::AssertEqual("sv1[0] == 0", sv1[0], 0.0f);
    ::AssertEqual("sv1[1] == 0", sv1[1], 0.0f);
    ::AssertEqual("sv1.GetX() == 0", sv1.GetX(), 0.0f);
    ::AssertEqual("sv1.GetY() == 0", sv1.GetY(), 0.0f);
    ::AssertEqual("sv1.X() == 0", sv1.X(), 0.0f);
    ::AssertEqual("sv1.Y() == 0", sv1.Y(), 0.0f);

    sv2 = sv1;
    ::AssertEqual("Shallow to shallow assignment", sv1, sv2);
    ::AssertEqual("sv2[0] == 0", sv2[0], 0.0f);
    ::AssertEqual("sv2[1] == 0", sv2[1], 0.0f);
    ::AssertEqual("sv2.GetX() == 0", sv2.GetX(), 0.0f);
    ::AssertEqual("sv2.GetY() == 0", sv2.GetY(), 0.0f);
    ::AssertEqual("sv2.X() == 0", sv2.X(), 0.0f);
    ::AssertEqual("sv1.Y() == 0", sv2.Y(), 0.0f);

    v1.SetX(2.0f);
    ::AssertEqual("v1[0] == 2", v1[0], 2.0f);
    ::AssertEqual("v1[1] == 0", v1[1], 0.0f);
    ::AssertEqual("v1.GetX() == 2", v1.GetX(), 2.0f);
    ::AssertEqual("v1.GetY() == 0", v1.GetY(), 0.0f);
    ::AssertEqual("v1.X() == 2", v1.X(), 2.0f);
    ::AssertEqual("v1.Y() == 0", v1.Y(), 0.0f);

    ::AssertNearlyEqual("v1.Length() == 2", v1.Length(), 2.0f);
    ::AssertEqual("v1.MaxNorm() == 3", v1.MaxNorm(), 2.0f);
    ::AssertFalse("!v1.IsNull()", v1.IsNull());

    v1.SetY(3.0f);
    ::AssertEqual("v1[0] == 2", v1[0], 2.0f);
    ::AssertEqual("v1[1] == 3", v1[1], 3.0f);
    ::AssertEqual("v1.GetX() == 2", v1.GetX(), 2.0f);
    ::AssertEqual("v1.GetY() == 3", v1.GetY(), 3.0f);
    ::AssertEqual("v1.X() == 2", v1.X(), 2.0f);
    ::AssertEqual("v1.Y() == 3", v1.Y(), 3.0f);
    
    ::AssertNearlyEqual("v1.Length()", v1.Length(), ::sqrtf(2.0f * 2.0f + 3.0f * 3.0f));
    ::AssertEqual("v1.MaxNorm() == 3", v1.MaxNorm(), 3.0f);
    ::AssertFalse("!v1.IsNull()", v1.IsNull());
}
