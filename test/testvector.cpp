/*
 * testvector.h  14.09.2006 (mueller)
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testhelper.h"

#include "vislib/EqualFunc.h"
#include "vislib/Vector.h"
#include "vislib/Vector2D.h"
#include "vislib/Vector3D.h"


void TestVector(void) {
    using namespace vislib::math;

    Vector2D<float, FltEqualFunc> sv1;
    
    ::AssertEqual("sv1[0] == 0", sv1[0], 0.0f);
    ::AssertEqual("sv1[1] == 0", sv1[1], 0.0f);
    ::AssertEqual("sv1.GetX() == 0", sv1.GetX(), 0.0f);
    ::AssertEqual("sv1.GetY() == 0", sv1.GetY(), 0.0f);
    ::AssertEqual("sv1.X() == 0", sv1.X(), 0.0f);
    ::AssertEqual("sv1.Y() == 0", sv1.Y(), 0.0f);

    ::AssertEqual("sv1.Length() == 0", sv1.Length(), 0.0f);
    ::AssertEqual("sv1.MaxNorm() == 0", sv1.MaxNorm(), 0.0f);
    ::AssertTrue("sv1.IsNull()", sv1.IsNull());

    sv1.SetX(2.0f);
    ::AssertEqual("sv1[0] == 2", sv1[0], 2.0f);
    ::AssertEqual("sv1[1] == 0", sv1[1], 0.0f);
    ::AssertEqual("sv1.GetX() == 2", sv1.GetX(), 2.0f);
    ::AssertEqual("sv1.GetY() == 0", sv1.GetY(), 0.0f);
    ::AssertEqual("sv1.X() == 2", sv1.X(), 2.0f);
    ::AssertEqual("sv1.Y() == 0", sv1.Y(), 0.0f);

    ::AssertNearlyEqual("sv1.Length() == 2", sv1.Length(), 2.0f);
    ::AssertEqual("sv1.MaxNorm() == 3", sv1.MaxNorm(), 2.0f);
    ::AssertFalse("!sv1.IsNull()", sv1.IsNull());

    sv1.SetY(3.0f);
    ::AssertEqual("sv1[0] == 2", sv1[0], 2.0f);
    ::AssertEqual("sv1[1] == 3", sv1[1], 3.0f);
    ::AssertEqual("sv1.GetX() == 2", sv1.GetX(), 2.0f);
    ::AssertEqual("sv1.GetY() == 3", sv1.GetY(), 3.0f);
    ::AssertEqual("sv1.X() == 2", sv1.X(), 2.0f);
    ::AssertEqual("sv1.Y() == 3", sv1.Y(), 3.0f);
    
    ::AssertNearlyEqual("sv1.Length()", sv1.Length(), ::sqrtf(2.0f * 2.0f + 3.0f * 3.0f));
    ::AssertEqual("sv1.MaxNorm() == 3", sv1.MaxNorm(), 3.0f);
    ::AssertFalse("!sv1.IsNull()", sv1.IsNull());
}
