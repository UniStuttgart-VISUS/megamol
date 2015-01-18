/*
 * testvector.cpp  27.11.2008
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 * Copyright (C) 2008 by Christoph Müller. Alle Rechte vorbehalten.
 */

#include "testpoint.h"

#include "vislib/Point.h"
#include "vislib/ShallowPoint.h"
#include "testhelper.h"

typedef vislib::math::Point<float, 2> Point2f;
typedef vislib::math::Point<float, 3> Point3f;
typedef vislib::math::Vector<float, 2> Vector2f;
typedef vislib::math::Vector<float, 3> Vector3f;


void TestPoint2D(void) {
    Point2f p1;
    Point2f p2;
    Point2f p3;

    ::AssertEqual("Default ctor x", p1.X(), 0.0f);
    ::AssertEqual("Default ctor x (get)", p1.GetX(), 0.0f);
    ::AssertEqual("Default ctor y", p1.Y(), 0.0f);
    ::AssertEqual("Default ctor y (get)", p1.GetY(), 0.0f);
    ::AssertTrue("IsOrigin", p1.IsOrigin());
    ::AssertTrue("Equality", p1 == p2);
    ::AssertFalse("Inequality", p1 != p2);

    p1.SetX(1.0f);
    ::AssertEqual("SetX", p1.X(), 1.0f);
    ::AssertFalse("!IsOrigin", p1.IsOrigin());
    ::AssertFalse("Equality", p1 == p2);
    ::AssertTrue("Inequality", p1 != p2);

    p1.SetY(2.0f);
    ::AssertEqual("SetY", p1.Y(), 2.0f);
    ::AssertFalse("!IsOrigin", p1.IsOrigin());
    ::AssertFalse("Equality", p1 == p2);
    ::AssertTrue("Inequality", p1 != p2);

    p1[0] = 2.0f;
    ::AssertEqual("Array write x", p1.X(), 2.0f);

    p1[1] = 4.0f;
    ::AssertEqual("Array write y", p1.Y(), 4.0f);

    AssertException("Illegal array access", p1[2], vislib::OutOfRangeException);

    p2 = p1;
    ::AssertTrue("Assignment", p1 == p2);

    p1.Set(0.0f, 0.0f);
    p2.Set(1.0f, 0.0f);
    p3 = p1.Interpolate(p2, 0.5f);
    ::AssertEqual("Interpolate x", p3.X(), 0.5f);
    ::AssertEqual("Interpolate y", p3.Y(), 0.0f);
}


void TestPoint3D(void) {
    Point3f p1;
    Point3f p2;
    Point3f p3;

    ::AssertEqual("Default ctor x", p1.X(), 0.0f);
    ::AssertEqual("Default ctor x (get)", p1.GetX(), 0.0f);
    ::AssertEqual("Default ctor y", p1.Y(), 0.0f);
    ::AssertEqual("Default ctor y (get)", p1.GetY(), 0.0f);
    ::AssertEqual("Default ctor z", p1.Z(), 0.0f);
    ::AssertEqual("Default ctor z (get)", p1.GetZ(), 0.0f);
    ::AssertTrue("IsOrigin", p1.IsOrigin());
    ::AssertTrue("Equality", p1 == p2);
    ::AssertFalse("Inequality", p1 != p2);

    p1.SetX(1.0f);
    ::AssertEqual("SetX", p1.X(), 1.0f);
    ::AssertFalse("!IsOrigin", p1.IsOrigin());
    ::AssertFalse("Equality", p1 == p2);
    ::AssertTrue("Inequality", p1 != p2);

    p1.SetY(2.0f);
    ::AssertEqual("SetY", p1.Y(), 2.0f);
    ::AssertFalse("!IsOrigin", p1.IsOrigin());
    ::AssertFalse("Equality", p1 == p2);
    ::AssertTrue("Inequality", p1 != p2);

    p1.SetZ(5.0f);
    ::AssertEqual("SetZ", p1.Z(), 5.0f);
    ::AssertFalse("!IsOrigin", p1.IsOrigin());
    ::AssertFalse("Equality", p1 == p2);
    ::AssertTrue("Inequality", p1 != p2);

    p1[0] = 2.0f;
    ::AssertEqual("Array write x", p1.X(), 2.0f);

    p1[1] = 4.0f;
    ::AssertEqual("Array write y", p1.Y(), 4.0f);

    p1[2] = 7.0f;
    ::AssertEqual("Array write z", p1.Z(), 7.0f);

    AssertException("Illegal array access", p1[3], vislib::OutOfRangeException);

    p2 = p1;
    ::AssertTrue("Assignment", p1 == p2);

    p1.Set(0.0f, 0.0f, 0.0f);
    p2.Set(1.0f, 0.0f, 0.0f);
    p3 = p1.Interpolate(p2, 0.5f);
    ::AssertEqual("Interpolate x", p3.X(), 0.5f);
    ::AssertEqual("Interpolate y", p3.Y(), 0.0f);
    ::AssertEqual("Interpolate z", p3.Y(), 0.0f);
}


void TestHalfspace2D(void) {
#define DO_TEST(res) \
    desc.Format("p = (%3.1f, %3.1f), n = (%3.1f, %3.1f), t = (%3.1f, %3.1f)", \
    planePt.X(), planePt.Y(), \
    normal.X(), normal.Y(), \
    testPt.X(), testPt.Y()); \
    AssertEqual(desc.PeekBuffer(), planePt.Halfspace(normal, testPt), vislib::math::res)
    
    using vislib::math::HalfSpace;
    vislib::StringA desc;
    Point2f planePt;
    Vector2f normal;
    Point2f testPt;
    
    planePt.Set(0.0f, 0.0f);
    normal.Set(1.0f, 0.0f);

    testPt.Set(0.0f, 0.0f);
    DO_TEST(HALFSPACE_IN_PLANE);

    testPt.SetY(1.0f);
    DO_TEST(HALFSPACE_IN_PLANE);

    testPt.SetY(-1.0f);
    DO_TEST(HALFSPACE_IN_PLANE);

    testPt.SetX(1.0f);
    DO_TEST(HALFSPACE_POSITIVE);

    testPt.SetX(-1.0f);
    DO_TEST(HALFSPACE_NEGATIVE);

    testPt.Set(0.5f, 0.5f);
    DO_TEST(HALFSPACE_POSITIVE);

    testPt.SetX(-1.0f);
    DO_TEST(HALFSPACE_NEGATIVE);
    

    normal.Set(0.0f, 1.0f);

    testPt.Set(0.0f, 0.0f);
    DO_TEST(HALFSPACE_IN_PLANE);

    testPt.SetX(1.0f);
    DO_TEST(HALFSPACE_IN_PLANE);

    testPt.SetX(-1.0f);
    DO_TEST(HALFSPACE_IN_PLANE);

    testPt.SetY(1.0f);
    DO_TEST(HALFSPACE_POSITIVE);

    testPt.SetY(-1.0f);
    DO_TEST(HALFSPACE_NEGATIVE);


    normal.Set(0.5f, 0.5f);

    testPt.Set(0.0f, 0.0f);
    DO_TEST(HALFSPACE_IN_PLANE);

    testPt.Set(0.5f, -0.5f);
    DO_TEST(HALFSPACE_IN_PLANE);

    testPt.Set(-0.5f, 0.5f);
    DO_TEST(HALFSPACE_IN_PLANE);

    testPt.Set(1.0f, 1.0f);
    DO_TEST(HALFSPACE_POSITIVE);

    testPt.Set(-1.0f, -1.0f);
    DO_TEST(HALFSPACE_NEGATIVE);

    testPt.Set(1.0f, 0.0f);
    DO_TEST(HALFSPACE_POSITIVE);

    testPt.Set(-1.0f, 0.0f);
    DO_TEST(HALFSPACE_NEGATIVE);


    planePt.Set(1.0f, 0.0f);
    normal.Set(1.0f, 0.0f);

    testPt.Set(0.0f, 0.0f);
    DO_TEST(HALFSPACE_NEGATIVE);

    testPt.SetY(1.0f);
    DO_TEST(HALFSPACE_NEGATIVE);

    testPt.SetY(-1.0f);
    DO_TEST(HALFSPACE_NEGATIVE);

    testPt.SetX(1.0f);
    DO_TEST(HALFSPACE_IN_PLANE);

    testPt.SetX(-1.0f);
    DO_TEST(HALFSPACE_NEGATIVE);

    testPt.SetX(2.0f);
    DO_TEST(HALFSPACE_POSITIVE);

#undef DO_TEST
}


void TestHalfspace3D(void) {
#define DO_TEST(res) \
    desc.Format("p = (%3.1f, %3.1f, %3.1f), n = (%3.1f, %3.1f, %3.1f), t = (%3.1f, %3.1f, %3.1f)", \
    planePt.X(), planePt.Y(), planePt.Z(), \
    normal.X(), normal.Y(), planePt.Z(), \
    testPt.X(), testPt.Y(), planePt.Z()); \
    AssertEqual(desc.PeekBuffer(), planePt.Halfspace(normal, testPt), vislib::math::res)
    
    using vislib::math::HalfSpace;
    vislib::StringA desc;
    Point3f planePt;
    Vector3f normal;
    Point3f testPt;
    
    // Repeat 2D tests - these must succeed, too.
    planePt.Set(0.0f, 0.0f, 0.0f);
    normal.Set(1.0f, 0.0f, 0.0f);

    testPt.Set(0.0f, 0.0f, 0.0f);
    DO_TEST(HALFSPACE_IN_PLANE);

    testPt.SetY(1.0f);
    DO_TEST(HALFSPACE_IN_PLANE);

    testPt.SetY(-1.0f);
    DO_TEST(HALFSPACE_IN_PLANE);

    testPt.SetX(1.0f);
    DO_TEST(HALFSPACE_POSITIVE);

    testPt.SetX(-1.0f);
    DO_TEST(HALFSPACE_NEGATIVE);

    testPt.Set(0.5f, 0.5f, 0.0f);
    DO_TEST(HALFSPACE_POSITIVE);

    testPt.SetX(-1.0f);
    DO_TEST(HALFSPACE_NEGATIVE);
    

    normal.Set(0.0f, 1.0f, 0.0f);

    testPt.Set(0.0f, 0.0f, 0.0f);
    DO_TEST(HALFSPACE_IN_PLANE);

    testPt.SetX(1.0f);
    DO_TEST(HALFSPACE_IN_PLANE);

    testPt.SetX(-1.0f);
    DO_TEST(HALFSPACE_IN_PLANE);

    testPt.SetY(1.0f);
    DO_TEST(HALFSPACE_POSITIVE);

    testPt.SetY(-1.0f);
    DO_TEST(HALFSPACE_NEGATIVE);


    normal.Set(0.5f, 0.5f, 0.0f);

    testPt.Set(0.0f, 0.0f, 0.0f);
    DO_TEST(HALFSPACE_IN_PLANE);

    testPt.Set(0.5f, -0.5f, 0.0f);
    DO_TEST(HALFSPACE_IN_PLANE);

    testPt.Set(-0.5f, 0.5f, 0.0f);
    DO_TEST(HALFSPACE_IN_PLANE);

    testPt.Set(1.0f, 1.0f, 0.0f);
    DO_TEST(HALFSPACE_POSITIVE);

    testPt.Set(-1.0f, -1.0f, 0.0f);
    DO_TEST(HALFSPACE_NEGATIVE);

    testPt.Set(1.0f, 0.0f, 0.0f);
    DO_TEST(HALFSPACE_POSITIVE);

    testPt.Set(-1.0f, 0.0f, 0.0f);
    DO_TEST(HALFSPACE_NEGATIVE);


    planePt.Set(1.0f, 0.0f, 0.0f);
    normal.Set(1.0f, 0.0f, 0.0f);

    testPt.Set(0.0f, 0.0f, 0.0f);
    DO_TEST(HALFSPACE_NEGATIVE);

    testPt.SetY(1.0f);
    DO_TEST(HALFSPACE_NEGATIVE);

    testPt.SetY(-1.0f);
    DO_TEST(HALFSPACE_NEGATIVE);

    testPt.SetX(1.0f);
    DO_TEST(HALFSPACE_IN_PLANE);

    testPt.SetX(-1.0f);
    DO_TEST(HALFSPACE_NEGATIVE);

    testPt.SetX(2.0f);
    DO_TEST(HALFSPACE_POSITIVE);


    // True 3D tests
    planePt.Set(0.0f, 0.0f, 0.0f);
    normal.Set(1.0f, 1.0f, 1.0f);

    testPt.Set(0.0f, 0.0f, 0.0f);
    DO_TEST(HALFSPACE_IN_PLANE);

    testPt.Set(1.0f, 1.0f, 1.0f);
    DO_TEST(HALFSPACE_POSITIVE);

    testPt.Set(1.0f, 0.0f, 0.0f);
    DO_TEST(HALFSPACE_POSITIVE);

    testPt.Set(0.0f, 1.0f, 0.0f);
    DO_TEST(HALFSPACE_POSITIVE);

    testPt.Set(0.0f, 0.0f, 1.0f);
    DO_TEST(HALFSPACE_POSITIVE);

    testPt.Set(-1.0f, 0.0f, 0.0f);
    DO_TEST(HALFSPACE_NEGATIVE);

    testPt.Set(0.0f, -1.0f, 0.0f);
    DO_TEST(HALFSPACE_NEGATIVE);

    testPt.Set(0.0f, 0.0f, -1.0f);
    DO_TEST(HALFSPACE_NEGATIVE);

    testPt.Set(1.0f, 0.0f, -1.0f);
    DO_TEST(HALFSPACE_IN_PLANE);

    testPt.Set(1.0f, -1.0f, 0.0f);
    DO_TEST(HALFSPACE_IN_PLANE);

    testPt.Set(0.0f, -10.0f, 10.0f);
    DO_TEST(HALFSPACE_IN_PLANE);

    testPt.Set(10.0f, -10.0f, 0.0f);
    DO_TEST(HALFSPACE_IN_PLANE);
}


void TestPoint(void) {
    ::TestPoint2D();
    ::TestHalfspace2D();
    ::TestPoint3D();
    ::TestHalfspace3D();
}
