/*
 * testpolynom.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "testpolynom.h"
#include "vislib/Polynom.h"
#include "testhelper.h"


/*
 * TestPolynom
 */
void TestPolynom(void) {
    vislib::math::Polynom<float, 3> poly3;

    AssertEqual("Coefficient a0 = 0", poly3[0], 0.0f);
    AssertEqual("Coefficient a1 = 0", poly3[1], 0.0f);
    AssertEqual("Coefficient a2 = 0", poly3[2], 0.0f);
    AssertTrue("Polynom is zero", poly3.IsZero());
    AssertEqual("Effective degree is 0", poly3.EffectiveDegree(), 0U);
    poly3[0] = 2.0f;
    AssertEqual("Coefficient a0 = 2", poly3[0], 2.0f);
    AssertEqual("Effective degree is 0", poly3.EffectiveDegree(), 0U);
    AssertFalse("Polynom is not zero", poly3.IsZero());
    poly3[1] = -4.0f;
    AssertEqual("Coefficient a1 = -4", poly3[1], -4.0f);
    AssertEqual("Effective degree is 1", poly3.EffectiveDegree(), 1U);
    AssertFalse("Polynom is not zero", poly3.IsZero());
    poly3[2] = 0.5f;
    AssertEqual("Coefficient a2 = 0.5", poly3[2], 0.5f);
    AssertEqual("Effective degree is 2", poly3.EffectiveDegree(), 2U);
    AssertFalse("Polynom is not zero", poly3.IsZero());
    poly3[3] = -1.0f;
    AssertEqual("Coefficient a3 = -1", poly3[3], -1.0f);
    AssertEqual("Effective degree is 3", poly3.EffectiveDegree(), 3U);
    AssertFalse("Polynom is not zero", poly3.IsZero());

    // - x^3 + 0.5 * x^2 - 4 * x + 2
    AssertEqual("p(0) = 2", poly3(0.0f), 2.0f);
    // 1 + 0.5 + 4 + 2
    AssertEqual("p(-1) = 7.5", poly3(-1.0f), 7.5f);
    // -8 + 2 - 8 + 2
    AssertEqual("p(2) = -12", poly3(2.0f), -12.0f);

    vislib::math::Polynom<float, 2> poly2 = poly3.Derivative();
    AssertEqual("Effective degree of derivative is 2", poly2.EffectiveDegree(), 2U);
    AssertEqual("Coefficient a'0 = -4", poly2[0], -4.0f);
    AssertEqual("Coefficient a'1 = 1", poly2[1], 1.0f);
    AssertEqual("Coefficient a'2 = -3", poly2[2], -3.0f);

    vislib::math::Polynom<float, 1> poly1 = poly2.Derivative();
    AssertEqual("Effective degree of second derivative is 1", poly1.EffectiveDegree(), 1U);
    AssertEqual("Coefficient a''0 = 1", poly1[0], 1.0f);
    AssertEqual("Coefficient a''1 = -6", poly1[1], -6.0f);

    //vislib::math::Polynom<float, 0> poly0 = poly1.Derivative(); // possible problem under linux
    //AssertEqual("Effective degree of third derivative is 0", poly0.EffectiveDegree(), 0U);
    //AssertEqual("Coefficient a'''0 = -6", poly0[0], -6.0f);
    //poly0.Derivative(); // problem under windows

    vislib::math::Polynom<double, 3> poly3d(poly3);
    vislib::math::Polynom<float, 3> poly3s(poly3);

    AssertEqual("Coefficient a0 = 2", poly3s[0], 2.0f);
    AssertEqual("Coefficient a1 = -4", poly3s[1], -4.0f);
    AssertEqual("Coefficient a2 = 0.5", poly3s[2], 0.5f);
    AssertEqual("Coefficient a3 = -1", poly3s[3], -1.0f);

    AssertEqual("Coefficient a0 = 2", poly3d[0], 2.0);
    AssertEqual("Coefficient a1 = -4", poly3d[1], -4.0);
    AssertEqual("Coefficient a2 = 0.5", poly3d[2], 0.5);
    AssertEqual("Coefficient a3 = -1", poly3d[3], -1.0);

    poly3s = poly3;
    poly3d = poly3s;

    AssertEqual("Coefficient a0 = 2", poly3s[0], 2.0f);
    AssertEqual("Coefficient a1 = -4", poly3s[1], -4.0f);
    AssertEqual("Coefficient a2 = 0.5", poly3s[2], 0.5f);
    AssertEqual("Coefficient a3 = -1", poly3s[3], -1.0f);

    AssertEqual("Coefficient a0 = 2", poly3d[0], 2.0);
    AssertEqual("Coefficient a1 = -4", poly3d[1], -4.0);
    AssertEqual("Coefficient a2 = 0.5", poly3d[2], 0.5);
    AssertEqual("Coefficient a3 = -1", poly3d[3], -1.0);

    AssertEqual("Assigment and comparison works", poly3, poly3s);
    AssertTrue("Assigment and comparison works", poly3 == poly3d);

    poly3d = poly1;
    AssertEqual("Effective degree of second derivative is 1", poly3d.EffectiveDegree(), 1U);
    AssertTrue("Assigment and comparison works", poly1 == poly3d);

    float roots[3];
    unsigned int rootsCnt = poly3.FindRoots(roots, 3);

}

