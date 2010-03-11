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
    vislib::math::Polynom<float, 3> poly1;

    AssertEqual("Coefficient a0 = 0", poly1[0], 0.0f);
    AssertEqual("Coefficient a1 = 0", poly1[1], 0.0f);
    AssertEqual("Coefficient a2 = 0", poly1[2], 0.0f);
    AssertTrue("Polynom is zero", poly1.IsZero());
    poly1[0] = 2.0f;
    AssertEqual("Coefficient a0 = 2", poly1[0], 2.0f);
    AssertFalse("Polynom is not zero", poly1.IsZero());
    poly1[1] = -4.0f;
    AssertEqual("Coefficient a1 = -4", poly1[1], -4.0f);
    AssertFalse("Polynom is not zero", poly1.IsZero());
    poly1[2] = 0.5f;
    AssertEqual("Coefficient a2 = 0.5", poly1[2], 0.5f);
    AssertFalse("Polynom is not zero", poly1.IsZero());

    AssertEqual("p(0) = 2", poly1(0.0f), 2.0f);
    AssertEqual("p(-1) = 6.6", poly1(-1.0f), 6.5f);
    AssertEqual("p(2) = -4", poly1(2.0f), -4.0f);

    //float roots[3];
    //unsigned int rootsCnt = poly1.FindRoots(roots, 3);

}

