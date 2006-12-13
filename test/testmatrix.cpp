/*
 * testmatrix.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testmatrix.h"

#include "testhelper.h"

#include "vislib/Quaternion.h"
#include "vislib/mathfunctions.h"
#include "vislib/Matrix.h"
#include "vislib/Matrix4.h"


void TestMatrix(void) {
    using namespace std;
    using namespace vislib;
    using namespace vislib::math;
    
    Matrix<double, 4, COLUMN_MAJOR> m1;
    Matrix<double, 4, ROW_MAJOR> m2;
    Matrix<double, 4, COLUMN_MAJOR> m3;
    Matrix<double, 4, ROW_MAJOR> m4;
    Matrix4<double, COLUMN_MAJOR> m5;


    ::AssertTrue("Default ctor creates id matrix.", m1.IsIdentity());
    ::AssertTrue("Default ctor creates id matrix.", m2.IsIdentity());
    ::AssertFalse("Default ctor creates no null matrix.", m1.IsNull());

    ::AssertTrue("Compare differently layouted matrices.", m1 == m2);

    ::AssertTrue("Compare Matrix<4> and Matrix4.", m1 == m5);

    ::AssertEqual("GetAt 0, 0", m1.GetAt(0, 0), 1.0);
    ::AssertEqual("Function call get at 0, 0", m1(0, 0), 1.0);
    ::AssertEqual("GetAt 1, 0", m1.GetAt(1, 0), 0.0);
    ::AssertEqual("Function call get at 1, 0", m1(1, 0), 0.0);

    m1.SetAt(0, 0, 0);
    ::AssertFalse("Id matrix destroyed.", m1.IsIdentity());

    ::AssertFalse("Compare differently layouted matrices.", m1 == m2);
    ::AssertTrue("Compare differently layouted matrices.", m1 != m2);

    m1.SetAt(0, 0, 1);
    ::AssertTrue("Id matrix restored.", m1.IsIdentity());

    m1.SetAt(0, 0, 0);
    ::AssertEqual("SetAt 0, 0", m1.GetAt(0, 0), 0.0);
    m1.SetAt(0, 1, 2.0);
    ::AssertEqual("SetAt 0, 1", m1.GetAt(0, 1), 2.0);
    m1.Transpose();
    ::AssertEqual("Transposed 0, 0", m1.GetAt(0, 0), 0.0);
    ::AssertNotEqual("Transposed 0, 1", m1.GetAt(0, 1), 2.0);
    ::AssertEqual("Transposed 1, 0", m1.GetAt(1, 0), 2.0);

    m1.SetIdentity();
    ::AssertTrue("Id matrix restored.", m1.IsIdentity());
    m1.Invert();
    ::AssertTrue("Inverting identity.", m1.IsIdentity());

    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            m1.SetAt(r, c, static_cast<float>(c));
        }
    }
    m3 = m1;
    ::AssertFalse("Inverting uninvertable matrix.", m1.Invert());
    ::AssertEqual("Univertable matrix has determinant 0.", m1.Determinant(), 0.0);
    ::AssertEqual("Matrix unchanged after invert failed.", m1, m3);

    m1.SetAt(0, 0, 1.0);
    m1.SetAt(1, 0, 2.0);
    m1.SetAt(2, 0, 3.0);
    m1.SetAt(3, 0, 4.0);

    m1.SetAt(0, 1, 4.0);
    m1.SetAt(1, 1, 1.0);
    m1.SetAt(2, 1, 2.0);
    m1.SetAt(3, 1, 3.0);

    m1.SetAt(0, 2, 3.0);
    m1.SetAt(1, 2, 4.0);
    m1.SetAt(2, 2, 1.0);
    m1.SetAt(3, 2, 2.0);

    m1.SetAt(0, 3, 2.0);
    m1.SetAt(1, 3, 3.0);
    m1.SetAt(2, 3, 4.0);
    m1.SetAt(3, 3, 1.0);

    m3 = m1;
    ::AssertEqual("Assignment.", m1, m3);

    ::AssertNearlyEqual("Determinant", m3.Determinant(), -160.0);

    m2 = m1;
    ::AssertTrue("Converstion assignment.", m1 == m2);

    m4 = m2;
    ::AssertEqual("Assignment.", m2, m4);

    ::AssertEqual("m1 @ 0, 0", m1.GetAt(0, 0), 1.0);
    ::AssertEqual("m1 @ 1, 0", m1.GetAt(1, 0), 2.0);
    ::AssertEqual("m1 @ 2, 0", m1.GetAt(2, 0), 3.0);
    ::AssertEqual("m1 @ 3, 0", m1.GetAt(3, 0), 4.0);

    ::AssertEqual("m1 @ 0, 1", m1.GetAt(0, 1), 4.0);
    ::AssertEqual("m1 @ 1, 1", m1.GetAt(1, 1), 1.0);
    ::AssertEqual("m1 @ 2, 1", m1.GetAt(2, 1), 2.0);
    ::AssertEqual("m1 @ 3, 1", m1.GetAt(3, 1), 3.0);

    ::AssertEqual("m1 @ 0, 2", m1.GetAt(0, 2), 3.0);
    ::AssertEqual("m1 @ 1, 2", m1.GetAt(1, 2), 4.0);
    ::AssertEqual("m1 @ 2, 2", m1.GetAt(2, 2), 1.0);
    ::AssertEqual("m1 @ 3, 2", m1.GetAt(3, 2), 2.0);

    ::AssertEqual("m1 @ 0, 3", m1.GetAt(0, 3), 2.0);
    ::AssertEqual("m1 @ 1, 3", m1.GetAt(1, 3), 3.0);
    ::AssertEqual("m1 @ 2, 3", m1.GetAt(2, 3), 4.0);
    ::AssertEqual("m1 @ 3, 3", m1.GetAt(3, 3), 1.0);

    ::AssertEqual("m1 @ 0, 0 == @ 0", m1.GetAt(0, 0), m1.PeekComponents()[0 + 0 * 4]);
    ::AssertEqual("m1 @ 1, 0 == @ 1", m1.GetAt(1, 0), m1.PeekComponents()[1 + 0 * 4]);
    ::AssertEqual("m1 @ 2, 0 == @ 2", m1.GetAt(2, 0), m1.PeekComponents()[2 + 0 * 4]);
    ::AssertEqual("m1 @ 3, 0 == @ 3", m1.GetAt(3, 0), m1.PeekComponents()[3 + 0 * 4]);

    ::AssertEqual("m1 @ 0, 1 == @ 4", m1.GetAt(0, 1), m1.PeekComponents()[0 + 1 * 4]);
    ::AssertEqual("m1 @ 1, 1 == @ 5", m1.GetAt(1, 1), m1.PeekComponents()[1 + 1 * 4]);
    ::AssertEqual("m1 @ 2, 1 == @ 6", m1.GetAt(2, 1), m1.PeekComponents()[2 + 1 * 4]);
    ::AssertEqual("m1 @ 3, 1 == @ 7", m1.GetAt(3, 1), m1.PeekComponents()[3 + 1 * 4]);

    ::AssertEqual("m1 @ 0, 2 == @ 8", m1.GetAt(0, 2), m1.PeekComponents()[0 + 2 * 4]);
    ::AssertEqual("m1 @ 1, 2 == @ 9", m1.GetAt(1, 2), m1.PeekComponents()[1 + 2 * 4]);
    ::AssertEqual("m1 @ 2, 2 == @ 10", m1.GetAt(2, 2), m1.PeekComponents()[2 + 2 * 4]);
    ::AssertEqual("m1 @ 3, 2 == @ 11", m1.GetAt(3, 2), m1.PeekComponents()[3 + 2 * 4]);

    ::AssertEqual("m1 @ 0, 3 == @ 12", m1.GetAt(0, 3), m1.PeekComponents()[0 + 3 * 4]);
    ::AssertEqual("m1 @ 1, 3 == @ 13", m1.GetAt(1, 3), m1.PeekComponents()[1 + 3 * 4]);
    ::AssertEqual("m1 @ 2, 3 == @ 14", m1.GetAt(2, 3), m1.PeekComponents()[2 + 3 * 4]);
    ::AssertEqual("m1 @ 3, 3 == @ 15", m1.GetAt(3, 3), m1.PeekComponents()[3 + 3 * 4]);


    ::AssertEqual("m1 @ 0, 0 == @ 0", m2.GetAt(0, 0), m2.PeekComponents()[0 * 4 + 0]);
    ::AssertEqual("m1 @ 1, 0 == @ 4", m2.GetAt(1, 0), m2.PeekComponents()[1 * 4 + 0]);
    ::AssertEqual("m1 @ 2, 0 == @ 8", m2.GetAt(2, 0), m2.PeekComponents()[2 * 4 + 0]);
    ::AssertEqual("m1 @ 3, 0 == @ 12", m2.GetAt(3, 0), m2.PeekComponents()[3 * 4 + 0]);

    ::AssertEqual("m1 @ 0, 1 == @ 1", m2.GetAt(0, 1), m2.PeekComponents()[0 * 4 + 1]);
    ::AssertEqual("m1 @ 1, 1 == @ 5", m2.GetAt(1, 1), m2.PeekComponents()[1 * 4 + 1]);
    ::AssertEqual("m1 @ 2, 1 == @ 9", m2.GetAt(2, 1), m2.PeekComponents()[2 * 4 + 1]);
    ::AssertEqual("m1 @ 3, 1 == @ 13", m2.GetAt(3, 1), m2.PeekComponents()[3 * 4 + 1]);

    ::AssertEqual("m1 @ 0, 2 == @ 2", m2.GetAt(0, 2), m2.PeekComponents()[0 * 4 + 2]);
    ::AssertEqual("m1 @ 1, 2 == @ 6", m2.GetAt(1, 2), m2.PeekComponents()[1 * 4 + 2]);
    ::AssertEqual("m1 @ 2, 2 == @ 10", m2.GetAt(2, 2), m2.PeekComponents()[2 * 4 + 2]);
    ::AssertEqual("m1 @ 3, 2 == @ 14", m2.GetAt(3, 2), m2.PeekComponents()[3 * 4 + 2]);

    ::AssertEqual("m1 @ 0, 3 == @ 3", m2.GetAt(0, 3), m2.PeekComponents()[0 * 4 + 3]);
    ::AssertEqual("m1 @ 1, 3 == @ 7", m2.GetAt(1, 3), m2.PeekComponents()[1 * 4 + 3]);
    ::AssertEqual("m1 @ 2, 3 == @ 11", m2.GetAt(2, 3), m2.PeekComponents()[2 * 4 + 3]);
    ::AssertEqual("m1 @ 3, 3 == @ 15", m2.GetAt(3, 3), m2.PeekComponents()[3 * 4 + 3]);


    ::AssertTrue("Invert matrix.", m1.Invert());
    ::AssertNearlyEqual("Inverted @ 0, 0.", m1.GetAt(0, 0), -9.0 / 40.0);
    ::AssertNearlyEqual("Inverted @ 1, 0.", m1.GetAt(1, 0), 11.0 / 40.0);
    ::AssertNearlyEqual("Inverted @ 2, 0.", m1.GetAt(2, 0), 1.0 / 40.0);
    ::AssertNearlyEqual("Inverted @ 3, 0.", m1.GetAt(3, 0), 1.0 / 40.0);

    ::AssertNearlyEqual("Inverted @ 0, 1.", m1.GetAt(0, 1), 1.0 / 40.0);
    ::AssertNearlyEqual("Inverted @ 1, 1.", m1.GetAt(1, 1), -9.0 / 40.0);
    ::AssertNearlyEqual("Inverted @ 2, 1.", m1.GetAt(2, 1), 11.0 / 40.0);
    ::AssertNearlyEqual("Inverted @ 3, 1.", m1.GetAt(3, 1), 1.0 / 40.0);

    ::AssertNearlyEqual("Inverted @ 0, 2.", m1.GetAt(0, 2), 1.0 / 40.0);
    ::AssertNearlyEqual("Inverted @ 1, 2.", m1.GetAt(1, 2), 1.0 / 40.0);
    ::AssertNearlyEqual("Inverted @ 2, 2.", m1.GetAt(2, 2), -9.0 / 40.0);
    ::AssertNearlyEqual("Inverted @ 3, 2.", m1.GetAt(3, 2), 11.0 / 40.0);

    ::AssertNearlyEqual("Inverted @ 0, 3.", m1.GetAt(0, 3), 11.0 / 40.0);
    ::AssertNearlyEqual("Inverted @ 1, 3.", m1.GetAt(1, 3), 1.0 / 40.0);
    ::AssertNearlyEqual("Inverted @ 2, 3.", m1.GetAt(2, 3), 1.0 / 40.0);
    ::AssertNearlyEqual("Inverted @ 3, 3.", m1.GetAt(3, 3), -9.0 / 40.0);

    ::AssertTrue("Invert matrix.", m2.Invert());
    ::AssertNearlyEqual("Inverted @ 0, 0.", m2.GetAt(0, 0), -9.0 / 40.0);
    ::AssertNearlyEqual("Inverted @ 1, 0.", m2.GetAt(1, 0), 11.0 / 40.0);
    ::AssertNearlyEqual("Inverted @ 2, 0.", m2.GetAt(2, 0), 1.0 / 40.0);
    ::AssertNearlyEqual("Inverted @ 3, 0.", m2.GetAt(3, 0), 1.0 / 40.0);

    ::AssertNearlyEqual("Inverted @ 0, 1.", m2.GetAt(0, 1), 1.0 / 40.0);
    ::AssertNearlyEqual("Inverted @ 1, 1.", m2.GetAt(1, 1), -9.0 / 40.0);
    ::AssertNearlyEqual("Inverted @ 2, 1.", m2.GetAt(2, 1), 11.0 / 40.0);
    ::AssertNearlyEqual("Inverted @ 3, 1.", m2.GetAt(3, 1), 1.0 / 40.0);

    ::AssertNearlyEqual("Inverted @ 0, 2.", m2.GetAt(0, 2), 1.0 / 40.0);
    ::AssertNearlyEqual("Inverted @ 1, 2.", m2.GetAt(1, 2), 1.0 / 40.0);
    ::AssertNearlyEqual("Inverted @ 2, 2.", m2.GetAt(2, 2), -9.0 / 40.0);
    ::AssertNearlyEqual("Inverted @ 3, 2.", m2.GetAt(3, 2), 11.0 / 40.0);

    ::AssertNearlyEqual("Inverted @ 0, 3.", m2.GetAt(0, 3), 11.0 / 40.0);
    ::AssertNearlyEqual("Inverted @ 1, 3.", m2.GetAt(1, 3), 1.0 / 40.0);
    ::AssertNearlyEqual("Inverted @ 2, 3.", m2.GetAt(2, 3), 1.0 / 40.0);
    ::AssertNearlyEqual("Inverted @ 3, 3.", m2.GetAt(3, 3), -9.0 / 40.0);

    m3 *= m1;
    ::AssertTrue("m * m^-1 = id", m3.IsIdentity());
    
    m4 = m4 * m2;
    ::AssertTrue("m * m^-1 = id", m4.IsIdentity());


    Vector<double, 3> axis(1.0, 0.0, 0.0);
    Quaternion<double> q1(1.0, axis);
    Matrix4<double, COLUMN_MAJOR> rm1(q1);

    ::AssertNearlyEqual("Rotation from quaterion @ 0, 0.", rm1.GetAt(0, 0), 1.0);
    ::AssertNearlyEqual("Rotation from quaterion @ 1, 0.", rm1.GetAt(1, 0), 0.0);
    ::AssertNearlyEqual("Rotation from quaterion @ 2, 0.", rm1.GetAt(2, 0), 0.0);
    ::AssertNearlyEqual("Rotation from quaterion @ 3, 0.", rm1.GetAt(3, 0), 0.0);

    ::AssertNearlyEqual("Rotation from quaterion @ 0, 1.", rm1.GetAt(0, 1), 0.0);
    ::AssertNearlyEqual<double>("Rotation from quaterion @ 1, 1.", rm1.GetAt(1, 1), cos(1.0));
    ::AssertNearlyEqual<double>("Rotation from quaterion @ 2, 1.", rm1.GetAt(2, 1), sin(1.0));
    ::AssertNearlyEqual("Rotation from quaterion @ 3, 1.", rm1.GetAt(3, 1), 0.0);

    ::AssertNearlyEqual("Rotation from quaterion @ 0, 2.", rm1.GetAt(0, 2), 0.0);
    ::AssertNearlyEqual<double>("Rotation from quaterion @ 1, 2.", rm1.GetAt(1, 2), -sin(1.0));
    ::AssertNearlyEqual<double>("Rotation from quaterion @ 2, 2.", rm1.GetAt(2, 2), cos(1.0));
    ::AssertNearlyEqual("Rotation from quaterion @ 3, 2.", rm1.GetAt(3, 2), 0.0);

    ::AssertNearlyEqual("Rotation from quaterion @ 0, 3.", rm1.GetAt(0, 3), 0.0);
    ::AssertNearlyEqual("Rotation from quaterion @ 1, 3.", rm1.GetAt(1, 3), 0.0);
    ::AssertNearlyEqual("Rotation from quaterion @ 2, 3.", rm1.GetAt(2, 3), 0.0);
    ::AssertNearlyEqual("Rotation from quaterion @ 3, 3.", rm1.GetAt(3, 3), 1.0);

}
