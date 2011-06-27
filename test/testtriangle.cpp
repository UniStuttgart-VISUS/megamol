/*
 * testtriangle.cpp 
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testhelper.h"

#include "vislib/ShallowShallowTriangle.h"
#include "vislib/ShallowTriangle.h"
#include "vislib/ShallowPoint.h"
#include "vislib/Triangle.h"
#include "vislib/Point.h"
#include "vislib/Vector.h"

void TestTriangle(void) {
	using namespace vislib::math;
    Point<float, 3> thePoints[3];
    ShallowPoint<float, 3> sp(thePoints[0].PeekCoordinates());
    float pointmem[18];
	Triangle<Point<float, 3> > t;
	Triangle<Point<float, 3> > t2;
	Vector<float, 3> v;
    ShallowTriangle<Point<float,3> > st(thePoints);
    ShallowShallowTriangle<float, 3> sst(pointmem);
    ShallowShallowTriangle<float, 3> sst2(pointmem);

	t[0].Set(1.0f, 0.0f, 0.0f);
	t[1].Set(0.0f, 1.0f, 0.0f);
	t[2].Set(-1.0f, 0.0f, 0.0f);

    //sp.GetX();
    //sp.Set(1.0f, 0.0f, 0.0f);
    //st = t;
    //t = st;
    //sst = t;

    AssertNotEqual("just-initialized triangle t2 is not equal to t", t, t2);
    AssertEqual("t and t2 have no common vertex", t.CountCommonVertices(t2), 0U);
    t2 = t;
    AssertEqual("after assignment triangles t and t2 are equal", t, t2);
    sst = t2;
    AssertTrue("after assignment triangles t and sst are equal", t == sst);
    t2[2].Set(3.0f, 1.0f, 0.0f);
    AssertNotEqual("after moving vertex 2 of t2, t and t2 are no longer equal", t, t2);
    AssertTrue("after moving vertex 2 of t2, t and sst are still equal", t == sst);
    AssertTrue("since they point to the same memory, triangles sst and sst2 are equal", sst == sst2);
    sst2.SetPointer(pointmem + 9);
    AssertFalse("after moving the content pointer, triangles sst and sst2 are no longer equal", sst == sst2);

    sst2[0].Set(1.0f, 0.0f, 0.0f);
    sst2[1].Set(0.0f, 1.0f, 0.0f);
    sst2[2].Set(-1.0f, 0.0f, 0.0f);
    AssertEqual("after manually setting sst2's components, triangles t and sst2 share 3 vertices",
        t.CountCommonVertices(sst2), 3U);

    AssertEqual("triangles t and t2 share 2 vertices", t.CountCommonVertices(t2), 2U);
    AssertEqual("triangles t and t2 thus have a common edge", t.HasCommonEdge(t2), true);

	AssertNearlyEqual("area is numerically correct", t.Area<float>(), 1.0f);
	AssertNearlyEqual("circumference is numerically correct", t.Circumference<float>(), 
		2.0f + 2.0f * vislib::math::Sqrt(2.0f));
	t.Normal(v);
	AssertNearlyEqual("normal vector is numerically correct", v, Vector<float, 3>(0.0f, 0.0f, 1.0f));
}
