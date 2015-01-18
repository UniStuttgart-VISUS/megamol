/*
 * testfrustum.h
 *
 * Copyright (C) 2006 - 2009 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#include "testfrustum.h"

#include "testhelper.h"
#include "vislib/RectangularPyramidalFrustum.h"
#define _USE_MATH_DEFINES
#include <cmath>


void TestWorldSpaceFrustum(void) {
    using namespace vislib::math;
    Plane<float> plane1, plane2;

    
    RectangularPyramidalFrustum<float> vonJedemDreck;

    AssertTrue("Apex vonJedemDreck", vonJedemDreck.GetApex() == Point<float, 3>());
    AssertTrue("Default normal", vonJedemDreck.GetBaseNormal() == Vector<float, 3>(0, 0 ,1));
    AssertTrue("Default up", vonJedemDreck.GetBaseUp() == Vector<float, 3>(0, 1 ,0));

    plane1 = vonJedemDreck.GetBottomBase();
    plane2 = vonJedemDreck.GetTopBase();
    AssertTrue("Bottom base normal", plane1.Normal() == vonJedemDreck.GetBaseNormal());
    AssertTrue("Top base normal", plane2.Normal() == -1.0f * vonJedemDreck.GetBaseNormal());

    vonJedemDreck.Set(
        Point<float, 3>(0, 0, 0),   // apex
        Vector<float, 3>(0, 0, 1),  // base normal = view
        Vector<float, 3>(0, 1, 0),  // base up
        10.0f, 5.0f, 1.0f, 10.0f);
    AssertFalse("Apex is not within frustum", vonJedemDreck.Contains(Point<float, 3>(0, 0, 0)));
    AssertTrue("Point on top base", vonJedemDreck.Contains(Point<float, 3>(0, 0, 1), true));
    AssertFalse("onIsIn", vonJedemDreck.Contains(Point<float, 3>(0, 0, 1), false));
    AssertTrue("Point between bases", vonJedemDreck.Contains(Point<float, 3>(0, 0, 6), true));
    AssertTrue("onIsIn", vonJedemDreck.Contains(Point<float, 3>(0, 0, 6), false));

    //Plane<float> plane1;
    //Plane<float> plane2;
    //Point<float, 3> pt1;

    //WorldSpaceFrustum<float> f1(
    //    -10.0f, 10.0f, 
    //    -5.0f, 5.0f, 
    //    1.0f, 10.0f,
    //    1.0f, 1.0f, 1.0f, 
    //    0.0f, 0.0f, 1.0f);

    //AssertEqual("Left initialised", f1.GetLeft(), -10.0f);
    //AssertEqual("Right initialised", f1.GetRight(), 10.0f);
    //AssertEqual("Bottom initialised", f1.GetBottom(), -5.0f);
    //AssertEqual("Top initialised", f1.GetTop(), 5.0f);
    //AssertEqual("Near initialised", f1.GetNear(), 1.0f);
    //AssertEqual("Far initialised", f1.GetFar(), 10.0f);
    //AssertTrue("Eye position initialised", f1.GetEyePosition() == Point<float, 3>(1.0f, 1.0f, 1.0f));
    //AssertTrue("View vector initialised", f1.GetViewVector() == Vector<float, 3>(0.0f, 0.0f, 1.0f));

    //plane1 = f1.GetNearClippingPlane();
    //plane2 = f1.GetFarClippingPlane();
    //AssertTrue("Near clipping plane normal vector", plane1.Normal() == Vector<float, 3>(0.0f, 0.0f, -1.0f));
    //AssertTrue("Near and far clipping plane normals are in same direction", 
    //    plane1.Normal().Dot(plane2.Normal()) == (float) (M_PI));
    //AssertTrue("Near clipping plane contains expected point", plane1.Contains(Point<float, 3>(1, 1, 2)));
    //AssertTrue("Far clipping plane contains expected point", plane2.Contains(Point<float, 3>(1, 1, 11)));

    //pt1.Set(1, 1, 4);
    //f1.Contains(pt1);

}

void TestFrustum(void) {
    ::TestWorldSpaceFrustum();
}
