/*
 * dimandrect.cpp
 *
 * Copyright (C) 2006 by Universitaet Stuttgart (VIS). Alle Rechte vorbehalten.
 */

#include "testdimandrect.h"
#include "testhelper.h"

#include "vislib/Dimension.h"
#include "vislib/Rectangle.h"
#include "vislib/ShallowRectangle.h"


void TestDimension2D(void) {
    typedef vislib::math::Dimension<float, 2> Dimension2DF;
    typedef vislib::math::Dimension<int, 2> Dimension2DI;

    std::cout << std::endl << "Dimension2D ..." << std::endl;

    Dimension2DF fdim1;
    Dimension2DF fdim2(1.0f, 2.0f);
    Dimension2DF fdim3(fdim2);
    Dimension2DI idim1(fdim2);

    ::AssertEqual("Zero dimension ctor.", fdim1[0], 0.0f);
    ::AssertEqual("Zero dimension ctor.", fdim1[1], 0.0f);
    ::AssertEqual("GetWidth().", fdim1.GetWidth(), 0.0f);
    ::AssertEqual("GetHeight().", fdim1.GetHeight(), 0.0f);
    ::AssertEqual("Width/height ctor.", fdim2[0], 1.0f);
    ::AssertEqual("Width/height ctor.", fdim2[1], 2.0f);
    ::AssertEqual("GetWidth().", fdim2.GetWidth(), 1.0f);
    ::AssertEqual("GetHeight().", fdim2.GetHeight(), 2.0f);
    ::AssertEqual("Width().", fdim2.Width(), 1.0f);
    ::AssertEqual("Height().", fdim2.Height(), 2.0f);
    ::AssertEqual("Copy ctor.", fdim3[0], fdim2[0]);
    ::AssertEqual("Copy ctor.", fdim3[1], fdim2[1]);
    ::AssertEqual("Conversion copy ctor.", idim1[0], int(fdim2[0]));
    ::AssertEqual("Conversion copy ctor.", idim1[1], int(fdim2[1]));

    fdim1 = fdim2;
    ::AssertEqual("Assignment.", fdim1[0], fdim2[0]);
    ::AssertEqual("Assignment.", fdim1[1], fdim2[1]);

    fdim1.SetWidth(10.0);
    ::AssertEqual("SetWidth", fdim1[0], 10.0f);
    fdim1.SetHeight(11.0);
    ::AssertEqual("SetHeight", fdim1[1], 11.0f);

    idim1 = fdim1;
    ::AssertEqual("Conversion assignment.", idim1[0], int(fdim1[0]));
    ::AssertEqual("Conversion assignment.", idim1[1], int(fdim1[1]));

    ::AssertTrue("Equality", fdim2 == fdim3);
    ::AssertTrue("Conversion equality", idim1 == fdim1);

    ::AssertFalse("Inequality", fdim2 != fdim3);
    ::AssertFalse("Conversion inequality", idim1 != fdim1);

}


void TestDimension3D(void) {
    typedef vislib::math::Dimension<float, 3> Dimension3DF;
    typedef vislib::math::Dimension<int, 3> Dimension3DI;

    std::cout << std::endl << "Dimension3D ..." << std::endl;

    Dimension3DF fdim1;
    Dimension3DF fdim2(1.0f, 2.0f, 3.0f);
    Dimension3DF fdim3(fdim2);
    Dimension3DI idim1(fdim2);

    ::AssertEqual("Zero dimension ctor.", fdim1[0], 0.0f);
    ::AssertEqual("Zero dimension ctor.", fdim1[1], 0.0f);
    ::AssertEqual("Zero dimension ctor.", fdim1[2], 0.0f);
    ::AssertEqual("GetWidth().", fdim1.GetWidth(), 0.0f);
    ::AssertEqual("GetHeight().", fdim1.GetHeight(), 0.0f);
    ::AssertEqual("GetDepth().", fdim1.GetDepth(), 0.0f);
    ::AssertEqual("Width/height/depth ctor.", fdim2[0], 1.0f);
    ::AssertEqual("Width/height/depth ctor.", fdim2[1], 2.0f);
    ::AssertEqual("Width/height/depth ctor.", fdim2[2], 3.0f);
    ::AssertEqual("GetWidth().", fdim2.GetWidth(), 1.0f);
    ::AssertEqual("GetHeight().", fdim2.GetHeight(), 2.0f);
    ::AssertEqual("GetDepth().", fdim2.GetDepth(), 3.0f);
    ::AssertEqual("Width().", fdim2.Width(), 1.0f);
    ::AssertEqual("Height().", fdim2.Height(), 2.0f);
    ::AssertEqual("Depth().", fdim2.Depth(), 3.0f);
    ::AssertEqual("Copy ctor.", fdim3[0], fdim2[0]);
    ::AssertEqual("Copy ctor.", fdim3[1], fdim2[1]);
    ::AssertEqual("Copy ctor.", fdim3[2], fdim2[2]);
    ::AssertEqual("Conversion copy ctor.", idim1[0], int(fdim2[0]));
    ::AssertEqual("Conversion copy ctor.", idim1[1], int(fdim2[1]));
    ::AssertEqual("Conversion copy ctor.", idim1[2], int(fdim2[2]));

    fdim1 = fdim2;
    ::AssertEqual("Assignment.", fdim1[0], fdim2[0]);
    ::AssertEqual("Assignment.", fdim1[1], fdim2[1]);
    ::AssertEqual("Assignment.", fdim1[2], fdim2[2]);

    fdim1.SetWidth(10.0);
    ::AssertEqual("SetWidth", fdim1[0], 10.0f);
    fdim1.SetHeight(11.0);
    ::AssertEqual("SetHeight", fdim1[1], 11.0f);
    fdim1.SetDepth(12.0);
    ::AssertEqual("SetDepth", fdim1[2], 12.0f);

    idim1 = fdim1;
    ::AssertEqual("Conversion assignment.", idim1[0], int(fdim1[0]));
    ::AssertEqual("Conversion assignment.", idim1[1], int(fdim1[1]));
    ::AssertEqual("Conversion assignment.", idim1[2], int(fdim1[2]));

    ::AssertTrue("Equality", fdim2 == fdim3);
    ::AssertTrue("Conversion equality", idim1 == fdim1);

    ::AssertFalse("Inequality", fdim2 != fdim3);
    ::AssertFalse("Conversion inequality", idim1 != fdim1);
}


void TestDimension(void) {
    ::TestDimension2D();
    ::TestDimension3D();
}


void TestRectangle(void) {
    typedef vislib::math::Rectangle<float> FloatRectangle;
    typedef vislib::math::Rectangle<int> IntRectangle;
    typedef vislib::math::ShallowRectangle<int> ShallowIntRectangle;
    typedef vislib::math::Point<int, 2> IntPoint;

    FloatRectangle frect1;
    FloatRectangle frect2(1.0, 1.0, 2.0, 2.0);
    FloatRectangle frect3(frect2);
    IntRectangle irect1(frect2);
    int irect2data[4];
    ::memset(irect2data, 0, sizeof(irect2data));
    ShallowIntRectangle irect2(irect2data);
    ShallowIntRectangle irect3(irect2);

    ::AssertEqual("Default ctor GetLeft().", frect1.GetLeft(), 0.0f);
    ::AssertEqual("Default ctor GetBottom().", frect1.GetBottom(), 0.0f);
    ::AssertEqual("Default ctor GetRight().", frect1.GetRight(), 0.0f);
    ::AssertEqual("Default ctor GetTop().", frect1.GetTop(), 0.0f);
    ::AssertEqual("Default ctor Left().", frect1.Left(), 0.0f);
    ::AssertEqual("Default ctor Bottom().", frect1.Bottom(), 0.0f);
    ::AssertEqual("Default ctor Right().", frect1.Right(), 0.0f);
    ::AssertEqual("Default ctor Top().", frect1.Top(), 0.0f);

    ::AssertTrue("IsEmpty", frect1.IsEmpty());

    ::AssertEqual("Init ctor GetLeft().", frect2.GetLeft(), 1.0f);
    ::AssertEqual("Init ctor GetBottom().", frect2.GetBottom(), 1.0f);
    ::AssertEqual("Init ctor GetRight().", frect2.GetRight(), 2.0f);
    ::AssertEqual("Init ctor GetTop().", frect2.GetTop(), 2.0f);
    ::AssertEqual("Init ctor Left().", frect2.Left(), 1.0f);
    ::AssertEqual("Init ctor Bottom().", frect2.Bottom(), 1.0f);
    ::AssertEqual("Init ctor Right().", frect2.Right(), 2.0f);
    ::AssertEqual("Init ctor Top().", frect2.Top(), 2.0f);

    ::AssertEqual("Copy ctor Left().", frect3.Left(), 1.0f);
    ::AssertEqual("Copy ctor Bottom().", frect3.Bottom(), 1.0f);
    ::AssertEqual("Copy ctor Right().", frect3.Right(), 2.0f);
    ::AssertEqual("Copy ctor Top().", frect3.Top(), 2.0f);

    ::AssertTrue("Equality", frect2 == frect3);
    ::AssertFalse("Inequality", frect2 != frect3);

    ::AssertEqual("Conversion ctor Left().", irect1.Left(), int(frect2.Left()));
    ::AssertEqual("Conversion ctor Bottom().", irect1.Bottom(), int(frect2.Bottom()));
    ::AssertEqual("Conversion ctor Right().", irect1.Right(), int(frect2.Right()));
    ::AssertEqual("Conversion ctor Top().", irect1.Top(), int(frect2.Top()));

    //::AssertTrue("Conversion equality", frect2 == irect1);
    //::AssertFalse("Conversion inequality", frect2 != irect1);

    ::AssertEqual("Shallow default ctor GetLeft().", irect2.GetLeft(), 0);
    ::AssertEqual("Shallow default ctor GetBottom().", irect2.GetBottom(), 0);
    ::AssertEqual("Shallow default ctor GetRight().", irect2.GetRight(), 0);
    ::AssertEqual("Shallow default ctor GetTop().", irect2.GetTop(), 0);
    ::AssertEqual("Shallow default ctor Left().", irect2.Left(), 0);
    ::AssertEqual("Shallow default ctor Bottom().", irect2.Bottom(), 0);
    ::AssertEqual("Shallow default ctor Right().", irect2.Right(), 0);
    ::AssertEqual("Shallow default ctor Top().", irect2.Top(), 0);
    
    ::AssertEqual<const int *>("Shallow data pointer.", irect2.PeekBounds(), irect2data);
    ::AssertEqual<const int *>("Shallow aliasing.", irect2.PeekBounds(), irect3.PeekBounds());

    ::AssertTrue("Equality", irect2 == irect3);
    ::AssertFalse("Inequality", irect2 != irect3);

    irect3 = frect3;
    ::AssertEqual("Shallow conversion assignment Left().", irect3.Left(), int(frect3.Left()));
    ::AssertEqual("Shallow conversion assignment Bottom().", irect3.Bottom(), int(frect3.Bottom()));
    ::AssertEqual("Shallow conversion assignment Right().", irect3.Right(), int(frect3.Right()));
    ::AssertEqual("Shallow conversion assignment Top().", irect3.Top(), int(frect3.Top()));

    ::AssertTrue("Shallow assignment aliasing", irect3 == irect2);


    IntRectangle irect4(1, 1, 10, 10);
    ::AssertTrue("Inside on left border.", irect4.Contains(IntPoint(1, 1), IntRectangle::BORDER_ALL));
    ::AssertTrue("Inside on bottom border.", irect4.Contains(IntPoint(1, 1), IntRectangle::BORDER_ALL));
    ::AssertTrue("Inside on right border.", irect4.Contains(IntPoint(10, 10), IntRectangle::BORDER_ALL));
    ::AssertTrue("Inside on top border.", irect4.Contains(IntPoint(10, 10), IntRectangle::BORDER_ALL));

    ::AssertFalse("Outside on left border.", irect4.Contains(IntPoint(1, 1)));
    ::AssertFalse("Outside on bottom border.", irect4.Contains(IntPoint(1, 1)));
    ::AssertFalse("Outside on right border.", irect4.Contains(IntPoint(10, 10)));
    ::AssertFalse("Outside on top border.", irect4.Contains(IntPoint(10, 10)));

    ::AssertFalse("Outside left.", irect4.Contains(IntPoint(0, 1)));
    ::AssertFalse("Outside bottom.", irect4.Contains(IntPoint(1, 0)));
    ::AssertFalse("Outside right.", irect4.Contains(IntPoint(11, 10)));
    ::AssertFalse("Outside top.", irect4.Contains(IntPoint(10, 11)));

    ::AssertFalse("Outside left including border.", irect4.Contains(IntPoint(0, 1), IntRectangle::BORDER_ALL));
    ::AssertFalse("Outside bottom including border.", irect4.Contains(IntPoint(1, 0), IntRectangle::BORDER_ALL));
    ::AssertFalse("Outside right including border.", irect4.Contains(IntPoint(11, 10), IntRectangle::BORDER_ALL));
    ::AssertFalse("Outside top including border.", irect4.Contains(IntPoint(10, 11), IntRectangle::BORDER_ALL));
}

