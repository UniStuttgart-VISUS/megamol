/*
 * BezierControlLines.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#define _USE_MATH_DEFINES
#include "misc/BezierControlLines.h"
#include "misc/BezierDataCall.h"
#include "vislib/BezierCurve.h"
#include "vislib/NamedColours.h"
#include "vislib/ShallowPoint.h"
#include "vislib/Vector.h"

using namespace megamol::core;


/*
 * misc::BezierControlLines::BezierControlLines
 */
misc::BezierControlLines::BezierControlLines(void) : Module(),
        dataSlot("data", "Provides with line data"),
        getDataSlot("getData", "Gets bezier data"),
        hash(0) {

    this->dataSlot.SetCallback(LinesDataCall::ClassName(), "GetData",
        &BezierControlLines::getDataCallback);
    this->dataSlot.SetCallback(LinesDataCall::ClassName(), "GetExtent",
        &BezierControlLines::getExtentCallback);
    this->MakeSlotAvailable(&this->dataSlot);

    this->getDataSlot.SetCompatibleCall<BezierDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

}


/*
 * misc::BezierControlLines::~BezierControlLines
 */
misc::BezierControlLines::~BezierControlLines(void) {
    this->Release();
}


/*
 * misc::BezierControlLines::create
 */
bool misc::BezierControlLines::create(void) {
    // intentionally empty
    return true;
}


/*
 * misc::BezierControlLines::release
 */
void misc::BezierControlLines::release(void) {
    // intentionally empty
}


/*
 * misc::BezierControlLines::getDataCallback
 */
bool misc::BezierControlLines::getDataCallback(Call& call) {
    LinesDataCall *ldc = dynamic_cast<LinesDataCall*>(&call);
    if (ldc == NULL) return false;

    BezierDataCall *bdc = this->getDataSlot.CallAs<BezierDataCall>();
    if ((bdc == NULL) || (!(*bdc)(0))) {
        ldc->SetData(0, NULL);
        ldc->SetDataHash(0);

    } else {
        if ((bdc->DataHash() == 0) || (bdc->DataHash() != this->hash)) {
            this->hash = bdc->DataHash();
            this->vertData[0].EnforceSize(4 * 3 * 4 * bdc->Count());
            this->idxData[0].EnforceSize(2 * 3 * 4 * bdc->Count());

            unsigned int lcnt = 0;
            this->vertData[1].AssertSize(bdc->Count() * 3 * 8 * 3 * 4);
            this->idxData[1].AssertSize(bdc->Count() * 3 * 12 * 2 * 4);

            for (unsigned int i = 0; i < bdc->Count(); i++) {
                const vislib::math::BezierCurve<BezierDataCall::BezierPoint, 3>& curve = bdc->Curves()[i];

                {
                    vislib::math::ShallowPoint<float, 3> cp0(this->vertData[0].AsAt<float>((i * 4 + 0) * 3 * 4));
                    vislib::math::ShallowPoint<float, 3> cp1(this->vertData[0].AsAt<float>((i * 4 + 1) * 3 * 4));
                    vislib::math::ShallowPoint<float, 3> cp2(this->vertData[0].AsAt<float>((i * 4 + 2) * 3 * 4));
                    vislib::math::ShallowPoint<float, 3> cp3(this->vertData[0].AsAt<float>((i * 4 + 3) * 3 * 4));

                    cp0 = curve.ControlPoint(0).Position();
                    cp1 = curve.ControlPoint(1).Position();
                    cp2 = curve.ControlPoint(2).Position();
                    cp3 = curve.ControlPoint(3).Position();
                }

                *this->idxData[0].AsAt<unsigned int>(((i * 3 + 0) * 2 + 0) * 4) = (i * 4 + 0);
                *this->idxData[0].AsAt<unsigned int>(((i * 3 + 0) * 2 + 1) * 4) = (i * 4 + 1);
                *this->idxData[0].AsAt<unsigned int>(((i * 3 + 1) * 2 + 0) * 4) = (i * 4 + 1);
                *this->idxData[0].AsAt<unsigned int>(((i * 3 + 1) * 2 + 1) * 4) = (i * 4 + 2);
                *this->idxData[0].AsAt<unsigned int>(((i * 3 + 2) * 2 + 0) * 4) = (i * 4 + 2);
                *this->idxData[0].AsAt<unsigned int>(((i * 3 + 2) * 2 + 1) * 4) = (i * 4 + 3);

                for (unsigned int j = 0; j < 3; j++) {
                    vislib::math::Vector<float, 3> tang
                        = curve.ControlPoint(j).Position()
                        - curve.ControlPoint(j + 1).Position();

                    if (tang.Length() < 0.0001f) continue; // too short
                    tang.Normalise();
                    vislib::math::Vector<float, 3> vx(1.0f, 0.0f, 0.0f);
                    vislib::math::Vector<float, 3> vy = tang.Cross(vx);
                    if (vy.Length() < 0.0001f) {
                        vx.Set(0.0f, 0.0f, 1.0f);
                        vy = tang.Cross(vx);
                    }
                    vy.Normalise();
                    vx = vy.Cross(tang);
                    vx.Normalise();

                    vislib::math::ShallowPoint<float, 3> p0(this->vertData[1].AsAt<float>((lcnt * 8 + 0) * 3 * 4));
                    vislib::math::ShallowPoint<float, 3> p1(this->vertData[1].AsAt<float>((lcnt * 8 + 1) * 3 * 4));
                    vislib::math::ShallowPoint<float, 3> p2(this->vertData[1].AsAt<float>((lcnt * 8 + 2) * 3 * 4));
                    vislib::math::ShallowPoint<float, 3> p3(this->vertData[1].AsAt<float>((lcnt * 8 + 3) * 3 * 4));
                    vislib::math::ShallowPoint<float, 3> p4(this->vertData[1].AsAt<float>((lcnt * 8 + 4) * 3 * 4));
                    vislib::math::ShallowPoint<float, 3> p5(this->vertData[1].AsAt<float>((lcnt * 8 + 5) * 3 * 4));
                    vislib::math::ShallowPoint<float, 3> p6(this->vertData[1].AsAt<float>((lcnt * 8 + 6) * 3 * 4));
                    vislib::math::ShallowPoint<float, 3> p7(this->vertData[1].AsAt<float>((lcnt * 8 + 7) * 3 * 4));

                    p0 = curve.ControlPoint(j).Position() + (-vx - vy) * curve.ControlPoint(j).Radius();
                    p1 = curve.ControlPoint(j).Position() + (vx - vy) * curve.ControlPoint(j).Radius();
                    p2 = curve.ControlPoint(j).Position() + (vx + vy) * curve.ControlPoint(j).Radius();
                    p3 = curve.ControlPoint(j).Position() + (-vx + vy) * curve.ControlPoint(j).Radius();

                    p4 = curve.ControlPoint(j + 1).Position() + (-vx - vy) * curve.ControlPoint(j + 1).Radius();
                    p5 = curve.ControlPoint(j + 1).Position() + (vx - vy) * curve.ControlPoint(j + 1).Radius();
                    p6 = curve.ControlPoint(j + 1).Position() + (vx + vy) * curve.ControlPoint(j + 1).Radius();
                    p7 = curve.ControlPoint(j + 1).Position() + (-vx + vy) * curve.ControlPoint(j + 1).Radius();

                    *this->idxData[1].AsAt<unsigned int>(((lcnt * 12 + 0) * 2 + 0) * 4) = (lcnt * 8 + 0);
                    *this->idxData[1].AsAt<unsigned int>(((lcnt * 12 + 0) * 2 + 1) * 4) = (lcnt * 8 + 1);
                    *this->idxData[1].AsAt<unsigned int>(((lcnt * 12 + 1) * 2 + 0) * 4) = (lcnt * 8 + 1);
                    *this->idxData[1].AsAt<unsigned int>(((lcnt * 12 + 1) * 2 + 1) * 4) = (lcnt * 8 + 2);
                    *this->idxData[1].AsAt<unsigned int>(((lcnt * 12 + 2) * 2 + 0) * 4) = (lcnt * 8 + 2);
                    *this->idxData[1].AsAt<unsigned int>(((lcnt * 12 + 2) * 2 + 1) * 4) = (lcnt * 8 + 3);
                    *this->idxData[1].AsAt<unsigned int>(((lcnt * 12 + 3) * 2 + 0) * 4) = (lcnt * 8 + 3);
                    *this->idxData[1].AsAt<unsigned int>(((lcnt * 12 + 3) * 2 + 1) * 4) = (lcnt * 8 + 0);

                    *this->idxData[1].AsAt<unsigned int>(((lcnt * 12 + 4) * 2 + 0) * 4) = (lcnt * 8 + 4);
                    *this->idxData[1].AsAt<unsigned int>(((lcnt * 12 + 4) * 2 + 1) * 4) = (lcnt * 8 + 5);
                    *this->idxData[1].AsAt<unsigned int>(((lcnt * 12 + 5) * 2 + 0) * 4) = (lcnt * 8 + 5);
                    *this->idxData[1].AsAt<unsigned int>(((lcnt * 12 + 5) * 2 + 1) * 4) = (lcnt * 8 + 6);
                    *this->idxData[1].AsAt<unsigned int>(((lcnt * 12 + 6) * 2 + 0) * 4) = (lcnt * 8 + 6);
                    *this->idxData[1].AsAt<unsigned int>(((lcnt * 12 + 6) * 2 + 1) * 4) = (lcnt * 8 + 7);
                    *this->idxData[1].AsAt<unsigned int>(((lcnt * 12 + 7) * 2 + 0) * 4) = (lcnt * 8 + 7);
                    *this->idxData[1].AsAt<unsigned int>(((lcnt * 12 + 7) * 2 + 1) * 4) = (lcnt * 8 + 4);

                    *this->idxData[1].AsAt<unsigned int>(((lcnt * 12 + 8) * 2 + 0) * 4) = (lcnt * 8 + 0);
                    *this->idxData[1].AsAt<unsigned int>(((lcnt * 12 + 8) * 2 + 1) * 4) = (lcnt * 8 + 4);
                    *this->idxData[1].AsAt<unsigned int>(((lcnt * 12 + 9) * 2 + 0) * 4) = (lcnt * 8 + 1);
                    *this->idxData[1].AsAt<unsigned int>(((lcnt * 12 + 9) * 2 + 1) * 4) = (lcnt * 8 + 5);
                    *this->idxData[1].AsAt<unsigned int>(((lcnt * 12 + 10) * 2 + 0) * 4) = (lcnt * 8 + 2);
                    *this->idxData[1].AsAt<unsigned int>(((lcnt * 12 + 10) * 2 + 1) * 4) = (lcnt * 8 + 6);
                    *this->idxData[1].AsAt<unsigned int>(((lcnt * 12 + 11) * 2 + 0) * 4) = (lcnt * 8 + 3);
                    *this->idxData[1].AsAt<unsigned int>(((lcnt * 12 + 11) * 2 + 1) * 4) = (lcnt * 8 + 7);

                    lcnt++;
                }

            }

            this->vertData[1].EnforceSize(lcnt * 8 * 3 * 4, true);
            this->idxData[1].EnforceSize(lcnt * 12 * 2 * 4, true);
            if (this->vertData[0].IsEmpty()) {
                this->vertData[0].EnforceSize(1);
                this->idxData[0].EnforceSize(1);
            }
            if (this->vertData[1].IsEmpty()) {
                this->vertData[1].EnforceSize(1);
                this->idxData[1].EnforceSize(1);
            }

            this->lines[0].Set(
                static_cast<unsigned int>(this->idxData[0].GetSize() / sizeof(unsigned int)),
                this->idxData[0].As<unsigned int>(), this->vertData[0].As<float>(),
                vislib::graphics::NamedColours::SlateGray);
            this->lines[1].Set(
                static_cast<unsigned int>(this->idxData[1].GetSize() / sizeof(unsigned int)),
                this->idxData[1].As<unsigned int>(), this->vertData[1].As<float>(),
                vislib::graphics::NamedColours::LightSteelBlue);
        }

        ldc->SetData(2, this->lines);
        ldc->SetDataHash(this->hash);
    }

    return true;
}


/*
 * misc::BezierControlLines::getExtentCallback
 */
bool misc::BezierControlLines::getExtentCallback(Call& call) {
    LinesDataCall *ldc = dynamic_cast<LinesDataCall*>(&call);
    if (ldc == NULL) return false;

    BezierDataCall *bdc = this->getDataSlot.CallAs<BezierDataCall>();
    if ((bdc == NULL) || (!(*bdc)(1))) {
        ldc->AccessBoundingBoxes().Clear();
        ldc->SetFrameCount(1);
        ldc->SetDataHash(0);

    } else {
        ldc->AccessBoundingBoxes() = bdc->AccessBoundingBoxes();
        ldc->SetFrameCount(bdc->FrameCount());
        ldc->SetDataHash(bdc->DataHash());

    }

    return true;
}
