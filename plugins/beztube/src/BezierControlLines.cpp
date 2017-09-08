/*
 * BezierControlLines.cpp
 *
 * Copyright (C) 2010 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#define _USE_MATH_DEFINES
#define VLDEPRECATED
#define VISLIB_DEPRECATED_H_INCLUDED
#include "BezierControlLines.h"
#include "vislib/math/BezierCurve.h"
#include "vislib/graphics/NamedColours.h"
#include "vislib/math/ShallowPoint.h"
#include "vislib/math/Vector.h"

using namespace megamol::beztube;


/*
 * BezierControlLines::BezierControlLines
 */
BezierControlLines::BezierControlLines(void) : Module(),
        dataSlot("data", "Provides with line data"),
        getDataSlot("getData", "Gets bezier data"),
        hash(0), frameId(0) {

    this->dataSlot.SetCallback(trisoup::LinesDataCall::ClassName(), "GetData",
        &BezierControlLines::getDataCallback);
    this->dataSlot.SetCallback(trisoup::LinesDataCall::ClassName(), "GetExtent",
        &BezierControlLines::getExtentCallback);
    this->MakeSlotAvailable(&this->dataSlot);

    this->getDataSlot.SetCompatibleCall<v1::BezierDataCallDescription>();
    this->getDataSlot.SetCompatibleCall<megamol::core::misc::BezierCurvesListDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

}


/*
 * BezierControlLines::~BezierControlLines
 */
BezierControlLines::~BezierControlLines(void) {
    this->Release();
}


/*
 * BezierControlLines::create
 */
bool BezierControlLines::create(void) {
    // intentionally empty
    return true;
}


/*
 * BezierControlLines::release
 */
void BezierControlLines::release(void) {
    // intentionally empty
}


/*
 * BezierControlLines::getDataCallback
 */
bool BezierControlLines::getDataCallback(megamol::core::Call& call) {
    trisoup::LinesDataCall *ldc = dynamic_cast<trisoup::LinesDataCall*>(&call);
    if (ldc == NULL) return false;

    v1::BezierDataCall *bdc = this->getDataSlot.CallAs<v1::BezierDataCall>();
    core::misc::BezierCurvesListDataCall *bcldc = this->getDataSlot.CallAs<core::misc::BezierCurvesListDataCall>();

    if (bdc != nullptr) {
        bdc->SetFrameID(ldc->FrameID(), ldc->IsFrameForced());
        if ((*bdc)(0)) {
            if ((bdc->DataHash() == 0) || (bdc->DataHash() != this->hash) || (bdc->FrameID() != this->frameId)) {
                this->hash = bdc->DataHash();
                this->frameId = bdc->FrameID();
                this->makeLines(*bdc);
            }
        } else {
            bdc = nullptr;
            bcldc = nullptr;
        }
    } else if (bcldc != nullptr) {
        bcldc->SetFrameID(ldc->FrameID(), ldc->IsFrameForced());
        if ((*bcldc)(0)) {
            if ((bcldc->DataHash() == 0) || (bcldc->DataHash() != this->hash) || (this->frameId != bcldc->FrameID())) {
                this->hash = bcldc->DataHash();
                this->frameId = bcldc->FrameID();
                this->makeLines(*bcldc);
            }
        } else {
            bcldc = nullptr;
        }
    }
    if ((bdc == nullptr) && (bcldc == nullptr)) {
        ldc->SetData(0, NULL);
        ldc->SetDataHash(0);
        ldc->SetFrameID(0);
    } else {
        ldc->SetData(2, this->lines);
        ldc->SetDataHash(this->hash);
        ldc->SetFrameID(this->frameId);
    }

    return true;
}


/*
 * BezierControlLines::getExtentCallback
 */
bool BezierControlLines::getExtentCallback(megamol::core::Call& call) {
    trisoup::LinesDataCall *ldc = dynamic_cast<trisoup::LinesDataCall*>(&call);
    if (ldc == NULL) return false;

    v1::BezierDataCall *bdc = this->getDataSlot.CallAs<v1::BezierDataCall>();
    core::misc::BezierCurvesListDataCall *bcldc = this->getDataSlot.CallAs<core::misc::BezierCurvesListDataCall>();

    if (bdc != nullptr) {
        bdc->SetFrameID(ldc->FrameID(), ldc->IsFrameForced());
        if ((*bdc)(1)) {
            ldc->AccessBoundingBoxes() = bdc->AccessBoundingBoxes();
            ldc->SetFrameCount(bdc->FrameCount());
            ldc->SetDataHash(bdc->DataHash());
        } else {
            bdc = nullptr;
            bcldc = nullptr;
        }
    } else if (bcldc != nullptr) {
        bcldc->SetFrameID(ldc->FrameID(), ldc->IsFrameForced());
        if ((*bcldc)(1)) {
            ldc->AccessBoundingBoxes() = bcldc->AccessBoundingBoxes();
            ldc->SetFrameCount(bcldc->FrameCount());
            ldc->SetDataHash(bcldc->DataHash());
        } else {
            bcldc = nullptr;
        }
    }
    if ((bdc == nullptr) && (bcldc == nullptr)) {
        ldc->AccessBoundingBoxes().Clear();
        ldc->SetFrameCount(1);
        ldc->SetDataHash(0);
    }

    return true;
}


/*
 * BezierControlLines::makeLines
 */
void BezierControlLines::makeLines(v1::BezierDataCall& dat) {
    this->vertData[0].EnforceSize(4 * 3 * 4 * dat.Count());
    this->idxData[0].EnforceSize(2 * 3 * 4 * dat.Count());

    unsigned int lcnt = 0;
    this->vertData[1].AssertSize(dat.Count() * 3 * 8 * 3 * 4);
    this->idxData[1].AssertSize(dat.Count() * 3 * 12 * 2 * 4);

    for (unsigned int i = 0; i < dat.Count(); i++) {
        const vislib::math::BezierCurve<v1::BezierDataCall::BezierPoint, 3>& curve = dat.Curves()[i];

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


/*
 * BezierControlLines::makeLines
 */
void BezierControlLines::makeLines(::megamol::core::misc::BezierCurvesListDataCall& dat) {
    using ::megamol::core::misc::BezierCurvesListDataCall;
    typedef vislib::math::Vector<float, 3> Vector;
    typedef vislib::math::Point<float, 3> Point;
    typedef vislib::math::ShallowPoint<float, 3> ShallowPoint;

    size_t allCnt = 0;
    for (size_t i = 0; i < dat.Count(); i++) {
        if (dat.GetCurves()[i].GetDataLayout() == BezierCurvesListDataCall::DATALAYOUT_NONE) continue;
        allCnt += dat.GetCurves()[i].GetIndexCount() / 4;
    }

    // core lines
    this->vertData[0].EnforceSize(4 * 3 * 4 * allCnt);
    this->idxData[0].EnforceSize(2 * 3 * 4 * allCnt);

    // hull boxes
    this->vertData[1].AssertSize(allCnt * 4 * 4 * 3 * 4); // 4 points per curve, 4 verictes per point, 3 coordinates per vertex, 4 byte per coordinate (float)
    this->idxData[1].AssertSize(allCnt * 4 * 4 * 2 * 4 // 4 points per curve, 4 lines per rect, 2 indices per line, 4 byte per index (unsiged int)
        + allCnt * 3 * 4 * 2 * 4);

    // for all lists
    allCnt = 0;
    for (size_t j = 0; j < dat.Count(); j++) {
        const BezierCurvesListDataCall::Curves& list = dat.GetCurves()[j];
        size_t bpp = 0;
        bool store_rad = false;
        switch (list.GetDataLayout()) {
        case BezierCurvesListDataCall::DATALAYOUT_XYZ_F: bpp = 3 * 4; break;
        case BezierCurvesListDataCall::DATALAYOUT_XYZ_F_RGB_B: bpp = 3 * 4 + 3 * 1; break;
        case BezierCurvesListDataCall::DATALAYOUT_XYZR_F_RGB_B: bpp = 4 * 4 + 3 * 1; store_rad = true; break;
        case BezierCurvesListDataCall::DATALAYOUT_XYZR_F: bpp = 4 * 4; store_rad = true; break;
        case BezierCurvesListDataCall::DATALAYOUT_NONE: // fall through
        default: continue; // skip this list
        }
        // for all curves in the list
        for (size_t i = 0; i < list.GetIndexCount() / 4; i++, allCnt++) {
            // store control point positions
            Point pts[4];
            for (size_t k = 0; k < 4; k++) {
                ::memcpy(this->vertData[0].At((allCnt * 4 + k) * 3 * 4), list.GetDataAt(bpp * list.GetIndex()[i * 4 + k]), 3 * 4);
                pts[k] = ShallowPoint(this->vertData[0].AsAt<float>((allCnt * 4 + k) * 3 * 4));
            }
            // three line segments for the core line
            *this->idxData[0].AsAt<unsigned int>(((allCnt * 3 + 0) * 2 + 0) * 4) = static_cast<unsigned int>(allCnt * 4 + 0);
            *this->idxData[0].AsAt<unsigned int>(((allCnt * 3 + 0) * 2 + 1) * 4) = static_cast<unsigned int>(allCnt * 4 + 1);
            *this->idxData[0].AsAt<unsigned int>(((allCnt * 3 + 1) * 2 + 0) * 4) = static_cast<unsigned int>(allCnt * 4 + 1);
            *this->idxData[0].AsAt<unsigned int>(((allCnt * 3 + 1) * 2 + 1) * 4) = static_cast<unsigned int>(allCnt * 4 + 2);
            *this->idxData[0].AsAt<unsigned int>(((allCnt * 3 + 2) * 2 + 0) * 4) = static_cast<unsigned int>(allCnt * 4 + 2);
            *this->idxData[0].AsAt<unsigned int>(((allCnt * 3 + 2) * 2 + 1) * 4) = static_cast<unsigned int>(allCnt * 4 + 3);

            // collect radii
            float rads[4];
            if (store_rad) {
                for (size_t k = 0; k < 4; k++) {
                    rads[k] = list.GetDataAt<float>(bpp * list.GetIndex()[i * 4 + k])[3];
                }
            } else {
                for (size_t k = 0; k < 4; k++) {
                    rads[k] = list.GetGlobalRadius();
                }
            }

            // collect tangent and basis vectors
            Vector tan[4];
            Vector v1[4];
            Vector v2[4];
            tan[0] = pts[1] - pts[0]; // forward diff
            tan[1] = pts[2] - pts[0]; // center diff
            tan[2] = pts[3] - pts[1]; // center diff
            tan[3] = pts[3] - pts[2]; // backward diff

            if (tan[0].IsNull()) {
                if (tan[1].IsNull()) {
                    if (tan[2].IsNull()) {
                        if (tan[3].IsNull()) {
                            tan[3].Set(1.0f, 0.0f, 0.0f); // as good as any
                        }
                        tan[0] = tan[1] = tan[2] = tan[3];
                    } else {
                        tan[0] = tan[1] = tan[2];
                    }
                } else {
                    tan[0] = tan[1];
                }
            }
            ASSERT(!tan[0].IsNull());
            if (tan[3].IsNull()) {
                if (tan[2].IsNull()) {
                    if (tan[1].IsNull()) {
                        tan[3] = tan[2] = tan[1] = tan[0];
                    }
                    tan[3] = tan[2] = tan[1];
                } else {
                    tan[3] = tan[2];
                }
            }
            ASSERT(!tan[3].IsNull());
            if (tan[1].IsNull()) {
                if (tan[2].IsNull()) {
                    tan[1] = tan[2] = tan[0];
                }
                tan[1] = tan[2];
            }
            ASSERT(!tan[1].IsNull());
            if (tan[2].IsNull()) {
                tan[2] = tan[1];
            }

            for (size_t k = 0; k < 4; k++) {
                ASSERT(!tan[k].IsNull());
                tan[k].Normalise();

                v1[k].Set(0.0f, 1.0f, 0.0);
                if (tan[k].IsParallel(v1[k])) v1[k].Set(0.0f, 0.0f, 1.0f);

                v2[k] = tan[k].Cross(v1[k]);
                ASSERT(!v2[k].IsNull());
                v2[k].Normalise();

                v1[k] = v2[k].Cross(tan[k]);
                v1[k].Normalise();

                v1[k] *= 0.5f * rads[k];
                v2[k] *= 0.5f * rads[k];
            }

            // write 4*4 vertices to vertex array
            for (size_t k = 0; k < 4; k++) {
                Point ep[4];
                ep[0] = pts[k] - v1[k] - v2[k];
                ep[1] = pts[k] + v1[k] - v2[k];
                ep[2] = pts[k] + v1[k] + v2[k];
                ep[3] = pts[k] - v1[k] + v2[k];
                for (size_t l = 0; l < 4; l++) {
                    ::memcpy(this->vertData[1].At(((allCnt * 4 + k) * 4 + l) * 3 * 4), ep[l].PeekCoordinates(), 3 * 4);
                }
            }

            // write 4*4 *2 line indices to index array
            for (size_t k = 0; k < 4; k++) {
                this->idxData[1].AsAt<unsigned int>((allCnt * 4 * 4 * 2 * 4) + (allCnt * 3 * 4 * 2 * 4))[0 + k * 8] = static_cast<unsigned int>(allCnt * 4 + k) * 4 + 0;
                this->idxData[1].AsAt<unsigned int>((allCnt * 4 * 4 * 2 * 4) + (allCnt * 3 * 4 * 2 * 4))[1 + k * 8] = static_cast<unsigned int>(allCnt * 4 + k) * 4 + 1;
                this->idxData[1].AsAt<unsigned int>((allCnt * 4 * 4 * 2 * 4) + (allCnt * 3 * 4 * 2 * 4))[2 + k * 8] = static_cast<unsigned int>(allCnt * 4 + k) * 4 + 1;
                this->idxData[1].AsAt<unsigned int>((allCnt * 4 * 4 * 2 * 4) + (allCnt * 3 * 4 * 2 * 4))[3 + k * 8] = static_cast<unsigned int>(allCnt * 4 + k) * 4 + 2;
                this->idxData[1].AsAt<unsigned int>((allCnt * 4 * 4 * 2 * 4) + (allCnt * 3 * 4 * 2 * 4))[4 + k * 8] = static_cast<unsigned int>(allCnt * 4 + k) * 4 + 2;
                this->idxData[1].AsAt<unsigned int>((allCnt * 4 * 4 * 2 * 4) + (allCnt * 3 * 4 * 2 * 4))[5 + k * 8] = static_cast<unsigned int>(allCnt * 4 + k) * 4 + 3;
                this->idxData[1].AsAt<unsigned int>((allCnt * 4 * 4 * 2 * 4) + (allCnt * 3 * 4 * 2 * 4))[6 + k * 8] = static_cast<unsigned int>(allCnt * 4 + k) * 4 + 3;
                this->idxData[1].AsAt<unsigned int>((allCnt * 4 * 4 * 2 * 4) + (allCnt * 3 * 4 * 2 * 4))[7 + k * 8] = static_cast<unsigned int>(allCnt * 4 + k) * 4 + 0;
            }
            // write 3 * 4 * 2 line indices along the index array
            for (size_t k = 0; k < 3; k++) {
                for (size_t l = 0; l < 4; l++) {
                    this->idxData[1].AsAt<unsigned int>((allCnt * 4 * 4 * 2 * 4) + (allCnt * 3 * 4 * 2 * 4))[32 + 0 + l * 2 + k * 8] = static_cast<unsigned int>((allCnt * 4 + k) * 4 + l);
                    this->idxData[1].AsAt<unsigned int>((allCnt * 4 * 4 * 2 * 4) + (allCnt * 3 * 4 * 2 * 4))[32 + 1 + l * 2 + k * 8] = static_cast<unsigned int>((allCnt * 4 + k + 1) * 4 + l);
                }
            }
        }
    }

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
