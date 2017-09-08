/*
 * BezierLines.cpp
 *
 * Copyright (C) 2013 by TU Dresden
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#define _USE_MATH_DEFINES
#include "BezierLines.h"
#include "vislib/math/BezierCurve.h"
#include "vislib/graphics/NamedColours.h"
#include "vislib/math/ShallowPoint.h"
#include "vislib/math/Vector.h"
#include "mmcore/param/IntParam.h"
#include "vislib/RawStorageWriter.h"

using namespace megamol::beztube;


/*
 * BezierLines::BezierLines
 */
BezierLines::BezierLines(void) : Module(),
        dataSlot("data", "Provides with line data"),
        getDataSlot("getData", "Gets bezier data"),
        inHash(0), outHash(0), frameId(0),
        numSegsSlot("numsegs", "Number of linear segments") {

    this->dataSlot.SetCallback(trisoup::LinesDataCall::ClassName(), "GetData",
        &BezierLines::getDataCallback);
    this->dataSlot.SetCallback(trisoup::LinesDataCall::ClassName(), "GetExtent",
        &BezierLines::getExtentCallback);
    this->MakeSlotAvailable(&this->dataSlot);

    this->getDataSlot.SetCompatibleCall<megamol::core::misc::BezierCurvesListDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->numSegsSlot << new core::param::IntParam(8, 1, 100);
    this->MakeSlotAvailable(&this->numSegsSlot);

}


/*
 * BezierLines::~BezierLines
 */
BezierLines::~BezierLines(void) {
    this->Release();
}


/*
 * BezierLines::create
 */
bool BezierLines::create(void) {
    // intentionally empty
    return true;
}


/*
 * BezierLines::release
 */
void BezierLines::release(void) {
    // intentionally empty
}


/*
 * BezierLines::getDataCallback
 */
bool BezierLines::getDataCallback(megamol::core::Call& call) {
    trisoup::LinesDataCall *ldc = dynamic_cast<trisoup::LinesDataCall*>(&call);
    if (ldc == NULL) return false;

    core::misc::BezierCurvesListDataCall *bcldc = this->getDataSlot.CallAs<core::misc::BezierCurvesListDataCall>();

    if (bcldc != nullptr) {
        bcldc->SetFrameID(ldc->FrameID(), ldc->IsFrameForced());
        if ((*bcldc)(0)) {
            if ((bcldc->DataHash() == 0) || (bcldc->DataHash() != this->inHash) || (this->frameId != bcldc->FrameID()) || this->numSegsSlot.IsDirty()) {
                this->inHash = bcldc->DataHash();
                this->frameId = bcldc->FrameID();
                this->outHash++;
                this->makeLines(*bcldc);
            }
        } else {
            bcldc = nullptr;
        }
    }

    if (bcldc == nullptr) {
        ldc->SetData(0, NULL);
        ldc->SetDataHash(0);
        ldc->SetFrameID(0);
    } else {
        ldc->SetData(1, &this->lines);
        ldc->SetDataHash(this->outHash);
        ldc->SetFrameID(this->frameId);
    }

    return true;
}


/*
 * BezierLines::getExtentCallback
 */
bool BezierLines::getExtentCallback(megamol::core::Call& call) {
    trisoup::LinesDataCall *ldc = dynamic_cast<trisoup::LinesDataCall*>(&call);
    if (ldc == NULL) return false;

    core::misc::BezierCurvesListDataCall *bcldc = this->getDataSlot.CallAs<core::misc::BezierCurvesListDataCall>();

    if (bcldc != nullptr) {
        bcldc->SetFrameID(ldc->FrameID(), ldc->IsFrameForced());
        if ((*bcldc)(1)) {
            ldc->AccessBoundingBoxes() = bcldc->AccessBoundingBoxes();
            ldc->SetFrameCount(bcldc->FrameCount());
            ldc->SetDataHash(this->outHash);
        } else {
            bcldc = nullptr;
        }
    }

    if (bcldc == nullptr) {
        ldc->AccessBoundingBoxes().Clear();
        ldc->SetFrameCount(1);
        ldc->SetDataHash(0);
    }

    return true;
}


/*
 * BezierLines::makeLines
 */
void BezierLines::makeLines(::megamol::core::misc::BezierCurvesListDataCall& dat) {
    using ::megamol::core::misc::BezierCurvesListDataCall;
    typedef vislib::math::Vector<float, 3> Vector;
    typedef vislib::math::Point<float, 3> Point;
    typedef vislib::math::ShallowPoint<float, 3> ShallowPoint;

    unsigned int numSegs = this->numSegsSlot.Param<core::param::IntParam>()->Value();
    this->numSegsSlot.ResetDirty();

    size_t allCnt = 0;
    for (size_t i = 0; i < dat.Count(); i++) {
        if (dat.GetCurves()[i].GetDataLayout() == BezierCurvesListDataCall::DATALAYOUT_NONE) continue;
        allCnt += dat.GetCurves()[i].GetIndexCount() / 4;
    }

    // core lines
    vislib::RawStorageWriter vertWriter(this->vertData);
    vislib::RawStorageWriter idxWriter(this->idxData);
    vislib::RawStorageWriter colWriter(this->colData);

    // for all lists
    allCnt = 0;
    unsigned int numPts = 0;
    for (size_t j = 0; j < dat.Count(); j++) {
        const BezierCurvesListDataCall::Curves& list = dat.GetCurves()[j];
        size_t bpp = 0;
        bool has_col = false;
        switch (list.GetDataLayout()) {
        case BezierCurvesListDataCall::DATALAYOUT_XYZ_F: bpp = 3 * 4; break;
        case BezierCurvesListDataCall::DATALAYOUT_XYZ_F_RGB_B: bpp = 3 * 4 + 3 * 1; has_col = true; break;
        case BezierCurvesListDataCall::DATALAYOUT_XYZR_F_RGB_B: bpp = 4 * 4 + 3 * 1; has_col = true; break;
        case BezierCurvesListDataCall::DATALAYOUT_XYZR_F: bpp = 4 * 4; break;
        case BezierCurvesListDataCall::DATALAYOUT_NONE: // fall through
        default: continue; // skip this list
        }

        // for all curves in the list
        for (size_t i = 0; i < list.GetIndexCount() / 4; i++, allCnt++) {
            // store control point positions
            vislib::math::BezierCurve<Point, 3> cPos;
            vislib::math::BezierCurve<Point, 3> cCol;
            for (unsigned int k = 0; k < 4; k++) {
                cPos[k] = ShallowPoint(const_cast<float*>(list.GetDataAt<float>(bpp * list.GetIndex()[i * 4 + k])));
            }
            if (has_col) {
                for (unsigned int k = 0; k < 4; k++) {
                    cCol[k] = Point(vislib::math::ShallowPoint<unsigned char, 3>(const_cast<unsigned char*>(
                        list.GetDataAt<unsigned char>(bpp * (1 + list.GetIndex()[i * 4 + k]) - 3)
                        )));
                }
            }

            for (unsigned int k = 0; k <= numSegs; k++) {
                float t = static_cast<float>(k) / static_cast<float>(numSegs);

                Point p;
                cPos.CalcPoint(p, t);
                vertWriter << p.X() << p.Y() << p.Z();

                if (has_col) {
                    cCol.CalcPoint(p, t);
                    colWriter
                        << static_cast<unsigned char>(vislib::math::Clamp(p.X(), 0.0f, 255.0f))
                        << static_cast<unsigned char>(vislib::math::Clamp(p.Y(), 0.0f, 255.0f))
                        << static_cast<unsigned char>(vislib::math::Clamp(p.Z(), 0.0f, 255.0f));
                } else {
                    colWriter.Write(list.GetGlobalColour(), 3);
                }

                numPts++;

                if (k > 0) {
                    idxWriter << (numPts - 2) << (numPts - 1);
                }
            }
        }
    }

    if (vertWriter.Position() == 0) {
        this->vertData.EnforceSize(1);
        this->idxData.EnforceSize(1);
        this->colData.EnforceSize(1);
    }

    this->lines.Set(
        static_cast<unsigned int>(idxWriter.Position() / sizeof(unsigned int)),
        this->idxData.As<unsigned int>(),
        this->vertData.As<float>(),
        this->colData.As<unsigned char>(), false);
}
