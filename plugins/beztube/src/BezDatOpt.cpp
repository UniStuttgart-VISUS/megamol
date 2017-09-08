/*
 * BezDatOpt.cpp
 *
 * Copyright (C) 2013 by TU Dresden
 * Alle Rechte vorbehalten.
 */
#include "stdafx.h"
#include "BezDatOpt.h"
//#include "param/FilePathParam.h"
//#include "vislib/VersionNumber.h"
//#include "vislib/Log.h"
//#include "vislib/String.h"
//#include "vislib/CharTraits.h"
//#include "vislib/mathfunctions.h"
//#include "vislib/RawStorage.h"
//#include "vislib/RawStorageWriter.h"

using namespace megamol;
using namespace megamol::beztube;


/*
 * BezDatOpt::BezDatOpt
 */
BezDatOpt::BezDatOpt(void) : core::Module(),
        outDataSlot("outData", "providing data"),
        inDataSlot("inData", "fetching data"),
        dataHash(0), frameID(0), data() {

    this->outDataSlot.SetCallback(core::misc::BezierCurvesListDataCall::ClassName(), "GetData", &BezDatOpt::getData);
    this->outDataSlot.SetCallback(core::misc::BezierCurvesListDataCall::ClassName(), "GetExtent", &BezDatOpt::getExtent);
    this->MakeSlotAvailable(&this->outDataSlot);

    this->inDataSlot.SetCompatibleCall<core::misc::BezierCurvesListDataCallDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);
}


/*
 * BezDatOpt::~BezDatOpt
 */
BezDatOpt::~BezDatOpt(void) {
    this->Release();
}


/*
 * BezDatOpt::create
 */
bool BezDatOpt::create(void) {
    // intentionally empty
    return true;
}


/*
 * BezDatOpt::release
 */
void BezDatOpt::release(void) {
    // intentionally empty
}


/*
 * BezDatOpt::is_equal
 */
bool BezDatOpt::is_equal(const full_point_type& lhs, const full_point_type& rhs, const float epsilon) {
    return vislib::math::IsEqual(lhs.x, rhs.x, epsilon)
        && vislib::math::IsEqual(lhs.y, rhs.y, epsilon)
        && vislib::math::IsEqual(lhs.z, rhs.z, epsilon)
        && vislib::math::IsEqual(lhs.r, rhs.r, epsilon)
        && (lhs.col[0] == rhs.col[0])
        && (lhs.col[1] == rhs.col[1])
        && (lhs.col[2] == rhs.col[2]);
    // ignoring count, as that one is not for storage
}


/*
 * BezDatOpt::getData
 */
bool BezDatOpt::getData(megamol::core::Call& call) {
    core::misc::BezierCurvesListDataCall *outCall = dynamic_cast<core::misc::BezierCurvesListDataCall*>(&call);
    if (outCall == NULL) return false;
    core::misc::BezierCurvesListDataCall *inCall = this->inDataSlot.CallAs<core::misc::BezierCurvesListDataCall>();
    if (inCall == NULL) return false;

    inCall->SetFrameID(outCall->FrameID(), outCall->IsFrameForced());

    if (!(*inCall)(1)) return false; // failed to fetch data
    if ((this->dataHash == 0) || (this->dataHash != inCall->DataHash()) || (this->frameID != inCall->FrameID())) {
        this->assertData(inCall->FrameID());
    }

    outCall->SetDataHash(this->dataHash);
    outCall->SetFrameID(this->frameID);
    outCall->SetData(this->data.PeekElements(), this->data.Count()); // sfx: hasStaticIndices = false;
    outCall->SetUnlocker(nullptr);

    return true;
}


/*
 * BezDatOpt::getExtent
 */
bool BezDatOpt::getExtent(megamol::core::Call& call) {
    core::misc::BezierCurvesListDataCall *outCall = dynamic_cast<core::misc::BezierCurvesListDataCall*>(&call);
    if (outCall == NULL) return false;
    core::misc::BezierCurvesListDataCall *inCall = this->inDataSlot.CallAs<core::misc::BezierCurvesListDataCall>();
    if (inCall == NULL) return false;

    inCall->SetFrameID(outCall->FrameID(), outCall->IsFrameForced());

    if (!(*inCall)(1)) return false; // failed to fetch data
    if ((this->dataHash == 0) || (this->dataHash != inCall->DataHash()) || (this->frameID != inCall->FrameID())) {
        this->assertData(inCall->FrameID());
    }

    outCall->SetExtent(
        inCall->FrameCount(),
        inCall->AccessBoundingBoxes());
    outCall->SetDataHash(this->dataHash);
    outCall->SetFrameID(this->frameID);
    outCall->SetHasStaticIndices(false); // since optimization is performed per frame
    outCall->SetUnlocker(nullptr);

    return true;
}


/*
 * BezDatOpt::assertData
 */
void BezDatOpt::assertData(unsigned int frameID) {
    core::misc::BezierCurvesListDataCall *inCall = this->inDataSlot.CallAs<core::misc::BezierCurvesListDataCall>();
    if (inCall == NULL) return;

    this->data.Clear();
    this->dataHash = 0;

    inCall->SetFrameID(frameID, true);
    if (!(*inCall)(0)) return; // failed to fetch data

    this->dataHash = inCall->DataHash();
    this->frameID = inCall->FrameID();
    this->data.SetCount(inCall->Count());
    for (size_t i = 0; i < this->data.Count(); i++) {
        this->optimize(this->data[i], inCall->GetCurves()[i]);
    }

}


/*
 * BezDatOpt::optimize
 */
void BezDatOpt::optimize(core::misc::BezierCurvesListDataCall::Curves& optDat,
        const core::misc::BezierCurvesListDataCall::Curves& inDat) {
    using core::misc::BezierCurvesListDataCall;

    const float epsilon = 0.000001f;

    optDat.Clear();

    // Step 1: can layout be simpified?
    float glob_rad = 0.5f;
    unsigned char glob_col[3];
    glob_col[0] = glob_col[1] = glob_col[2] = 127;
    core::misc::BezierCurvesListDataCall::DataLayout layout
        = core::misc::BezierCurvesListDataCall::DATALAYOUT_NONE;
    this->opt_layout(layout, glob_rad, glob_col, inDat);

    // Step 2: remove duplicate points
    unsigned int point_count = static_cast<unsigned int>(inDat.GetDataPointCount());
    full_point_type *points = new full_point_type[point_count];
    unsigned int index_count = static_cast<unsigned int>(inDat.GetIndexCount());
    unsigned int *indices = new unsigned int[index_count];

    // first copy data in simpler data structures (less memory efficient)
    unsigned int bpp = 0;
    bool has_rad = false;
    bool has_col = false;
    switch (inDat.GetDataLayout()) {
        case BezierCurvesListDataCall::DATALAYOUT_XYZ_F_RGB_B: bpp = 3 * 4 + 3 * 1; has_col = true; break;
        case BezierCurvesListDataCall::DATALAYOUT_XYZR_F_RGB_B: bpp = 4 * 4 + 3 * 1; has_rad = true; has_col = true; break;
        case BezierCurvesListDataCall::DATALAYOUT_XYZR_F: bpp = 4 * 4; has_rad = true; break;
        case BezierCurvesListDataCall::DATALAYOUT_XYZ_F: // fall through
        case BezierCurvesListDataCall::DATALAYOUT_NONE: // fall through
        default: break;
    }
    for (unsigned int i = 0; i < point_count; i++) {
        points[i].x = inDat.GetDataAt<float>(i * bpp)[0];
        points[i].y = inDat.GetDataAt<float>(i * bpp)[1];
        points[i].z = inDat.GetDataAt<float>(i * bpp)[2];
        points[i].r = has_rad ? inDat.GetDataAt<float>(i * bpp)[4] : glob_rad;
        points[i].col[0] = has_col ? inDat.GetDataAt<unsigned char>((i + 1) * bpp - 3)[0] : glob_col[0];
        points[i].col[1] = has_col ? inDat.GetDataAt<unsigned char>((i + 1) * bpp - 3)[1] : glob_col[1];
        points[i].col[2] = has_col ? inDat.GetDataAt<unsigned char>((i + 1) * bpp - 3)[2] : glob_col[2];
        points[i].counter = 0;
    }

    // copy index and count point usage
    for (unsigned int i = 0; i < index_count; i++) {
        indices[i] = inDat.GetIndex()[i];
        if (points[indices[i]].counter < 3) points[indices[i]].counter++;
        if (((i % 4) == 1) || ((i % 4) == 2)) {
            // inner control points count double
            if (points[indices[i]].counter < 3) points[indices[i]].counter++;
        }
    }
    // Debug output of initial point usage
    //printf("p:");
    //for (unsigned int i = 0; i < point_count; i++) {
    //    printf("%u", static_cast<unsigned int>(points[i].counter));
    //}
    //printf("\n");

    // Remove point duplicates. This is O(n^2) but I don't care for now
    for (unsigned int i = 0; i < point_count; i++) {
        if (points[i].counter == 0) continue;
        for (unsigned int j = 0; j < i; j++) {
            if (points[j].counter == 0) continue;
            if (is_equal(points[i], points[j], epsilon)) {
                for (unsigned int k = 0; k < index_count; k++) {
                    if (indices[k] == i) {
                        indices[k] = j;
                        if (points[j].counter < 3) points[j].counter++;
                    }
                }
                points[i].counter = 0;
                break; // for j
            }
        }
    }
    // Debug output of new point usage
    //printf("p:");
    //for (unsigned int i = 0; i < point_count; i++) {
    //    printf("%u", static_cast<unsigned int>(points[i].counter));
    //}
    //printf("\n");

    // Point usage marks:
    //  0 Point is no longer in use
    //  1 Point is an end point (keep)
    //  2 Point is control point (modify)
    //  3 Point is used by multiple curves (keep)

    // Step 3: collect connected elements and compute these independent

    // TODO: Implement


    // Step 4: fit spline curve

    // TODO: Implement

    delete[] points;
    delete[] indices;

}


/*
 * BezDatOpt::opt_layout
 */
void BezDatOpt::opt_layout(core::misc::BezierCurvesListDataCall::DataLayout& out_layout,
        float& out_glob_rad, unsigned char *out_glob_col,
        const core::misc::BezierCurvesListDataCall::Curves& inDat) {
    using core::misc::BezierCurvesListDataCall;

    const float epsilon = 0.000001f;

    unsigned int bpp = 0;
    bool has_rad = false;
    bool has_col = false;
    switch (inDat.GetDataLayout()) {
        case BezierCurvesListDataCall::DATALAYOUT_XYZ_F_RGB_B: bpp = 3 * 4 + 3 * 1; has_col = true; break;
        case BezierCurvesListDataCall::DATALAYOUT_XYZR_F_RGB_B: bpp = 4 * 4 + 3 * 1; has_rad = true; has_col = true; break;
        case BezierCurvesListDataCall::DATALAYOUT_XYZR_F: bpp = 4 * 4; has_rad = true; break;
        case BezierCurvesListDataCall::DATALAYOUT_XYZ_F: // fall through
        case BezierCurvesListDataCall::DATALAYOUT_NONE: // fall through
        default: break;
    }

    if ((bpp > 0) && (has_rad || has_col)) {
        size_t pc = inDat.GetDataPointCount();
        bool need_rad = !has_rad;
        bool need_col = !has_col;
        for (size_t i = 0; i < pc; i++) {
            if (need_rad && need_col) break; // nothing will help
            if (has_rad && !need_rad) {
                float r = inDat.GetDataAt<float>(i * bpp)[3];
                if (i == 0) out_glob_rad = r;
                else if (!vislib::math::IsEqual(r, out_glob_rad, epsilon)) need_rad = true;
            }
            if (has_col && !need_col) {
                const unsigned char* col = inDat.GetDataAt<unsigned char>((i + 1) * bpp - 3);
                if (i == 0) ::memcpy(out_glob_col, col, 3);
                else if (::memcmp(out_glob_col, col, 3) != 0) need_col = true;
            }
        }
        has_rad = need_rad;
        has_col = need_col;
        if (need_rad) out_glob_rad = inDat.GetGlobalRadius();
        if (need_col) ::memcpy(out_glob_col, inDat.GetGlobalColour(), 3);
    }

    if (!has_rad) out_glob_rad = inDat.GetGlobalRadius();
    if (!has_col) ::memcpy(out_glob_col, inDat.GetGlobalColour(), 3);
    if (has_rad) {
        if (has_col) {
            out_layout = BezierCurvesListDataCall::DATALAYOUT_XYZR_F_RGB_B;
        } else {
            out_layout = BezierCurvesListDataCall::DATALAYOUT_XYZR_F;
        }
    } else {
        if (has_col) {
            out_layout = BezierCurvesListDataCall::DATALAYOUT_XYZ_F_RGB_B;
        } else {
            if ((inDat.GetDataLayout() == BezierCurvesListDataCall::DATALAYOUT_NONE) 
                    || (inDat.GetDataPointCount() <= 0)) {
                out_layout = BezierCurvesListDataCall::DATALAYOUT_NONE;
            } else {
                out_layout = BezierCurvesListDataCall::DATALAYOUT_XYZ_F;
            }
        }
    }
}
