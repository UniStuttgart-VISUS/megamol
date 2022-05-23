/*
 * SiffCSplineFitter.cpp
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#include "SiffCSplineFitter.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/utility/log/Log.h"
#include "vislib/math/BezierCurve.h"
#include "vislib/math/Point.h"
#include "vislib/math/ShallowPoint.h"
#include <climits>


#include "geometry_calls/MultiParticleDataCall.h"
#include "mmcore/param/EnumParam.h"

namespace megamol::datatools {

/*
 * SiffCSplineFitter::SiffCSplineFitter
 */
SiffCSplineFitter::SiffCSplineFitter(void)
        : core::Module()
        , getDataSlot("getdata", "The slot exposing the loaded data")
        , inDataSlot("indata", "The slot for fetching siff data")
        , colourMapSlot("colourMapping", "The parameter controlling the colour mapping")
        , deCycleSlot("deCycle", "To compensate cyclic boundary conditions")
        , bbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f)
        , cbox(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f)
        , curves()
        , datahash(0)
        , inhash(0) {

    core::param::EnumParam* ep = new core::param::EnumParam(1);
    ep->SetTypePair(0, "site colour");
    ep->SetTypePair(1, "time");
    this->colourMapSlot << ep;
    this->MakeSlotAvailable(&this->colourMapSlot);

    this->deCycleSlot << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->deCycleSlot);

    this->getDataSlot.SetCallback(
        geocalls::BezierCurvesListDataCall::ClassName(), "GetData", &SiffCSplineFitter::getDataCallback);
    this->getDataSlot.SetCallback(
        geocalls::BezierCurvesListDataCall::ClassName(), "GetExtent", &SiffCSplineFitter::getExtentCallback);
    this->MakeSlotAvailable(&this->getDataSlot);

    this->inDataSlot.SetCompatibleCall<geocalls::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);
}


/*
 * SiffCSplineFitter::~SiffCSplineFitter
 */
SiffCSplineFitter::~SiffCSplineFitter(void) {
    this->Release();
}


/*
 * SiffCSplineFitter::create
 */
bool SiffCSplineFitter::create(void) {
    // intentionally empty
    return true;
}


/*
 * SiffCSplineFitter::release
 */
void SiffCSplineFitter::release(void) {
    this->curves.Clear();
    this->bbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    this->cbox.Set(-1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f);
    this->datahash = 0;
}


/*
 * SiffCSplineFitter::getDataCallback
 */
bool SiffCSplineFitter::getDataCallback(core::Call& caller) {
    geocalls::BezierCurvesListDataCall* bdc = dynamic_cast<geocalls::BezierCurvesListDataCall*>(&caller);
    if (bdc == NULL)
        return false;

    bdc->SetData(this->curves.PeekElements(), this->curves.Count());
    bdc->SetDataHash(this->datahash);
    bdc->SetFrameCount(1);
    bdc->AccessBoundingBoxes().SetObjectSpaceBBox(bbox);
    bdc->AccessBoundingBoxes().SetObjectSpaceClipBox(cbox);
    bdc->SetFrameID(0);
    bdc->SetUnlocker(NULL);

    return true;
}


/*
 * SiffCSplineFitter::getExtentCallback
 */
bool SiffCSplineFitter::getExtentCallback(core::Call& caller) {
    geocalls::BezierCurvesListDataCall* bdc = dynamic_cast<geocalls::BezierCurvesListDataCall*>(&caller);
    if (bdc == NULL)
        return false;

    this->assertData();

    bdc->SetDataHash(this->datahash);
    bdc->SetFrameCount(1);
    bdc->AccessBoundingBoxes().SetObjectSpaceBBox(bbox);
    bdc->AccessBoundingBoxes().SetObjectSpaceClipBox(cbox);

    return true;
}


/*
 * SiffCSplineFitter::assertData
 */
void SiffCSplineFitter::assertData(void) {
    using megamol::core::utility::log::Log;

    geocalls::MultiParticleDataCall* mpdc = this->inDataSlot.CallAs<geocalls::MultiParticleDataCall>();
    if (mpdc == NULL)
        return;

    if (!(*mpdc)(1))
        return;
    vislib::math::Cuboid<float> bbox = mpdc->AccessBoundingBoxes().ObjectSpaceBBox();
    vislib::math::Cuboid<float> cbox = mpdc->AccessBoundingBoxes().ObjectSpaceClipBox();

    if (!(*mpdc)(0))
        return;
    if (!this->colourMapSlot.IsDirty() && !this->deCycleSlot.IsDirty() && (mpdc->DataHash() == this->inhash))
        return; // data did not change

    this->colourMapSlot.ResetDirty();
    this->deCycleSlot.ResetDirty();
    this->inhash = mpdc->DataHash();
    this->datahash++;
    this->curves.Clear();
    this->bbox = bbox;
    this->cbox = cbox;

    if (mpdc->GetParticleListCount() < 1)
        return;
    if (mpdc->GetParticleListCount() != 1) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "Spline fitter only supports single list data ATM");
    }
    unsigned int cnt = static_cast<unsigned int>(mpdc->AccessParticles(0).GetCount());
    if (cnt == 0)
        return;
    const unsigned char* vdata = static_cast<const unsigned char*>(mpdc->AccessParticles(0).GetVertexData());
    unsigned int vstride = mpdc->AccessParticles(0).GetVertexDataStride();
    const unsigned char* cdata = static_cast<const unsigned char*>(mpdc->AccessParticles(0).GetColourData());
    unsigned int cstride = mpdc->AccessParticles(0).GetColourDataStride();
    if (mpdc->AccessParticles(0).GetColourDataType() != geocalls::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "Spline fitter only supports colour data UINT8_RGB");
        return; // without colour we cannot detect the frames!
    } else if (cstride < 3)
        cstride = 3;
    if (mpdc->AccessParticles(0).GetVertexDataType() !=
        geocalls::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Spline fitter only supports vertex data FLOAT_XYZR");
        return;
    } else if (vstride < 4 * sizeof(float))
        vstride = 4 * sizeof(float);

    // count sites and frames
    // sites are always in same order in each frame
    const unsigned int ccdtestsize = 10; // sequence of 10 sites as frame detector
    const unsigned char* ccd = cdata + cstride * ccdtestsize;
    unsigned int frameSize = 0; // number of sites per frame
    for (unsigned int i = ccdtestsize; i < cnt - ccdtestsize; i++, ccd += cstride) {
        bool miss = false;
        for (unsigned int j = 0; j < ccdtestsize; j++) {
            if (::memcmp(ccd + j * cstride, cdata + cstride * j, 3) != 0) {
                miss = true;
                break;
            }
        }
        if (!miss) {
            frameSize = i;
            break;
        }
    }
    if (frameSize == 0) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR, "Frame detection failed");
        return;
    }
    unsigned int frameCnt = cnt / frameSize;
    if ((cnt % frameSize) != 0) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_WARN, "IMPORTANT!!! FrameSize * FrameCount != FileSize");
    }
    Log::DefaultLog.WriteMsg(Log::LEVEL_INFO + 50, "Found %u sites in %u frames\n", frameSize, frameCnt);

    unsigned char colR, colG, colB;
    float rad;
    float* pos = new float[frameCnt * 3];
    float* times = new float[frameCnt * 2];

    this->curves.AssertCapacity(frameSize);
    for (unsigned int i = 0; i < frameSize; i++) {
        colR = cdata[0];
        colG = cdata[1];
        colB = cdata[2];
        cdata += cstride;
        rad = reinterpret_cast<const float*>(vdata)[3];
        for (unsigned int j = 0; j < frameCnt; j++) {
            int off = (j * frameSize + i) * vstride;
            pos[j * 3] = reinterpret_cast<const float*>(vdata + off)[0];
            pos[j * 3 + 1] = reinterpret_cast<const float*>(vdata + off)[1];
            pos[j * 3 + 2] = reinterpret_cast<const float*>(vdata + off)[2];
            times[j * 2] = times[j * 2 + 1] =
                ((frameCnt > 1) ? (static_cast<float>(j) / static_cast<float>(frameCnt - 1)) : 0.0f);
        }

        this->addSpline(pos, times, frameCnt, rad, colR, colG, colB);
    }

    delete[] pos;
    delete[] times;
}


/*
 * SiffCSplineFitter::addSpline
 */
void SiffCSplineFitter::addSpline(
    float* pos, float* times, unsigned int cnt, float rad, unsigned char colR, unsigned char colG, unsigned char colB) {
    typedef vislib::math::ShallowPoint<float, 3> ShallowPoint;
    typedef vislib::math::Point<float, 3> Point;
    typedef vislib::math::Vector<float, 3> Vector;

    this->curves.Add(geocalls::BezierCurvesListDataCall::Curves());
    geocalls::BezierCurvesListDataCall::Curves* list = &this->curves.Last();

    const bool useTimeColour = (this->colourMapSlot.Param<core::param::EnumParam>()->Value() == 1);
    const bool deCycle = this->deCycleSlot.Param<core::param::BoolParam>()->Value();

    vislib::RawStorage data;
    vislib::RawStorage index;
    geocalls::BezierCurvesListDataCall::DataLayout layout =
        useTimeColour ? geocalls::BezierCurvesListDataCall::DATALAYOUT_XYZ_F_RGB_B
                      : geocalls::BezierCurvesListDataCall::DATALAYOUT_XYZ_F;
    size_t bpp = useTimeColour ? (3 * 4 + 3 * 1) : (3 * 4);

    if (cnt == 0)
        return;

    /*
        typedef vislib::math::BezierCurve<BezierDataCall::BezierPoint, 3> BezierCurve;
        BezierCurve curve;
    */
    const float bminB = 0.15f;
    const float bmaxB = 1.0f - bminB;

    // fix splines by either changing positions:
    //  deCycle allows splines to leave the bounding box, but the spline will stay in one
    //  false will split the splines into several segments all contained in the bounding box
    if (deCycle) {
        // compensate cyclic boundary conditions
        Vector off;
        Point p1(pos);

        float bminX = this->bbox.Right() * bminB + this->bbox.Left() * bmaxB;
        float bmaxX = this->bbox.Right() * bmaxB + this->bbox.Left() * bminB;
        float bminY = this->bbox.Top() * bminB + this->bbox.Bottom() * bmaxB;
        float bmaxY = this->bbox.Top() * bmaxB + this->bbox.Bottom() * bminB;
        float bminZ = this->bbox.Front() * bminB + this->bbox.Back() * bmaxB;
        float bmaxZ = this->bbox.Front() * bmaxB + this->bbox.Back() * bminB;

        for (unsigned int i = 1; i < cnt; i++) {
            ShallowPoint p2(pos + i * 3);

            if ((p1.X() < bminX) && (p2.X() > bmaxX)) { // cb-jump in +X
                off.SetX(off.X() + this->bbox.Width());
            }
            if ((p2.X() < bminX) && (p1.X() > bmaxX)) { // cb-jump in -X
                off.SetX(off.X() - this->bbox.Width());
            }
            if ((p1.Y() < bminY) && (p2.Y() > bmaxY)) { // cb-jump in +Y
                off.SetY(off.Y() + this->bbox.Height());
            }
            if ((p2.Y() < bminY) && (p1.Y() > bmaxY)) { // cb-jump in -Y
                off.SetY(off.Y() - this->bbox.Height());
            }
            if ((p1.Z() < bminZ) && (p2.Z() > bmaxZ)) { // cb-jump in +Z
                off.SetZ(off.Z() + this->bbox.Depth());
            }
            if ((p2.Z() < bminZ) && (p1.Z() > bmaxZ)) { // cb-jump in -Z
                off.SetZ(off.Z() - this->bbox.Depth());
            }

            p1 = p2;
            p2 -= off;
            this->cbox.GrowToPoint(p2);
        }

    } else {
        // split cyclic boundary jumps
        unsigned int s = 0;
        bool splitted = false;

        float bminX = this->bbox.Right() * bminB + this->bbox.Left() * bmaxB;
        float bmaxX = this->bbox.Right() * bmaxB + this->bbox.Left() * bminB;
        float bminY = this->bbox.Top() * bminB + this->bbox.Bottom() * bmaxB;
        float bmaxY = this->bbox.Top() * bmaxB + this->bbox.Bottom() * bminB;
        float bminZ = this->bbox.Front() * bminB + this->bbox.Back() * bmaxB;
        float bmaxZ = this->bbox.Front() * bmaxB + this->bbox.Back() * bminB;

        for (unsigned int i = 1; i < cnt; i++) {
            ShallowPoint p1(pos + (i - 1) * 3);
            ShallowPoint p2(pos + i * 3);
            float d = p1.Distance(p2);
            if (d < 2.0f * rad)
                continue;
            // test if is a cb-jump

            if (((p1.X() < bminX) && (p2.X() > bmaxX))       // cb-jump in X
                || ((p2.X() < bminX) && (p1.X() > bmaxX))    // cb-jump in X
                || ((p1.Y() < bminY) && (p2.Y() > bmaxY))    // cb-jump in Y
                || ((p2.Y() < bminY) && (p1.Y() > bmaxY))    // cb-jump in Y
                || ((p1.Z() < bminZ) && (p2.Z() > bmaxZ))    // cb-jump in Z
                || ((p2.Z() < bminZ) && (p1.Z() > bmaxZ))) { // cb-jump in Z

                // recurse for next spline segement
                SIZE_T list_idx = static_cast<SIZE_T>(this->curves.IndexOf(*list));
                this->addSpline(pos + s * 3, times + s * 2, i - s, rad, colR, colG, colB);
                list = &this->curves[list_idx];
                s = i;
                splitted = true;
            }
        }
        if (splitted) {
            pos += s * 3;
            times += s * 2;
            cnt -= s;
        }
    }

    if (cnt == 1) {
        // super trivial nonsense: simply dots
        unsigned char col[3];
        data.Append(pos, 3 * sizeof(float));
        if (useTimeColour) {
            this->timeColour(times[0], col[0], col[1], col[2]);
            data.Append(col, 3);
        }
        data.Append(pos, 3 * sizeof(float));
        if (useTimeColour) {
            this->timeColour((times[0] + times[1]) * 0.5f, col[0], col[1], col[2]);
            data.Append(col, 3);
        }
        data.Append(pos, 3 * sizeof(float));
        if (useTimeColour) {
            this->timeColour(times[1], col[0], col[1], col[2]);
            data.Append(col, 3);
        }

        unsigned int i[4] = {0, 1, 1, 2};
        index.Append(i, 4 * sizeof(unsigned int));

    } else {
        // looping and/or fitting
        // first fit a polyline
        vislib::Array<Point> lines;          // the polyline
        vislib::Array<unsigned int> indices; // the index of input point

        indices.Add(0);
        lines.Add(ShallowPoint(pos));
        indices.Add(cnt - 1);
        lines.Add(ShallowPoint(pos + (cnt - 1) * 3));

        const float distEps = rad;

        bool refined = true;
        while (refined) {
            refined = false;
            for (unsigned int l = 1; l < lines.Count(); l++) {
                // The line segment
                ShallowPoint p1(pos + indices[l - 1] * 3);
                ShallowPoint p2(pos + indices[l] * 3);
                Vector lv = p2 - p1;
                float ll = lv.Normalise();

                unsigned int maxP = UINT_MAX;
                float maxDist = -1.0f;

                // search point with max distance to current line segment
                for (unsigned int p = indices[l - 1] + 1; p < indices[l]; p++) {
                    ShallowPoint pn(pos + p * 3);
                    Vector pv = pn - p1;
                    float l = lv.Dot(pv);
                    float dist;

                    if (l < 0.0) {
                        dist = pn.Distance(p1);
                    } else if (l > ll) {
                        dist = pn.Distance(p2);
                    } else {
                        dist = pn.Distance(p1 + lv * l);
                    }

                    if (maxDist < dist) {
                        maxDist = dist;
                        maxP = p;
                    }
                }

                if (maxDist > distEps) {
                    // split this line
                    indices.Insert(l, maxP);
                    lines.Insert(l, ShallowPoint(pos + maxP * 3));
                    l++;
                    refined = true;
                }
            }
        }

        // approx lines though curved (bullshit but good enough for the IEEE-VIS)
        if (lines.Count() == 2) {
            // simple line:
            unsigned char col[3];
            data.Append(lines[0].PeekCoordinates(), 3 * sizeof(float));
            if (useTimeColour) {
                this->timeColour(times[indices[0] * 2], col[0], col[1], col[2]);
                data.Append(col, 3);
            }
            data.Append(lines[1].PeekCoordinates(), 3 * sizeof(float));
            if (useTimeColour) {
                this->timeColour(times[indices[1] * 2 + 1], col[0], col[1], col[2]);
                data.Append(col, 3);
            }

            unsigned int i[4] = {0, 0, 1, 1};
            index.Append(i, 4 * sizeof(unsigned int));

        } else {
            layout = useTimeColour ? geocalls::BezierCurvesListDataCall::DATALAYOUT_XYZR_F_RGB_B
                                   : geocalls::BezierCurvesListDataCall::DATALAYOUT_XYZR_F;
            bpp = useTimeColour ? (4 * 4 + 3 * 1) : (4 * 4);

            // segment length for radius mapping
            float maxLen = 1.0f;
            for (unsigned int i = 1; i < indices.Count(); i++) {
                float len = static_cast<float>(indices[i] - indices[i - 1]);
                len /= lines[i - 1].Distance(lines[i]);
                if (maxLen < len)
                    maxLen = len;
            }
            vislib::Array<float> radii(indices.Count() - 1, 1.0f);
            for (unsigned int i = 1; i < indices.Count(); i++) {
                float len = static_cast<float>(indices[i] - indices[i - 1]);
                len /= lines[i - 1].Distance(lines[i]);
                radii[i - 1] = 0.5f + 0.5f * len / maxLen;
            }

            // first curve
            unsigned char col[3];
            unsigned int idx = 0;
            float r = rad * radii[0];
            data.Append(lines[0].PeekCoordinates(), 3 * sizeof(float));
            data.Append(&r, sizeof(float));
            if (useTimeColour) {
                this->timeColour(times[indices[0] * 2], col[0], col[1], col[2]);
                data.Append(col, 3);
            }
            index.Append(&idx, sizeof(unsigned int));

            data.Append(lines[0].Interpolate(lines[1], 0.75).PeekCoordinates(), 3 * sizeof(float));
            data.Append(&r, sizeof(float));
            if (useTimeColour)
                data.Append(col, 3);
            idx++;
            index.Append(&idx, sizeof(unsigned int));

            // inner curves
            for (unsigned int i = 1; i < lines.Count() - 2; i++) {
                data.Append(lines[i].Interpolate(lines[i + 1], 0.25f).PeekCoordinates(), 3 * sizeof(float));
                r = rad * radii[i];
                data.Append(&r, sizeof(float));
                if (useTimeColour) {
                    this->timeColour(
                        times[indices[i] * 2] * 0.5f + times[indices[i] * 2 + 1] * 0.5f, col[0], col[1], col[2]);
                    data.Append(col, 3);
                }
                idx++;
                index.Append(&idx, sizeof(unsigned int));

                data.Append(lines[i].Interpolate(lines[i + 1], 0.5f).PeekCoordinates(), 3 * sizeof(float));
                data.Append(&r, sizeof(float));
                if (useTimeColour)
                    data.Append(col, 3);
                idx++;
                index.Append(&idx, sizeof(unsigned int));
                index.Append(&idx, sizeof(unsigned int)); // use this point twice

                data.Append(lines[i].Interpolate(lines[i + 1], 0.75f).PeekCoordinates(), 3 * sizeof(float));
                data.Append(&r, sizeof(float));
                if (useTimeColour)
                    data.Append(col, 3);
                idx++;
                index.Append(&idx, sizeof(unsigned int));
            }

            // last curve
            data.Append(lines[lines.Count() - 2].Interpolate(lines.Last(), 0.25f).PeekCoordinates(), 3 * sizeof(float));
            r = rad * radii.Last();
            data.Append(&r, sizeof(float));
            if (useTimeColour) {
                this->timeColour(times[indices.Last() * 2 + 1], col[0], col[1], col[2]);
                data.Append(col, 3);
            }
            idx++;
            index.Append(&idx, sizeof(unsigned int));
            data.Append(lines.Last().PeekCoordinates(), 3 * sizeof(float));
            data.Append(&r, sizeof(float));
            if (useTimeColour)
                data.Append(col, 3);
            idx++;
            index.Append(&idx, sizeof(unsigned int));
        }
    }

    if ((data.GetSize() != 0) && (index.GetSize() != 0)) {
        // copy the data to new flat arrays, so that the list object can overtake the memory
        unsigned char* dat = new unsigned char[data.GetSize()];
        ::memcpy(dat, data, data.GetSize());
        unsigned int* idx = new unsigned int[index.GetSize() / sizeof(unsigned int)];
        ::memcpy(idx, index, index.GetSize());

        // the list object now takes ownership of the data
        list->Set(layout, dat, data.GetSize() / bpp, true, idx, index.GetSize() / sizeof(unsigned int), true, rad, colR,
            colG, colB);

        dat = nullptr; // do not delete!
        idx = nullptr; // do not delete!
    } else {
        list->Set(geocalls::BezierCurvesListDataCall::DATALAYOUT_NONE, nullptr, 0, nullptr, 0);
    }
}


/*
 * SiffCSplineFitter::timeColour
 */
void SiffCSplineFitter::timeColour(float time, unsigned char& outR, unsigned char& outG, unsigned char& outB) {
    if (time < 0.0f)
        time = 0.0f;
    else if (time > 1.0f)
        time = 1.0f;
    time *= 3.0f;
    float r1, r2, g1, g2, b1, b2;

    if (time < 1.0f) {
        r1 = 1.0f;
        g1 = 0.0f;
        b1 = 0.0f;
        r2 = 0.0f;
        g2 = 1.0f;
        b2 = 0.0f;
    } else if (time < 2.0f) {
        r1 = 1.0f;
        g1 = 1.0f;
        b1 = 0.0f;
        r2 = -1.0f;
        g2 = 0.0f;
        b2 = 1.0f;
        time -= 1.0f;
    } else {
        r1 = 0.0f;
        g1 = 1.0f;
        b1 = 1.0f;
        r2 = 0.0f;
        g2 = -1.0f;
        b2 = 0.0f;
        time -= 2.0f;
    }

    outR = static_cast<unsigned char>(255.0f * (r1 + r2 * time));
    outG = static_cast<unsigned char>(255.0f * (g1 + g2 * time));
    outB = static_cast<unsigned char>(255.0f * (b1 + b2 * time));
}
} // namespace megamol::datatools
