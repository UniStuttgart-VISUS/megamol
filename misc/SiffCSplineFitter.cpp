/*
 * SiffCSplineFitter.cpp
 *
 * Copyright (C) 2010 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "SiffCSplineFitter.h"
#include "BezierDataCall.h"
#include "moldyn/MultiParticleDataCall.h"
//#include "param/FilePathParam.h"
#include "param/EnumParam.h"
#include "vislib/BezierCurve.h"
#include "vislib/Log.h"
#include "vislib/mathfunctions.h"
#include "vislib/Point.h"
#include "vislib/ShallowPoint.h"
/*
#include <climits>
#include "param/BoolParam.h"
#include "param/FloatParam.h"
#include "utility/ColourParser.h"
#include "vislib/forceinline.h"
#include "vislib/StringTokeniser.h"
#include "vislib/sysfunctions.h"
#include "vislib/SystemMessage.h"
#include "vislib/mathfunctions.h"
#include "vislib/MemmappedFile.h"
#include "vislib/Vector.h"
*/

using namespace megamol::core;


/*
 * misc::SiffCSplineFitter::SiffCSplineFitter
 */
misc::SiffCSplineFitter::SiffCSplineFitter(void) : Module(),
        getDataSlot("getdata", "The slot exposing the loaded data"),
        inDataSlot("indata", "The slot for fetching siff data"),
        colourMapSlot("colourMapping", "The parameter controlling the colour mapping"),
        minX(-1.0f), minY(-1.0f), minZ(-1.0f), maxX(1.0f), maxY(1.0f),
        maxZ(1.0f), curves(), datahash(0), inhash(0) {

    param::EnumParam *ep = new param::EnumParam(1);
    ep->SetTypePair(0, "site colour");
    ep->SetTypePair(1, "time");
    this->colourMapSlot << ep;
    this->MakeSlotAvailable(&this->colourMapSlot);

    this->getDataSlot.SetCallback("BezierDataCall", "GetData",
        &SiffCSplineFitter::getDataCallback);
    this->getDataSlot.SetCallback("BezierDataCall", "GetExtent",
        &SiffCSplineFitter::getExtentCallback);
    this->MakeSlotAvailable(&this->getDataSlot);

    this->inDataSlot.SetCompatibleCall<moldyn::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->inDataSlot);
}


/*
 * misc::SiffCSplineFitter::~SiffCSplineFitter
 */
misc::SiffCSplineFitter::~SiffCSplineFitter(void) {
    this->Release();
}


/*
 * misc::SiffCSplineFitter::create
 */
bool misc::SiffCSplineFitter::create(void) {
    // intentionally empty
    return true;
}


/*
 * misc::SiffCSplineFitter::release
 */
void misc::SiffCSplineFitter::release(void) {
    this->curves.Clear();
    this->minX = this->minY = this->minZ = -.0f;
    this->maxX = this->maxY = this->maxZ = 1.0f;
    this->datahash = 0;
}


/*
 * misc::SiffCSplineFitter::getDataCallback
 */
bool misc::SiffCSplineFitter::getDataCallback(Call& caller) {
    BezierDataCall *bdc = dynamic_cast<BezierDataCall*>(&caller);
    if (bdc == NULL) return false;

    this->assertData();

    bdc->SetData(static_cast<unsigned int>(this->curves.Count()),
        this->curves.PeekElements());
    bdc->SetDataHash(this->datahash);
    bdc->SetExtent(1 /* static data */,
        this->minX, this->minY, this->minZ,
        this->maxX, this->maxY, this->maxZ);
    bdc->SetFrameID(0);
    bdc->SetUnlocker(NULL);

    return true;
}


/*
 * misc::SiffCSplineFitter::getExtentCallback
 */
bool misc::SiffCSplineFitter::getExtentCallback(Call& caller) {
    BezierDataCall *bdc = dynamic_cast<BezierDataCall*>(&caller);
    if (bdc == NULL) return false;

    this->assertData();

    bdc->SetDataHash(this->datahash);
    bdc->SetExtent(1 /* static data */,
        this->minX, this->minY, this->minZ,
        this->maxX, this->maxY, this->maxZ);

    return true;
}


/*
 * misc::SiffCSplineFitter::assertData
 */
void misc::SiffCSplineFitter::assertData(void) {
    using vislib::sys::Log;

    moldyn::MultiParticleDataCall *mpdc = this->inDataSlot.CallAs<moldyn::MultiParticleDataCall>();
    if (mpdc == NULL) return;

    if (!(*mpdc)(1)) return;
    vislib::math::Cuboid<float> bbox = mpdc->AccessBoundingBoxes().ClipBox();

    if (!(*mpdc)(0)) return;
    if (!this->colourMapSlot.IsDirty() && (mpdc->DataHash() == this->inhash)) return; // data did not change

    this->colourMapSlot.ResetDirty();
    this->inhash = mpdc->DataHash();
    this->datahash++;
    this->curves.Clear();
    this->minX = bbox.GetLeft();
    this->minY = bbox.GetBottom();
    this->minZ = bbox.GetBack();
    this->maxX = bbox.GetRight();
    this->maxY = bbox.GetTop();
    this->maxZ = bbox.GetFront();

    if (mpdc->GetParticleListCount() < 1) return;
    if (mpdc->GetParticleListCount() != 1) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_WARN,
            "Spline fitter only supports single list data ATM");
    }
    unsigned int cnt = static_cast<unsigned int>(mpdc->AccessParticles(0).GetCount());
    if (cnt == 0) return;
    const unsigned char *vdata = static_cast<const unsigned char*>(mpdc->AccessParticles(0).GetVertexData());
    unsigned int vstride = mpdc->AccessParticles(0).GetVertexDataStride();
    const unsigned char *cdata = static_cast<const unsigned char*>(mpdc->AccessParticles(0).GetColourData());
    unsigned int cstride = mpdc->AccessParticles(0).GetColourDataStride();
    if (mpdc->AccessParticles(0).GetColourDataType() != moldyn::MultiParticleDataCall::Particles::COLDATA_UINT8_RGB) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_WARN,
            "Spline fitter only supports colour data UINT8_RGB");
        return; // without colour we cannot detect the frames!
    } else if (cstride < 3) cstride = 3;
    if (mpdc->AccessParticles(0).GetVertexDataType() != moldyn::MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR) {
        Log::DefaultLog.WriteMsg(Log::LEVEL_ERROR,
            "Spline fitter only supports vertex data FLOAT_XYZR");
        return;
    } else if (vstride < 4 * sizeof(float)) vstride = 4 * sizeof(float);

    // count sites and frames
    // sites are always in same order in each frame
    const int ccdtestsize = 10; // sequence of 10 sites as frame detector
    const unsigned char *ccd = cdata + cstride * ccdtestsize;
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
    float *pos = new float[frameCnt * 3];
    float *times = new float[frameCnt * 2];

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
            times[j * 2] = times[j * 2 + 1] = ((frameCnt > 1) ? (static_cast<float>(j) / static_cast<float>(frameCnt - 1)) : 0.0f);
        }

        this->addSpline(pos, times, frameCnt, rad, colR, colG, colB);

    }

    delete[] pos;
    delete[] times;
}


/*
 * misc::SiffCSplineFitter::addSpline
 */
void misc::SiffCSplineFitter::addSpline(float *pos, float *times, unsigned int cnt, float rad, unsigned char colR, unsigned char colG, unsigned char colB) {
    typedef vislib::math::ShallowPoint<float, 3> ShallowPoint;
    typedef vislib::math::Point<float, 3> Point;
    typedef vislib::math::Vector<float, 3> Vector;
    typedef vislib::math::BezierCurve<misc::BezierDataCall::BezierPoint, 3> BezierCurve;

    const bool useTimeColour = (this->colourMapSlot.Param<param::EnumParam>()->Value() == 1);

    if (cnt == 0) return;
    int p = 0;
    BezierCurve curve;
    for (unsigned int i = 1; i < cnt; i++) {
        float d = ShallowPoint(pos + p * 3).Distance(ShallowPoint(pos + i * 3));
        if (d > 0.25f * rad) {
            p++;
            pos[p * 3 + 0] = pos[i * 3 + 0];
            pos[p * 3 + 1] = pos[i * 3 + 1];
            pos[p * 3 + 2] = pos[i * 3 + 2];
            times[p * 2 + 0] = times[i * 2 + 0];
            times[p * 2 + 1] = times[i * 2 + 1];
        } else {
            times[p * 2 + 1] = times[i * 2 + 1];
        }
    }
    ////if (1 + p < cnt) {
    ////    printf("Reduced from %u to %d\n", cnt, p + 1);
    ////}
    cnt = static_cast<unsigned int>(1 + p);

    // split cyclic boundary jumps
    unsigned int s = 0;
    bool splitted = false;

    const float bminB = 0.15f;
    const float bmaxB = 1.0f - bminB;
    float bminX = this->maxX * bminB + this->minX * bmaxB;
    float bmaxX = this->maxX * bmaxB + this->minX * bminB;
    float bminY = this->maxY * bminB + this->minY * bmaxB;
    float bmaxY = this->maxY * bmaxB + this->minY * bminB;
    float bminZ = this->maxZ * bminB + this->minZ * bmaxB;
    float bmaxZ = this->maxZ * bmaxB + this->minZ * bminB;

    for (unsigned int i = 1; i < cnt; i++) {
        ShallowPoint p1(pos + (i - 1) * 3);
        ShallowPoint p2(pos + i * 3);
        float d = p1.Distance(p2);
        if (d < 2.0f * rad) continue;
        // test if is a cb-jump

        if (       ((p1.X() < bminX) && (p2.X() > bmaxX)) // cb-jump in X
                || ((p2.X() < bminX) && (p1.X() > bmaxX)) // cb-jump in X
                || ((p1.Y() < bminY) && (p2.Y() > bmaxY)) // cb-jump in Y
                || ((p2.Y() < bminY) && (p1.Y() > bmaxY)) // cb-jump in Y
                || ((p1.Z() < bminZ) && (p2.Z() > bmaxZ)) // cb-jump in Z
                || ((p2.Z() < bminZ) && (p1.Z() > bmaxZ)) ) { // cb-jump in Z

            this->addSpline(pos + s * 3, times + s * 2, i - s, rad, colR, colG, colB);
            s = i;
            splitted = true;
        }
    }
    if (splitted) {
        pos += s * 3;
        times += s * 2;
        cnt -= s;
    }

    if (cnt == 1) {
        // super trivial nonsense
        if (useTimeColour) this->timeColour(times[0], colR, colG, colB);
        curve[0].Set(pos[0] + 0.05f * rad, pos[1], pos[2], rad, colR, colG, colB);
        if (useTimeColour) this->timeColour((times[0] + times[1]) * 0.5f, colR, colG, colB);
        curve[1].Set(pos[0], pos[1], pos[2], rad, colR, colG, colB);
        curve[2].Set(pos[0], pos[1], pos[2], rad, colR, colG, colB);
        if (useTimeColour) this->timeColour(times[1], colR, colG, colB);
        curve[3].Set(pos[0] - 0.05f * rad, pos[1], pos[2], rad, colR, colG, colB);
        this->curves.Add(curve);
        return;
    }

    if (cnt <= 4) {
        // trivial nonsense
        unsigned char r1, r2, r3, r4, g1, g2, g3, g4, b1, b2, b3, b4;
        r1 = r2 = r3 = r4 = colR;
        g1 = g2 = g3 = g4 = colG;
        b1 = b2 = b3 = b4 = colB;

        ShallowPoint p1(pos);
        if (useTimeColour) this->timeColour(times[0], r1, g1, b1);
        Point p2;
        Point p3;
        ShallowPoint p4(pos + (cnt - 1) * 3);
        if (useTimeColour) this->timeColour(times[(cnt - 1) * 2 + 1], r4, g4, b4);
        if (cnt == 2) {
            p2 = p1.Interpolate(p4, 0.33f);
            if (useTimeColour) this->timeColour(times[1] * 0.333f + times[2] * 0.667f, r2, g2, b2);
            p3 = p1.Interpolate(p4, 0.66f);
            if (useTimeColour) this->timeColour(times[1] * 0.667f + times[2] * 0.333f, r3, g3, b3);
        } else if (cnt == 3) {
            p2.Set(pos[3], pos[4], pos[5]);
            if (useTimeColour) this->timeColour(times[1] * 0.333f + (times[2] + times[3]) * 0.5f * 0.667f, r2, g2, b2);
            p3 = p2.Interpolate(p4, 0.33f);
            if (useTimeColour) this->timeColour(times[4] * 0.333f + (times[2] + times[3]) * 0.5f * 0.667f, r3, g3, b3);
            p2 = p2.Interpolate(p1, 0.33f);
        } else if (cnt == 4) {
            p2.Set(pos[3], pos[4], pos[5]);
            if (useTimeColour) this->timeColour(times[2] * 0.667f + times[3] * 0.333f, r2, g2, b2);
            p3.Set(pos[6], pos[7], pos[8]);
            if (useTimeColour) this->timeColour(times[4] * 0.333f + times[5] * 0.667f, r3, g3, b3);
        }

        curve[0].Set(p1, rad, r1, g1, b1);
        curve[1].Set(p2, rad, r2, g2, b2);
        curve[2].Set(p3, rad, r3, g3, b3);
        curve[3].Set(p4, rad, r4, g4, b4);
        this->curves.Add(curve);
        return;
    }

    // looping and/or fitting
    // first fit a polyline
    vislib::Array<Point> lines;
    vislib::Array<unsigned int> indices;

    indices.Add(0);
    lines.Add(ShallowPoint(pos));
    indices.Add(cnt - 1);
    lines.Add(ShallowPoint(pos + (cnt - 1) * 3));

    const float distEps = rad * 0.75f;

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
                indices.Insert(l, maxP);
                lines.Insert(l, ShallowPoint(pos + maxP * 3));
                l++;
                refined = true;
            }

        }
    }

    // approx lines though curved (bullshit but good enough for the IEEE-VIS)
    if (lines.Count() == 2) {
        if (useTimeColour) this->timeColour(times[indices[0] * 2], colR, colG, colB);
        curve[0].Set(lines[0], rad, colR, colG, colB);
        if (useTimeColour) this->timeColour(
            (times[indices[0] * 2] * 0.667f + times[indices[0] * 2 + 1] * 0.333f) * 0.667f +
            (times[indices[1] * 2] * 0.667f + times[indices[1] * 2 + 1] * 0.333f) * 0.333f
            , colR, colG, colB);
        curve[1].Set(lines[0].Interpolate(lines[1], 0.333f), rad, colR, colG, colB);
        if (useTimeColour) this->timeColour(
            (times[indices[0] * 2] * 0.333f + times[indices[0] * 2 + 1] * 0.667f) * 0.333f +
            (times[indices[1] * 2] * 0.333f + times[indices[1] * 2 + 1] * 0.667f) * 0.667f
            , colR, colG, colB);
        curve[2].Set(lines[0].Interpolate(lines[1], 0.667f), rad, colR, colG, colB);
        if (useTimeColour) this->timeColour(times[indices[1] * 2 + 1], colR, colG, colB);
        curve[3].Set(lines[1], rad, colR, colG, colB);
        this->curves.Add(curve);
        return;
    }

    float maxLen = 1.0f;
    for (unsigned int i = 1; i < indices.Count(); i++) {
        float len = static_cast<float>(indices[i] - indices[i - 1]);
        len /= lines[i - 1].Distance(lines[i]);
        if (maxLen < len) maxLen = len;
    }
    vislib::Array<float> radii(indices.Count() - 1, 1.0f);
    for (unsigned int i = 1; i < indices.Count(); i++) {
        float len = static_cast<float>(indices[i] - indices[i - 1]);
        len /= lines[i - 1].Distance(lines[i]);
        radii[i - 1] = len / maxLen;
    }

    // first curve
    if (useTimeColour) this->timeColour(times[indices[0] * 2], colR, colG, colB);
    curve[0].Set(lines[0], rad, colR, colG, colB);
    if (useTimeColour) this->timeColour(
        (times[indices[0] * 2] * 0.333f + times[indices[0] * 2 + 1] * 0.667f) * 0.333f +
        (times[indices[1] * 2] * 0.333f + times[indices[1] * 2 + 1] * 0.667f) * 0.667f
        , colR, colG, colB);
    curve[1].Set(lines[0].Interpolate(lines[1], 0.667f), rad * radii[0], colR, colG, colB);

    // inner curves
    for (unsigned int i = 2; i < lines.Count() - 1; i++) {
        if (useTimeColour) this->timeColour(
            (times[indices[i - 1] * 2] * 0.75f + times[indices[i - 1] * 2 + 1] * 0.25f) * 0.75f +
            (times[indices[i] * 2] * 0.75f + times[indices[i] * 2 + 1] * 0.25f) * 0.25f
            , colR, colG, colB);
        curve[2].Set(lines[i - 1].Interpolate(lines[i], 0.25f), rad * radii[i - 1], colR, colG, colB);
        if (useTimeColour) this->timeColour(
            (times[indices[i - 1] * 2] * 0.5f + times[indices[i - 1] * 2 + 1] * 0.5f) * 0.5f +
            (times[indices[i] * 2] * 0.5f + times[indices[i] * 2 + 1] * 0.5f) * 0.5f
            , colR, colG, colB);
        curve[3].Set(lines[i - 1].Interpolate(lines[i], 0.5f), rad * 0.5f * (radii[i - 1] + radii[i]), colR, colG, colB);
        this->curves.Add(curve);
        curve[0] = curve[3];
        if (useTimeColour) this->timeColour(
            (times[indices[i - 1] * 2] * 0.25f + times[indices[i - 1] * 2 + 1] * 0.75f) * 0.25f +
            (times[indices[i] * 2] * 0.25f + times[indices[i] * 2 + 1] * 0.75f) * 0.75f
            , colR, colG, colB);
        curve[1].Set(lines[i - 1].Interpolate(lines[i], 0.75f), rad * radii[i], colR, colG, colB);
    }

    // last curve
    if (useTimeColour) this->timeColour(
        (times[indices[lines.Count() - 2] * 2] * 0.667f + times[indices[lines.Count() - 2] * 2 + 1] * 0.333f) * 0.667f +
        (times[indices[lines.Count() - 1] * 2] * 0.667f + times[indices[lines.Count() - 1] * 2 + 1] * 0.333f) * 0.333f
        , colR, colG, colB);
    curve[2].Set(lines[lines.Count() - 2].Interpolate(lines[lines.Count() - 1], 0.333f), rad * radii[radii.Count() - 1], colR, colG, colB);
    curve[3].Set(lines[lines.Count() - 1], rad, colR, colG, colB);
    this->curves.Add(curve);

}


/*
 * misc::SiffCSplineFitter::timeColour
 */
void misc::SiffCSplineFitter::timeColour(float time, unsigned char &outR, unsigned char &outG, unsigned char &outB) {
    if (time < 0.0f) time = 0.0f;
    else if (time > 1.0f) time = 1.0f;
    time *= 3.0f;
    float r1, r2, g1, g2, b1, b2;

    if (time < 1.0f) {
        r1 = 1.0f; g1 = 0.0f; b1 = 0.0f;
        r2 = 0.0f; g2 = 1.0f; b2 = 0.0f;
    } else if (time < 2.0f) {
        r1 = 1.0f; g1 = 1.0f; b1 = 0.0f;
        r2 = -1.0f; g2 = 0.0f; b2 = 1.0f;
        time -= 1.0f;
    } else {
        r1 = 0.0f; g1 = 1.0f; b1 = 1.0f;
        r2 = 0.0f; g2 = -1.0f; b2 = 0.0f;
        time -= 2.0f;
    }

    outR = static_cast<unsigned char>(255.0f * (r1 + r2 * time));
    outG = static_cast<unsigned char>(255.0f * (g1 + g2 * time));
    outB = static_cast<unsigned char>(255.0f * (b1 + b2 * time));
}
