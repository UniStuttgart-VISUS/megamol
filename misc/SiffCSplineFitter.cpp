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
#include "param/FilePathParam.h"
#include "vislib/BezierCurve.h"
#include "vislib/Log.h"
#include "vislib/mathfunctions.h"
#include "vislib/Point.h"
#include "vislib/ShallowPoint.h"
/*
#include <climits>
#include "param/BoolParam.h"
#include "param/EnumParam.h"
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
        minX(-1.0f), minY(-1.0f), minZ(-1.0f), maxX(1.0f), maxY(1.0f),
        maxZ(1.0f), curves(), datahash(0), inhash(0) {

    //this->filenameSlot << new param::FilePathParam("");
    //this->MakeSlotAvailable(&this->filenameSlot);

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
    if (mpdc->DataHash() == this->inhash) return; // data did not change

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
        }

        this->addSpline(pos, frameCnt, rad, colR, colG, colB);

    }

    delete[] pos;
}


/*
 * misc::SiffCSplineFitter::addSpline
 */
void misc::SiffCSplineFitter::addSpline(float *pos, unsigned int cnt, float rad, unsigned char colR, unsigned char colG, unsigned char colB) {
    if (cnt == 0) return;
    int p = 0;
    for (unsigned int i = 1; i < cnt; i++) {
        float d = vislib::math::ShallowPoint<float, 3>(pos + p * 3).Distance(
            vislib::math::ShallowPoint<float, 3>(pos + i * 3));
        if (d > 0.5f * rad) {
            p++;
            pos[p * 3 + 0] = pos[i * 3 + 0];
            pos[p * 3 + 1] = pos[i * 3 + 1];
            pos[p * 3 + 2] = pos[i * 3 + 2];
        }
    }
    ////if (1 + p < cnt) {
    ////    printf("Reduced from %u to %d\n", cnt, p + 1);
    ////}
    cnt = static_cast<unsigned int>(1 + p);

    // split at jumps
    unsigned int s = 0;
    bool splitted = false;
    for (unsigned int i = 1; i < cnt; i++) {
        float d = vislib::math::ShallowPoint<float, 3>(pos + (i - 1) * 3).Distance(
            vislib::math::ShallowPoint<float, 3>(pos + i * 3));
        if (d > 2.0f * rad) {
            this->addSpline(pos + s * 3, i - s, rad, colR, colG, colB);
            s = i;
            splitted = true;
        }
    }
    if (splitted) {
        pos += s * 3;
        cnt -= s;
    }

    if (cnt == 1) {
        // super trivial nonsense
        vislib::math::BezierCurve<misc::BezierDataCall::BezierPoint, 3> curve;
        curve.ControlPoint(0).Set(pos[0] + 0.25f * rad, pos[1], pos[2], rad, colR, colG, colB);
        curve.ControlPoint(1).Set(pos[0], pos[1], pos[2], rad, colR, colG, colB);
        curve.ControlPoint(2).Set(pos[0], pos[1], pos[2], rad, colR, colG, colB);
        curve.ControlPoint(3).Set(pos[0] - 0.25f * rad, pos[1], pos[2], rad, colR, colG, colB);
        this->curves.Add(curve);
        return;
    }

    if (cnt <= 4) {
        // trivial nonsense
        vislib::math::ShallowPoint<float, 3> p1(pos);
        vislib::math::Point<float, 3> p2;
        vislib::math::Point<float, 3> p3;
        vislib::math::ShallowPoint<float, 3> p4(pos + (cnt - 1) * 3);
        if (cnt == 2) {
            p2 = p1.Interpolate(p4, 0.33f);
            p3 = p1.Interpolate(p4, 0.66f);
        } else if (cnt == 3) {
            p2.Set(pos[3], pos[4], pos[5]);
            p3 = p2.Interpolate(p4, 0.33f);
            p2 = p2.Interpolate(p1, 0.33f);
        } else if (cnt == 4) {
            p2.Set(pos[3], pos[4], pos[5]);
            p3.Set(pos[6], pos[7], pos[8]);
        }

        vislib::math::BezierCurve<misc::BezierDataCall::BezierPoint, 3> curve;
        curve.ControlPoint(0).Set(p1[0], p1[1], p1[2], rad, colR, colG, colB);
        curve.ControlPoint(1).Set(p2[0], p2[1], p2[2], rad, colR, colG, colB);
        curve.ControlPoint(2).Set(p3[0], p3[1], p3[2], rad, colR, colG, colB);
        curve.ControlPoint(3).Set(p4[0], p4[1], p4[2], rad, colR, colG, colB);
        this->curves.Add(curve);
        return;
    }

    // looping and/or fitting
    // first fit a polyline
    vislib::Array<vislib::math::Point<float, 3> >lines;
    lines.Add(vislib::math::Point<float, 3>(pos[0], pos[1], pos[2]));
    lines.Add(vislib::math::Point<float, 3>(pos[(cnt - 1) * 3], pos[(cnt - 1) * 3 + 1], pos[(cnt - 1) * 3 + 2]));

    // approx lines though curved (bullshit but good enough for the IEEE-VIS)
    if (lines.Count() == 2) {
        vislib::math::BezierCurve<misc::BezierDataCall::BezierPoint, 3> curve;
        curve.ControlPoint(0).Set(lines[0][0], lines[0][1], lines[0][2], rad, colR, colG, colB);
        curve.ControlPoint(1).Set(lines[0][0] * 0.667f + lines[1][0] * 0.333f, lines[0][1] * 0.667f + lines[1][1] * 0.333f, lines[0][2] * 0.667f + lines[1][2] * 0.333f, rad, colR, colG, colB);
        curve.ControlPoint(1).Set(lines[0][0] * 0.333f + lines[1][0] * 0.667f, lines[0][1] * 0.333f + lines[1][1] * 0.667f, lines[0][2] * 0.333f + lines[1][2] * 0.667f, rad, colR, colG, colB);
        curve.ControlPoint(3).Set(lines[1][0], lines[1][1], lines[1][2], rad, colR, colG, colB);
        this->curves.Add(curve);
    }

    //vislib::Array<float> dist(cnt - 1, 0.0f);
    //for (unsigned int i = 0; i < cnt - 1; i++) {
    //    dist[i] = vislib::math::ShallowPoint<float, 3>(pos + i * 3).Distance(
    //        vislib::math::ShallowPoint<float, 3>(pos + (i + 1) * 3));
    //}
    //dist.Sort(vislib::math::Compare); // distances between positions ascending

    // compress and split
    // split whenever the spline jumps more than the media distance
    //  possibly rejoin later ift jump is not a cyclic boundery condition
    // compress line whenever a position change is less then 

    //float *p1 = pos;
    //float *p2 = pos + (3 * ((cnt / 3) - 1));
    //float *p3 = pos + (3 * ((2 * cnt / 3) - 1));
    //float *p4 = pos + (3 * (cnt - 1));

    //vislib::math::BezierCurve<misc::BezierDataCall::BezierPoint, 3> curve;
    //curve.ControlPoint(0).Set(p1[0], p1[1], p1[2], rad, colR, colG, colB);
    //curve.ControlPoint(1).Set(p2[0], p2[1], p2[2], rad, colR, colG, colB);
    //curve.ControlPoint(2).Set(p3[0], p3[1], p3[2], rad, colR, colG, colB);
    //curve.ControlPoint(3).Set(p4[0], p4[1], p4[2], rad, colR, colG, colB);
    //this->curves.Add(curve);

}
