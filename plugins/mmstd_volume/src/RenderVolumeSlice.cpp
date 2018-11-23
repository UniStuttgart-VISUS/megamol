/*
 * RenderVolumeSlice.cpp
 *
 * Copyright (C) 2012 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "RenderVolumeSlice.h"
#include "mmcore/CallVolumeData.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/FloatParam.h"
#include <climits>
#include <cfloat>
#include <cmath>
#include "mmcore/CoreInstance.h"
#include "mmcore/view/CallClipPlane.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "mmcore/view/CallRender3D.h"
#include "vislib/assert.h"

using namespace megamol;
using namespace megamol::stdplugin;
using namespace megamol::stdplugin::volume;


/*
 * RenderVolumeSlice::RenderVolumeSlice
 */
RenderVolumeSlice::RenderVolumeSlice(void) : Renderer3DModule(),
        getVolSlot("getVol", "The call for data"),
        getTFSlot("gettransferfunction", "The call for Transfer function"),
        getClipPlaneSlot("getclipplane", "The call for clipping plane"),
        attributeSlot("attr", "The attribute to show"),
        lowValSlot("low", "The low value"),
        highValSlot("high", "The high value") {

    this->getVolSlot.SetCompatibleCall<core::CallVolumeDataDescription>();
    this->MakeSlotAvailable(&this->getVolSlot);

    this->getTFSlot.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->getTFSlot);

    this->getClipPlaneSlot.SetCompatibleCall<core::view::CallClipPlaneDescription>();
    this->MakeSlotAvailable(&this->getClipPlaneSlot);

    this->attributeSlot << new core::param::StringParam("0");
    this->MakeSlotAvailable(&this->attributeSlot);

    this->lowValSlot << new core::param::FloatParam(0.0f);
    this->MakeSlotAvailable(&this->lowValSlot);

    this->highValSlot << new core::param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->highValSlot);

}


/*
 * RenderVolumeSlice::RenderVolumeSlice
 */
RenderVolumeSlice::~RenderVolumeSlice(void) {
    this->Release();
}


/*
 * RenderVolumeSlice::RenderVolumeSlice
 */
bool RenderVolumeSlice::create(void) {
    // intentionally empty
    return true;
}


/*
 * RenderVolumeSlice::RenderVolumeSlice
 */
bool RenderVolumeSlice::GetExtents(core::Call& call) {
    core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    core::CallVolumeData *c2 = this->getVolSlot.CallAs<core::CallVolumeData>();
    if ((c2 != NULL) && ((*c2)(1))) {
        cr->SetTimeFramesCount(c2->FrameCount());
        cr->AccessBoundingBoxes() = c2->AccessBoundingBoxes();
        cr->AccessBoundingBoxes().MakeScaledWorld(1.0f);

    } else {
        cr->SetTimeFramesCount(1);
        cr->AccessBoundingBoxes().Clear();
    }

    return true;
}


/*
 * RenderVolumeSlice::RenderVolumeSlice
 */
void RenderVolumeSlice::release(void) {
    // intentionally empty
}


/*
 * RenderVolumeSlice::RenderVolumeSlice
 */
bool RenderVolumeSlice::Render(core::Call& call) {
    core::view::CallRender3D *cr = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    // get volume data
    core::CallVolumeData *cvd = this->getVolSlot.CallAs<core::CallVolumeData>();
    if (cvd == NULL) return false;
    cvd->SetFrameID(static_cast<unsigned int>(cr->Time()));
    if (!(*cvd)(1)) return false;
    vislib::math::Cuboid<float> bbox = cvd->GetBoundingBoxes().ObjectSpaceBBox();
    cvd->SetFrameID(static_cast<unsigned int>(cr->Time()));
    if (!(*cvd)(0)) return false;

    float stepSize, ssy, ssz;
    stepSize = bbox.Width() / static_cast<float>(cvd->XSize());
    ssy = bbox.Height() / static_cast<float>(cvd->YSize());
    ssz = bbox.Depth() / static_cast<float>(cvd->ZSize());
    stepSize = fabs(stepSize + ssy + ssz) / 3.0f;

    // find the volumed attribute
    vislib::StringA attrName(this->attributeSlot.Param<core::param::StringParam>()->Value());
    unsigned int attrIdx = cvd->FindAttribute(attrName);
    if (attrIdx == UINT_MAX) {
        try {
            attrIdx = static_cast<unsigned int>(vislib::CharTraitsA::ParseInt(attrName));
        } catch(...) {
            return false;
        }
    }
    ASSERT(attrIdx != UINT_MAX);

    // get clip plane
    core::view::CallClipPlane *ccp = this->getClipPlaneSlot.CallAs<core::view::CallClipPlane>();
    if ((ccp == NULL) || (!(*ccp)())) return false;

    // get transfer function
    core::view::CallGetTransferFunction *cgtf = this->getTFSlot.CallAs<core::view::CallGetTransferFunction>();
    if ((cgtf == NULL) || (!(*cgtf)())) cgtf = NULL;

    unsigned int colTabSize;
    if (cgtf != NULL) {
        ::glDisable(GL_TEXTURE_2D);
        ::glEnable(GL_TEXTURE_1D);
        ::glEnable(GL_COLOR_MATERIAL);
        ::glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
        ::glBindTexture(GL_TEXTURE_1D, cgtf->OpenGLTexture());
        colTabSize = cgtf->TextureSize();
    } else {
        ::glDisable(GL_TEXTURE_1D);
        ::glBindTexture(GL_TEXTURE_1D, 0);
        colTabSize = 2;
    }

    float minVal = this->lowValSlot.Param<core::param::FloatParam>()->Value();
    float maxVal = this->highValSlot.Param<core::param::FloatParam>()->Value();

    vislib::math::Point<float, 3> ps[12];
    bool psv[12];

    {
        vislib::math::Plane<float> minX(bbox.GetLeftBottomBack(), vislib::math::Vector<float, 3>(-1.0f, 0.0f, 0.0f));
        vislib::math::Plane<float> minY(bbox.GetLeftBottomBack(), vislib::math::Vector<float, 3>(0.0f, -1.0f, 0.0f));
        vislib::math::Plane<float> minZ(bbox.GetLeftBottomBack(), vislib::math::Vector<float, 3>(0.0f, 0.0f, -1.0f));
        vislib::math::Plane<float> maxX(bbox.GetRightTopFront(), vislib::math::Vector<float, 3>(1.0f, 0.0f, 0.0f));
        vislib::math::Plane<float> maxY(bbox.GetRightTopFront(), vislib::math::Vector<float, 3>(0.0f, 1.0f, 0.0f));
        vislib::math::Plane<float> maxZ(bbox.GetRightTopFront(), vislib::math::Vector<float, 3>(0.0f, 0.0f, 1.0f));

        psv[0] = ccp->GetPlane().CalcIntersectionPoint(minX, minY, ps[0]);
        psv[1] = ccp->GetPlane().CalcIntersectionPoint(minX, minZ, ps[1]);
        psv[2] = ccp->GetPlane().CalcIntersectionPoint(minX, maxY, ps[2]);
        psv[3] = ccp->GetPlane().CalcIntersectionPoint(minX, maxZ, ps[3]);
        psv[4] = ccp->GetPlane().CalcIntersectionPoint(maxX, minY, ps[4]);
        psv[5] = ccp->GetPlane().CalcIntersectionPoint(maxX, minZ, ps[5]);
        psv[6] = ccp->GetPlane().CalcIntersectionPoint(maxX, maxY, ps[6]);
        psv[7] = ccp->GetPlane().CalcIntersectionPoint(maxX, maxZ, ps[7]);
        psv[8] = ccp->GetPlane().CalcIntersectionPoint(minY, minZ, ps[8]);
        psv[9] = ccp->GetPlane().CalcIntersectionPoint(minY, maxZ, ps[9]);
        psv[10] = ccp->GetPlane().CalcIntersectionPoint(maxY, minZ, ps[10]);
        psv[11] = ccp->GetPlane().CalcIntersectionPoint(maxY, maxZ, ps[11]);
    }

    unsigned int cnt = 0;
    vislib::math::Vector<float, 3> v;
    for (int i = 0; i < 12; i++) {
        if (!psv[i]) continue;
        v += vislib::math::Vector<float, 3>(ps[i]);
        cnt++;
    }
    if (cnt < 3) return false;
    v /= static_cast<float>(cnt);

    vislib::math::Point<float, 3> orig(v.PeekComponents());

    vislib::math::Vector<float, 3> x, y, z;
    ccp->CalcPlaneSystem(x, y, z);

    float minX, maxX, minY, maxY;
    minX = minY = 0.0f;
    maxX = maxY = 0.0f;
    for (int i = 0; i < 12; i++) {
        if (!psv[i]) continue;
        v = ps[i] - orig;
        float xc = v.Dot(x);
        float yc = v.Dot(y);
        if (minX > xc) minX = xc;
        if (maxX < xc) maxX = xc;
        if (minY > yc) minY = yc;
        if (maxY < yc) maxY = yc;
    }
    // loop ranges
    int minXi = static_cast<int>(minX / stepSize);
    int minYi = static_cast<int>(minY / stepSize);
    int maxXi = static_cast<int>(maxX / stepSize);
    int maxYi = static_cast<int>(maxY / stepSize);
    int w = (maxXi - minXi) + 1;
    int h = (maxYi - minYi) + 1;

    vislib::math::Vector<float, 3> *pos = new vislib::math::Vector<float, 3>[w * h];
    float *val = new float[w * h];

    for (int ix = minXi; ix <= maxXi; ix++) {
        int io = ix - minXi;
        for (int iy = minYi; iy <= maxYi; iy++) {
            int idx = (iy - minYi) * w + io;
            pos[idx] = orig;
            v = x;
            v *= static_cast<float>(ix) * stepSize;
            pos[idx] += v;
            v = y;
            v *= static_cast<float>(iy) * stepSize;
            pos[idx] += v;

            int vix = static_cast<int>(0.5f + (pos[idx].X() - bbox.Left()) * static_cast<float>(cvd->XSize()) / bbox.Width());
            int viy = static_cast<int>(0.5f + (pos[idx].Y() - bbox.Bottom()) * static_cast<float>(cvd->YSize()) / bbox.Height());
            int viz = static_cast<int>(0.5f + (pos[idx].Z() - bbox.Back()) * static_cast<float>(cvd->ZSize()) / bbox.Depth());
            if (vix < 0) vix = 0;
            if (vix >= static_cast<int>(cvd->XSize())) vix = cvd->XSize() - 1;
            if (viy < 0) viy = 0;
            if (viy >= static_cast<int>(cvd->YSize())) viy = cvd->YSize() - 1;
            if (viz < 0) viz = 0;
            if (viz >= static_cast<int>(cvd->ZSize())) viz = cvd->ZSize() - 1;

            val[idx] = cvd->Attribute(attrIdx).Floats()[vix + cvd->XSize() * (viy + cvd->YSize() * viz)];
            val[idx] = (val[idx] - minVal) / (maxVal - minVal);
            if (val[idx] < 0.0f) val[idx] = 0.0f;
            if (val[idx] > 1.0f) val[idx] = 1.0f;
        }
    }

    ::glNormal3fv(z.PeekComponents());

    ::glDisable(GL_CULL_FACE);
    GLint lmts = 1;
    ::glLightModeliv(GL_LIGHT_MODEL_TWO_SIDE, &lmts);
    ::glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    ::glColor3ub(255, 255, 255);
    ::glTexCoord1f(0.0f);

    int idx[4];
    for (int ix = minXi; ix < maxXi; ix++) {
        //::glBegin(GL_QUAD_STRIP);
        for (int iy = minYi; iy < maxYi; iy++) {
            idx[0] = (ix - minXi) + w * (iy - minYi);
            idx[1] = idx[0] + 1;
            idx[2] = idx[1] + w;
            idx[3] = idx[0] + w;

            float vf = (val[idx[0]] + val[idx[1]] + val[idx[2]] + val[idx[3]]) * 0.25f;
            v = pos[idx[0]];
            v += pos[idx[1]];
            v += pos[idx[2]];
            v += pos[idx[3]];
            v *= 0.25f;

            ::glBegin(GL_TRIANGLE_FAN);
            if (cgtf == NULL) ::glColor3f(vf, vf, vf); else ::glTexCoord1f(vf);
            ::glVertex3fv(v.PeekComponents());
            for (int ii = 0; ii <= 4; ii++) {
                vf = val[idx[ii % 4]];
                if (cgtf == NULL) ::glColor3f(vf, vf, vf); else ::glTexCoord1f(vf);
                ::glVertex3fv(pos[idx[ii % 4]].PeekComponents());
            }
            ::glEnd();

/*            for (int ii = 0; ii < 4; ii++) {
                if (cgtf == NULL) {
                    ::glColor3f(val[idx], val[idx], val[idx]);
                } else {
                    ::glTexCoord1f(val[idx]);
                }
                ::glVertex3fv(pos[idx].PeekComponents());
                if (ii % 2) {
                    idx += w - 1;
                } else {
                    idx++;
                }
            }*/
        }
        //::glEnd();
    }

    delete[] pos;
    delete[] val;

    glDisable(GL_TEXTURE_1D);
    lmts = 0;
    glLightModeliv(GL_LIGHT_MODEL_TWO_SIDE, &lmts);

    return true;
}
