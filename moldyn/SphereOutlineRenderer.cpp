/*
 * SphereOutlineRenderer.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#define USE_MATH_DEFINES
#include "SphereOutlineRenderer.h"
#include "MultiParticleDataCall.h"
#include "CoreInstance.h"
#include "view/CallRender3D.h"
#include <GL/gl.h>
#include "vislib/assert.h"
#include "vislib/mathfunctions.h"
#include "vislib/Vector.h"
#include "vislib/Quaternion.h"
#include "vislib/ShallowPoint.h"
#include "vislib/ShallowVector.h"
#include <cmath>

using namespace megamol::core;


/*
 * moldyn::SphereOutlineRenderer::SphereOutlineRenderer
 */
moldyn::SphereOutlineRenderer::SphereOutlineRenderer(void) : Renderer3DModule(),
        getDataSlot("getdata", "Connects to the data source") {

    this->getDataSlot.SetCompatibleCall<moldyn::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

}


/*
 * moldyn::SphereOutlineRenderer::~SphereOutlineRenderer
 */
moldyn::SphereOutlineRenderer::~SphereOutlineRenderer(void) {
    this->Release();
}


/*
 * moldyn::SphereOutlineRenderer::create
 */
bool moldyn::SphereOutlineRenderer::create(void) {

    // TODO: Implement

    return true;
}


/*
 * moldyn::SphereOutlineRenderer::GetCapabilities
 */
bool moldyn::SphereOutlineRenderer::GetCapabilities(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    cr->SetCapabilities(
        view::CallRender3D::CAP_RENDER
        | view::CallRender3D::CAP_ANIMATION
        );

    return true;
}


/*
 * moldyn::SphereOutlineRenderer::GetExtents
 */
bool moldyn::SphereOutlineRenderer::GetExtents(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    MultiParticleDataCall *c2 = this->getDataSlot.CallAs<MultiParticleDataCall>();
    if ((c2 != NULL) && ((*c2)(1))) {
        cr->SetTimeFramesCount(c2->FrameCount());
        cr->AccessBoundingBoxes() = c2->AccessBoundingBoxes();

        float scaling = cr->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        if (scaling > 0.0000001) {
            scaling = 10.0f / scaling;
        } else {
            scaling = 1.0f;
        }
        cr->AccessBoundingBoxes().MakeScaledWorld(scaling);

    } else {
        cr->SetTimeFramesCount(1);
        cr->AccessBoundingBoxes().Clear();
    }

    return true;
}


/*
 * moldyn::SphereOutlineRenderer::release
 */
void moldyn::SphereOutlineRenderer::release(void) {

    // TODO: Implement

}


/*
 * moldyn::SphereOutlineRenderer::Render
 */
bool moldyn::SphereOutlineRenderer::Render(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    MultiParticleDataCall *c2 = this->getDataSlot.CallAs<MultiParticleDataCall>();
    float scaling = 1.0f;
    if (c2 != NULL) {
        c2->SetFrameID(static_cast<unsigned int>(cr->Time()));
        if (!(*c2)(1)) return false;

        // calculate scaling
        scaling = c2->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        if (scaling > 0.0000001) {
            scaling = 10.0f / scaling;
        } else {
            scaling = 1.0f;
        }

        c2->SetFrameID(static_cast<unsigned int>(cr->Time()));
        if (!(*c2)(0)) return false;
    } else {
        return false;
    }

    glScalef(scaling, scaling, scaling); // ... unklar ob problematisch, aber eigentlich nicht

    vislib::math::Vector<float, 3> &camDir = cr->GetCameraParameters()->EyeDirection();
    vislib::math::Vector<float, 3> &camX = cr->GetCameraParameters()->EyeRightVector();
    vislib::math::Vector<float, 3> &camY = cr->GetCameraParameters()->EyeUpVector();

    const unsigned int segCnt = 100;
    const int angleOffsetSteps = 3;
    const float angleOffsetStepSize = 0.125f;
    const float colR = 1.0f;
    const float colG = 1.0f;
    const float colB = 1.0f;

    vislib::math::Vector<float, 3> *vec = new vislib::math::Vector<float, 3>[segCnt];
    float *ang = new float[segCnt];
    for (unsigned int i = 0; i < segCnt; i++) {
        float a = static_cast<float>(M_PI) * static_cast<float>(2 * i) / static_cast<float>(segCnt);

        vec[i] = camX * cos(a) + camY * sin(a);
        vec[i].Normalise();
        ang[i] = static_cast<float>(M_PI) * 0.5f;
    }

    ::glDisable(GL_LIGHTING);
    ::glEnable(GL_BLEND);
    ::glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    ::glDisable(GL_LINE_SMOOTH);
    ::glLineWidth(1.0f);

    vislib::math::Vector<float, 3> v;

    if (c2 != NULL) {
        for (unsigned int i = 0; i < c2->GetParticleListCount(); i++) {
            MultiParticleDataCall::Particles &parts = c2->AccessParticles(i);
            float rad = parts.GetGlobalRadius();
            bool loadRad = parts.GetVertexDataType() == MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR;
            const float *posData = static_cast<const float*>(parts.GetVertexData());
            if ((parts.GetVertexDataType() != MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR)
                && (parts.GetVertexDataType() != MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ)) continue;
            unsigned int stride = vislib::math::Max<unsigned int>((loadRad ? 4 : 3) * sizeof(float), parts.GetVertexDataStride());
            // colour ignored (for now)
            for (UINT64 j = 0; j < parts.GetCount(); j++) {
                vislib::math::ShallowVector<float, 3> pos(const_cast<float*>(posData));
                if (loadRad) rad = posData[3];
                vislib::math::Point<float, 3> posP(const_cast<float*>(posData));
                posP.Set(posP.X() * scaling, posP.Y() * scaling, posP.Z() * scaling);

                // Calculate outline angles
                float d = cr->GetCameraParameters()->EyePosition().Distance(posP);
                float p = (rad * rad * scaling * scaling) / d;
                float q = d - p;
                float h = ::sqrt(p * q);
                float a = ::atan2(h, -p);
                for (unsigned int s = 0; s < segCnt; s++) {
                    ang[s] = a;
                }

                // Draw "sphere" outline
                for (int angOffStep = -angleOffsetSteps; angOffStep <= angleOffsetSteps; angOffStep++) {
                    float angOff = static_cast<float>(angOffStep) * angleOffsetStepSize;

                    float colA = 1.0f;
                    if (angleOffsetSteps > 0) {
                        colA -= static_cast<float>(vislib::math::Abs(angOffStep)) / static_cast<float>(angleOffsetSteps * 2);
                        if (angOffStep < 0) colA *= 0.5f;
                    }

                    ::glColor4f(colR, colG, colB, colA);
                    ::glBegin(GL_LINE_LOOP);
                    for (unsigned int s = 0; s < segCnt; s++) {
                        float sa = sin(ang[s] + angOff);
                        float ca = cos(ang[s] + angOff);
                        v = pos + (vec[s] * sa + camDir * ca) * rad;

                        ::glVertex3fv(v.PeekComponents());
                    }
                    ::glEnd();
                }


                posData = reinterpret_cast<const float*>(reinterpret_cast<const char*>(posData) + stride);
            }
        }
    }

    delete[] ang;
    delete[] vec;

    return true;
}
