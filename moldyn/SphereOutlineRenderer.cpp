/*
 * SphereOutlineRenderer.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "SphereOutlineRenderer.h"
#include "MultiParticleDataCall.h"
#include "CoreInstance.h"
#include "view/CallRender3D.h"
#include <GL/gl.h>
#include "vislib/assert.h"

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

    glScalef(scaling, scaling, scaling);

    // TODO: Implement

/*
    if (c2 != NULL) {
        unsigned int cial = glGetAttribLocationARB(this->sphereShader, "colIdx");

        for (unsigned int i = 0; i < c2->GetParticleListCount(); i++) {
            MultiParticleDataCall::Particles &parts = c2->AccessParticles(i);
            float minC = 0.0f, maxC = 0.0f;
            unsigned int colTabSize = 0;

            // colour
            switch (parts.GetColourDataType()) {
                case MultiParticleDataCall::Particles::COLDATA_NONE:
                    glColor3ubv(parts.GetGlobalColour());
                    break;
                case MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
                    glEnableClientState(GL_COLOR_ARRAY);
                    glColorPointer(3, GL_UNSIGNED_BYTE,
                        parts.GetColourDataStride(), parts.GetColourData());
                    break;
                case MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
                    glEnableClientState(GL_COLOR_ARRAY);
                    glColorPointer(4, GL_UNSIGNED_BYTE,
                        parts.GetColourDataStride(), parts.GetColourData());
                    break;
                case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
                    glEnableClientState(GL_COLOR_ARRAY);
                    glColorPointer(3, GL_FLOAT,
                        parts.GetColourDataStride(), parts.GetColourData());
                    break;
                case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
                    glEnableClientState(GL_COLOR_ARRAY);
                    glColorPointer(4, GL_FLOAT,
                        parts.GetColourDataStride(), parts.GetColourData());
                    break;
                case MultiParticleDataCall::Particles::COLDATA_FLOAT_I: {
                    glEnableVertexAttribArrayARB(cial);
                    glVertexAttribPointerARB(cial, 1, GL_FLOAT, GL_FALSE,
                        parts.GetColourDataStride(), parts.GetColourData());

                    glEnable(GL_TEXTURE_1D);

                    view::CallGetTransferFunction *cgtf = this->getTFSlot.CallAs<view::CallGetTransferFunction>();
                    if ((cgtf != NULL) && ((*cgtf)())) {
                        glBindTexture(GL_TEXTURE_1D, cgtf->OpenGLTexture());
                        colTabSize = cgtf->TextureSize();
                    } else {
                        glBindTexture(GL_TEXTURE_1D, this->greyTF);
                        colTabSize = 2;
                    }

                    glUniform1iARB(this->sphereShader.ParameterLocation("colTab"), 0);
                    minC = parts.GetMinColourIndexValue();
                    maxC = parts.GetMaxColourIndexValue();
                    glColor3ub(127, 127, 127);
                } break;
                default:
                    glColor3ub(127, 127, 127);
                    break;
            }

            // radius and position
            switch (parts.GetVertexDataType()) {
                case MultiParticleDataCall::Particles::VERTDATA_NONE:
                    continue;
                case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
                    glEnableClientState(GL_VERTEX_ARRAY);
                    glUniform4fARB(this->sphereShader.ParameterLocation("inConsts1"),
                        parts.GetGlobalRadius(), minC, maxC, float(colTabSize));
                    glVertexPointer(3, GL_FLOAT,
                        parts.GetVertexDataStride(), parts.GetVertexData());
                    break;
                case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                    glEnableClientState(GL_VERTEX_ARRAY);
                    glUniform4fARB(this->sphereShader.ParameterLocation("inConsts1"),
                        -1.0f, minC, maxC, float(colTabSize));
                    glVertexPointer(4, GL_FLOAT,
                        parts.GetVertexDataStride(), parts.GetVertexData());
                    break;
                default:
                    continue;
            }

            glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));

            glDisableClientState(GL_COLOR_ARRAY);
            glDisableClientState(GL_VERTEX_ARRAY);
            glDisableVertexAttribArrayARB(cial);
            glDisable(GL_TEXTURE_1D);
        }

        c2->Unlock();

    }
*/

    return true;
}
