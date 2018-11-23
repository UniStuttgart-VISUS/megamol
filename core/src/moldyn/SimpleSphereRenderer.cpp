/*
 * SimpleSphereRenderer.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "mmcore/moldyn/SimpleSphereRenderer.h"
#include "mmcore/moldyn/MultiParticleDataCall.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/view/CallClipPlane.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "mmcore/view/CallRender3D.h"
#include "vislib/assert.h"
#include "mmcore/param/FloatParam.h"

using namespace megamol::core;


/*
 * moldyn::SimpleSphereRenderer::SimpleSphereRenderer
 */
moldyn::SimpleSphereRenderer::SimpleSphereRenderer(void) : AbstractSimpleSphereRenderer(),
        sphereShader(),
        scalingParam("scaling", "scaling factor for particle radii") {
    // intentionally empty

    this->scalingParam << new core::param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->scalingParam);
}


/*
 * moldyn::SimpleSphereRenderer::~SimpleSphereRenderer
 */
moldyn::SimpleSphereRenderer::~SimpleSphereRenderer(void) {
    this->Release();
}


/*
 * moldyn::SimpleSphereRenderer::create
 */
bool moldyn::SimpleSphereRenderer::create(void) {
    ASSERT(IsAvailable());

    vislib::graphics::gl::ShaderSource vert, frag;

    if (!instance()->ShaderSourceFactory().MakeShaderSource("simplesphere::vertex", vert)) {
        return false;
    }
    if (!instance()->ShaderSourceFactory().MakeShaderSource("simplesphere::fragment", frag)) {
        return false;
    }

    //printf("\nVertex Shader:\n%s\n\nFragment Shader:\n%s\n",
    //    vert.WholeCode().PeekBuffer(),
    //    frag.WholeCode().PeekBuffer());

    try {
        if (!this->sphereShader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to compile sphere shader: Unknown error\n");
            return false;
        }

    } catch(vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile sphere shader (@%s): %s\n", 
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(
            ce.FailedAction()) ,ce.GetMsgA());
        return false;
    } catch(vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile sphere shader: %s\n", e.GetMsgA());
        return false;
    } catch(...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile sphere shader: Unknown exception\n");
        return false;
    }

    return AbstractSimpleSphereRenderer::create();
}


/*
 * moldyn::SimpleSphereRenderer::release
 */
void moldyn::SimpleSphereRenderer::release(void) {
    this->sphereShader.Release();
    AbstractSimpleSphereRenderer::release();
}


/*
 * moldyn::SimpleSphereRenderer::Render
 */
bool moldyn::SimpleSphereRenderer::Render(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    float scaling = 1.0f;
    MultiParticleDataCall *c2 = this->getData(static_cast<unsigned int>(cr->Time()), scaling);
    if (c2 == NULL) return false;

    float clipDat[4];
    float clipCol[4];
    this->getClipData(clipDat, clipCol);
    
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    float viewportStuff[4];
    ::glGetFloatv(GL_VIEWPORT, viewportStuff);
    glPointSize(vislib::math::Max(viewportStuff[2], viewportStuff[3]));
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    this->sphereShader.Enable();
    glUniform4fv(this->sphereShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fv(this->sphereShader.ParameterLocation("camIn"), 1, cr->GetCameraParameters()->Front().PeekComponents());
    glUniform3fv(this->sphereShader.ParameterLocation("camRight"), 1, cr->GetCameraParameters()->Right().PeekComponents());
    glUniform3fv(this->sphereShader.ParameterLocation("camUp"), 1, cr->GetCameraParameters()->Up().PeekComponents());
    glUniform1f(this->sphereShader.ParameterLocation("scaling"), this->scalingParam.Param<param::FloatParam>()->Value());

    glUniform4fv(this->sphereShader.ParameterLocation("clipDat"), 1, clipDat);
    glUniform4fv(this->sphereShader.ParameterLocation("clipCol"), 1, clipCol);

    glScalef(scaling, scaling, scaling);

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
                glColorPointer(3, GL_UNSIGNED_BYTE, parts.GetColourDataStride(), parts.GetColourData());
                break;
            case MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
                glEnableClientState(GL_COLOR_ARRAY);
                glColorPointer(4, GL_UNSIGNED_BYTE, parts.GetColourDataStride(), parts.GetColourData());
                break;
            case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
                glEnableClientState(GL_COLOR_ARRAY);
                glColorPointer(3, GL_FLOAT, parts.GetColourDataStride(), parts.GetColourData());
                break;
            case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
                glEnableClientState(GL_COLOR_ARRAY);
                glColorPointer(4, GL_FLOAT, parts.GetColourDataStride(), parts.GetColourData());
                break;
            case MultiParticleDataCall::Particles::COLDATA_FLOAT_I:
            case MultiParticleDataCall::Particles::COLDATA_DOUBLE_I: {
                glEnableVertexAttribArrayARB(cial);
                if (parts.GetColourDataType() == SimpleSphericalParticles::COLDATA_FLOAT_I) {
                    glVertexAttribPointerARB(
                        cial, 1, GL_FLOAT, GL_FALSE, parts.GetColourDataStride(), parts.GetColourData());
                } else {
                    glVertexAttribLPointer(cial, 1, GL_DOUBLE, parts.GetColourDataStride(), parts.GetColourData());
                }
                glEnable(GL_TEXTURE_1D);

                view::CallGetTransferFunction *cgtf = this->getTFSlot.CallAs<view::CallGetTransferFunction>();
                if ((cgtf != NULL) && ((*cgtf)())) {
                    glBindTexture(GL_TEXTURE_1D, cgtf->OpenGLTexture());
                    colTabSize = cgtf->TextureSize();
                } else {
                    glBindTexture(GL_TEXTURE_1D, this->greyTF);
                    colTabSize = 2;
                }

                glUniform1i(this->sphereShader.ParameterLocation("colTab"), 0);
                minC = parts.GetMinColourIndexValue();
                maxC = parts.GetMaxColourIndexValue();
                glColor3ub(127, 127, 127);
            } break;
            case MultiParticleDataCall::Particles::COLDATA_USHORT_RGBA:
                glEnableClientState(GL_COLOR_ARRAY);
                glColorPointer(4, GL_UNSIGNED_SHORT, parts.GetColourDataStride(), parts.GetColourData());
                break;
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
                glUniform4f(this->sphereShader.ParameterLocation("inConsts1"), parts.GetGlobalRadius(), minC, maxC, float(colTabSize));
                glVertexPointer(3, GL_FLOAT, parts.GetVertexDataStride(), parts.GetVertexData());
                break;
            case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                glEnableClientState(GL_VERTEX_ARRAY);
                glUniform4f(this->sphereShader.ParameterLocation("inConsts1"), -1.0f, minC, maxC, float(colTabSize));
                glVertexPointer(4, GL_FLOAT, parts.GetVertexDataStride(), parts.GetVertexData());
                break;
            case MultiParticleDataCall::Particles::VERTDATA_DOUBLE_XYZ:
                glEnableClientState(GL_VERTEX_ARRAY);
                glUniform4f(this->sphereShader.ParameterLocation("inConsts1"), parts.GetGlobalRadius(), minC, maxC, float(colTabSize));
                glVertexPointer(3, GL_DOUBLE, parts.GetVertexDataStride(), parts.GetVertexData());
                break;
            case MultiParticleDataCall::Particles::VERTDATA_SHORT_XYZ:
                glEnableClientState(GL_VERTEX_ARRAY);
                glUniform4f(this->sphereShader.ParameterLocation("inConsts1"), parts.GetGlobalRadius(), minC, maxC, float(colTabSize));
                glVertexPointer(3, GL_SHORT, parts.GetVertexDataStride(), parts.GetVertexData());
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

    this->sphereShader.Disable();

    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);

    return true;
}
