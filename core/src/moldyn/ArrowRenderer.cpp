/*
 * ArrowRenderer.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "vislib/graphics/gl/IncludeAllGL.h"
#include "mmcore/moldyn/ArrowRenderer.h"
#include "mmcore/moldyn/DirectionalParticleDataCall.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/view/CallClipPlane.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "mmcore/view/CallRender3D.h"
#include "vislib/assert.h"

using namespace megamol::core;


/*
 * moldyn::ArrowRenderer::ArrowRenderer
 */
moldyn::ArrowRenderer::ArrowRenderer(void) : Renderer3DModule(),
        arrowShader(), getDataSlot("getdata", "Connects to the data source"),
        getTFSlot("gettransferfunction", "Connects to the transfer function module"),
        //getClipPlaneSlot("getclipplane", "Connects to a clipping plane module"),
        greyTF(0),
        lengthScaleSlot("lengthScale", ""), lengthFilterSlot("lengthFilter", "Filters the arrows by length") {

    this->getDataSlot.SetCompatibleCall<moldyn::DirectionalParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->getTFSlot.SetCompatibleCall<view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->getTFSlot);

    //this->getClipPlaneSlot.SetCompatibleCall<view::CallClipPlaneDescription>();
    //this->MakeSlotAvailable(&this->getClipPlaneSlot);
    
    this->lengthScaleSlot << new param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->lengthScaleSlot);
	
    this->lengthFilterSlot << new param::FloatParam( 0.0f, 0.0);
    this->MakeSlotAvailable(&this->lengthFilterSlot);
}


/*
 * moldyn::ArrowRenderer::~ArrowRenderer
 */
moldyn::ArrowRenderer::~ArrowRenderer(void) {
    this->Release();
}


/*
 * moldyn::ArrowRenderer::create
 */
bool moldyn::ArrowRenderer::create(void) {
    if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions()) {
        return false;
    }

    vislib::graphics::gl::ShaderSource vert, frag;

    if (!instance()->ShaderSourceFactory().MakeShaderSource("arrow::vertex", vert)) {
        return false;
    }
    if (!instance()->ShaderSourceFactory().MakeShaderSource("arrow::fragment", frag)) {
        return false;
    }

    try {
        if (!this->arrowShader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
                "Unable to compile arrow shader: Unknown error\n");
            return false;
        }

    } catch(vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile arrow shader (@%s): %s\n", 
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(
            ce.FailedAction()) ,ce.GetMsgA());
        return false;
    } catch(vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile arrow shader: %s\n", e.GetMsgA());
        return false;
    } catch(...) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
            "Unable to compile arrow shader: Unknown exception\n");
        return false;
    }

    glEnable(GL_TEXTURE_1D);
    glGenTextures(1, &this->greyTF);
    unsigned char tex[6] = {
        0, 0, 0,  255, 255, 255
    };
    glBindTexture(GL_TEXTURE_1D, this->greyTF);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, 2, 0, GL_RGB, GL_UNSIGNED_BYTE, tex);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glBindTexture(GL_TEXTURE_1D, 0);
    glDisable(GL_TEXTURE_1D);

    return true;
}


/*
 * moldyn::ArrowRenderer::GetExtents
 */
bool moldyn::ArrowRenderer::GetExtents(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    DirectionalParticleDataCall *c2 = this->getDataSlot.CallAs<DirectionalParticleDataCall>();
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
 * moldyn::ArrowRenderer::release
 */
void moldyn::ArrowRenderer::release(void) {
    this->arrowShader.Release();
    ::glDeleteTextures(1, &this->greyTF);
}


/*
 * moldyn::ArrowRenderer::Render
 */
bool moldyn::ArrowRenderer::Render(Call& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    DirectionalParticleDataCall *c2 = this->getDataSlot.CallAs<DirectionalParticleDataCall>();
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
  
	float lengthScale = this->lengthScaleSlot.Param<param::FloatParam>()->Value();
    float lengthFilter = this->lengthFilterSlot.Param<param::FloatParam>()->Value();

    //view::CallClipPlane *ccp = this->getClipPlaneSlot.CallAs<view::CallClipPlane>();
    //float clipDat[4];
    //float clipCol[3];
    //if ((ccp != NULL) && (*ccp)()) {
    //    clipDat[0] = ccp->GetPlane().Normal().X();
    //    clipDat[1] = ccp->GetPlane().Normal().Y();
    //    clipDat[2] = ccp->GetPlane().Normal().Z();
    //    vislib::math::Vector<float, 3> grr(ccp->GetPlane().Point().PeekCoordinates());
    //    clipDat[3] = grr.Dot(ccp->GetPlane().Normal());
    //    clipCol[0] = static_cast<float>(ccp->GetColour()[0]) / 255.0f;
    //    clipCol[1] = static_cast<float>(ccp->GetColour()[1]) / 255.0f;
    //    clipCol[2] = static_cast<float>(ccp->GetColour()[2]) / 255.0f;

    //} else {
    //    clipDat[0] = clipDat[1] = clipDat[2] = clipDat[3] = 0.0f;
    //    clipCol[0] = clipCol[1] = clipCol[2] = 0.75f;
    //}

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

    this->arrowShader.Enable();

    glUniform4fv(this->arrowShader.ParameterLocation("viewAttr"), 1, viewportStuff);
    glUniform3fv(this->arrowShader.ParameterLocation("camIn"), 1, cr->GetCameraParameters()->Front().PeekComponents());
    glUniform3fv(this->arrowShader.ParameterLocation("camRight"), 1, cr->GetCameraParameters()->Right().PeekComponents());
    glUniform3fv(this->arrowShader.ParameterLocation("camUp"), 1, cr->GetCameraParameters()->Up().PeekComponents());
    this->arrowShader.SetParameter("lengthScale", lengthScale);
    this->arrowShader.SetParameter("lengthFilter", lengthFilter);

    //glUniform4fvARB(this->arrowShader.ParameterLocation("clipDat"), 1, clipDat);
    //glUniform3fvARB(this->arrowShader.ParameterLocation("clipCol"), 1, clipCol);

	glScalef(scaling, scaling, scaling);

    if (c2 != NULL) {
        unsigned int cial = glGetAttribLocationARB(this->arrowShader, "colIdx");
        unsigned int tpal = glGetAttribLocationARB(this->arrowShader, "dir");

        for (unsigned int i = 0; i < c2->GetParticleListCount(); i++) {
            DirectionalParticleDataCall::Particles &parts = c2->AccessParticles(i);
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
                case MultiParticleDataCall::Particles::COLDATA_FLOAT_I: {
                    glEnableVertexAttribArrayARB(cial);
                    glVertexAttribPointerARB(cial, 1, GL_FLOAT, GL_FALSE, parts.GetColourDataStride(), parts.GetColourData());

                    glEnable(GL_TEXTURE_1D);
                    view::CallGetTransferFunction *cgtf = this->getTFSlot.CallAs<view::CallGetTransferFunction>();
                    if ((cgtf != NULL) && ((*cgtf)())) {
                        glBindTexture(GL_TEXTURE_1D, cgtf->OpenGLTexture());
                        colTabSize = cgtf->TextureSize();
                    } else {
                        glBindTexture(GL_TEXTURE_1D, this->greyTF);
                        colTabSize = 2;
                    }

                    glUniform1i(this->arrowShader.ParameterLocation("colTab"), 0);
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
                    glUniform4f(this->arrowShader.ParameterLocation("inConsts1"), parts.GetGlobalRadius(), minC, maxC, float(colTabSize));
                    glVertexPointer(3, GL_FLOAT, parts.GetVertexDataStride(), parts.GetVertexData());
                    break;
                case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                    glEnableClientState(GL_VERTEX_ARRAY);
                    glUniform4f(this->arrowShader.ParameterLocation("inConsts1"), -1.0f, minC, maxC, float(colTabSize));
                    glVertexPointer(4, GL_FLOAT, parts.GetVertexDataStride(), parts.GetVertexData());
                    break;
                default:
                    continue;
            }

            // direction
            switch (parts.GetDirDataType()) {
                case DirectionalParticleDataCall::Particles::DIRDATA_FLOAT_XYZ:
                    ::glEnableVertexAttribArrayARB(tpal);
                    ::glVertexAttribPointerARB(tpal, 3, GL_FLOAT, GL_FALSE, parts.GetDirDataStride(), parts.GetDirData());
                    break;
                default:
                    continue;
            }

            glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));

            glDisableClientState(GL_COLOR_ARRAY);
            glDisableClientState(GL_VERTEX_ARRAY);
            glDisableVertexAttribArrayARB(cial);
            glDisableVertexAttribArrayARB(tpal);
            glDisable(GL_TEXTURE_1D);
        }

        c2->Unlock();

    }

    this->arrowShader.Disable();

    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);

    return true;
}
