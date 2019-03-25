/*
 * AbstractSimpleSphereRenderer.cpp
 *
 * Copyright (C) 2012 by CGV (TU Dresden)
 * Alle Rechte vorbehalten.
 */


#include "stdafx.h"
#include "mmcore/moldyn/AbstractSimpleSphereRenderer.h"


using namespace megamol::core;


/*
 * moldyn::AbstractSimpleSphereRenderer::AbstractSimpleSphereRenderer
 */
moldyn::AbstractSimpleSphereRenderer::AbstractSimpleSphereRenderer(void) : Renderer3DModule(),
        getDataSlot("getdata", "Connects to the data source"),
        getTFSlot("gettransferfunction", "Connects to the transfer function module"),
        getClipPlaneSlot("getclipplane", "Connects to a clipping plane module"),
        greyTF(0),
        forceTimeSlot("forceTime", "Flag to force the time code to the specified value. Set to true when rendering a video."),
        useLocalBBoxParam("useLocalBBox", "Enforce usage of local bbox for camera setup") {

    this->getDataSlot.SetCompatibleCall<moldyn::MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->getTFSlot.SetCompatibleCall<view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->getTFSlot);

    this->getClipPlaneSlot.SetCompatibleCall<view::CallClipPlaneDescription>();
    this->MakeSlotAvailable(&this->getClipPlaneSlot);

    this->forceTimeSlot.SetParameter(new param::BoolParam(false));
    this->MakeSlotAvailable(&this->forceTimeSlot);

    this->useLocalBBoxParam << new param::BoolParam(false);
    this->MakeSlotAvailable(&this->useLocalBBoxParam);
}


/*
 * moldyn::AbstractSimpleSphereRenderer::~AbstractSimpleSphereRenderer
 */
moldyn::AbstractSimpleSphereRenderer::~AbstractSimpleSphereRenderer(void) {
    this->Release();
}


/*
 * moldyn::AbstractSimpleSphereRenderer::create
 */
bool moldyn::AbstractSimpleSphereRenderer::create(void) {
    //if (!vislib::graphics::gl::GLSLShader::InitialiseExtensions()) {
    //    return false;
    //}

    //vislib::graphics::gl::ShaderSource vert, frag;

    //if (!instance()->ShaderSourceFactory().MakeShaderSource("simplesphere::vertex", vert)) {
    //    return false;
    //}
    //if (!instance()->ShaderSourceFactory().MakeShaderSource("simplesphere::fragment", frag)) {
    //    return false;
    //}

    ////printf("\nVertex Shader:\n%s\n\nFragment Shader:\n%s\n",
    ////    vert.WholeCode().PeekBuffer(),
    ////    frag.WholeCode().PeekBuffer());

    //try {
    //    if (!this->sphereShader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
    //        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
    //            "Unable to compile sphere shader: Unknown error\n");
    //        return false;
    //    }

    //} catch(vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
    //    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
    //        "Unable to compile sphere shader (@%s): %s\n", 
    //        vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(
    //        ce.FailedAction()) ,ce.GetMsgA());
    //    return false;
    //} catch(vislib::Exception e) {
    //    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
    //        "Unable to compile sphere shader: %s\n", e.GetMsgA());
    //    return false;
    //} catch(...) {
    //    vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR,
    //        "Unable to compile sphere shader: Unknown exception\n");
    //    return false;
    //}

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
 * moldyn::AbstractSimpleSphereRenderer::GetExtents
 */
bool moldyn::AbstractSimpleSphereRenderer::GetExtents(view::CallRender3D& call) {
    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    MultiParticleDataCall *c2 = this->getDataSlot.CallAs<MultiParticleDataCall>();
    if ((c2 != NULL)) {
        c2->SetFrameID(static_cast<unsigned int>(cr->Time()), this->isTimeForced());
        if (!(*c2)(1)) return false;
        cr->SetTimeFramesCount(c2->FrameCount());
        auto const plcount = c2->GetParticleListCount();
        if (this->useLocalBBoxParam.Param<param::BoolParam>()->Value() && plcount > 0) {
            auto bbox = c2->AccessParticles(0).GetBBox();
            auto cbbox = bbox;
            cbbox.Grow(c2->AccessParticles(0).GetGlobalRadius());
            for (unsigned pidx = 1; pidx < plcount; ++pidx) {
                auto temp = c2->AccessParticles(pidx).GetBBox();
                bbox.Union(temp);
                temp.Grow(c2->AccessParticles(pidx).GetGlobalRadius());
                cbbox.Union(temp);
            }
            cr->AccessBoundingBoxes().SetObjectSpaceBBox(bbox);
            cr->AccessBoundingBoxes().SetObjectSpaceClipBox(cbbox);
        } else {
            cr->AccessBoundingBoxes() = c2->AccessBoundingBoxes();
        }

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
 * moldyn::AbstractSimpleSphereRenderer::release
 */
void moldyn::AbstractSimpleSphereRenderer::release(void) {
    //this->sphereShader.Release();
    ::glDeleteTextures(1, &this->greyTF);
}


/*
 * moldyn::AbstractSimpleSphereRenderer::getData
 */
moldyn::MultiParticleDataCall *moldyn::AbstractSimpleSphereRenderer::getData(unsigned int t, float& outScaling) {
    MultiParticleDataCall *c2 = this->getDataSlot.CallAs<MultiParticleDataCall>();
    outScaling = 1.0f;
    if (c2 != NULL) {
        c2->SetFrameID(t, this->isTimeForced());
        if (!(*c2)(1)) return NULL;

        // calculate scaling
        auto const plcount = c2->GetParticleListCount();
        if (this->useLocalBBoxParam.Param<param::BoolParam>()->Value() && plcount > 0) {
            outScaling = c2->AccessParticles(0).GetBBox().LongestEdge();
            for (unsigned pidx = 0; pidx < plcount; ++pidx) {
                auto const temp = c2->AccessParticles(pidx).GetBBox().LongestEdge();
                if (outScaling < temp) {
                    outScaling = temp;
                }
            }
        } else {
            outScaling = c2->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
        }
        if (outScaling > 0.0000001) {
            outScaling = 10.0f / outScaling;
        } else {
            outScaling = 1.0f;
        }

        c2->SetFrameID(t, this->isTimeForced());
        if (!(*c2)(0)) return NULL;

        return c2;
    } else {
        return NULL;
    }
}


/*
 * moldyn::AbstractSimpleSphereRenderer::getClipData
 */
void moldyn::AbstractSimpleSphereRenderer::getClipData(float *clipDat, float *clipCol) {
    view::CallClipPlane *ccp = this->getClipPlaneSlot.CallAs<view::CallClipPlane>();
    if ((ccp != NULL) && (*ccp)()) {
        clipDat[0] = ccp->GetPlane().Normal().X();
        clipDat[1] = ccp->GetPlane().Normal().Y();
        clipDat[2] = ccp->GetPlane().Normal().Z();
        vislib::math::Vector<float, 3> grr(ccp->GetPlane().Point().PeekCoordinates());
        clipDat[3] = grr.Dot(ccp->GetPlane().Normal());
        clipCol[0] = static_cast<float>(ccp->GetColour()[0]) / 255.0f;
        clipCol[1] = static_cast<float>(ccp->GetColour()[1]) / 255.0f;
        clipCol[2] = static_cast<float>(ccp->GetColour()[2]) / 255.0f;
        clipCol[3] = static_cast<float>(ccp->GetColour()[3]) / 255.0f;

    } else {
        clipDat[0] = clipDat[1] = clipDat[2] = clipDat[3] = 0.0f;
        clipCol[0] = clipCol[1] = clipCol[2] = 0.75f;
        clipCol[3] = 1.0f;
    }
}



///*
// * moldyn::AbstractSimpleSphereRenderer::Render
// */
//bool moldyn::AbstractSimpleSphereRenderer::Render(view::CallRender3D& call) {
//    view::CallRender3D *cr = dynamic_cast<view::CallRender3D*>(&call);
//    if (cr == NULL) return false;
//
//    MultiParticleDataCall *c2 = this->getDataSlot.CallAs<MultiParticleDataCall>();
//    float scaling = 1.0f;
//    if (c2 != NULL) {
//        c2->SetFrameID(static_cast<unsigned int>(cr->Time()));
//        if (!(*c2)(1)) return false;
//
//        // calculate scaling
//        scaling = c2->AccessBoundingBoxes().ObjectSpaceBBox().LongestEdge();
//        if (scaling > 0.0000001) {
//            scaling = 10.0f / scaling;
//        } else {
//            scaling = 1.0f;
//        }
//
//        c2->SetFrameID(static_cast<unsigned int>(cr->Time()));
//        if (!(*c2)(0)) return false;
//    } else {
//        return false;
//    }
//
//    view::CallClipPlane *ccp = this->getClipPlaneSlot.CallAs<view::CallClipPlane>();
//    float clipDat[4];
//    float clipCol[4];
//    if ((ccp != NULL) && (*ccp)()) {
//        clipDat[0] = ccp->GetPlane().Normal().X();
//        clipDat[1] = ccp->GetPlane().Normal().Y();
//        clipDat[2] = ccp->GetPlane().Normal().Z();
//        vislib::math::Vector<float, 3> grr(ccp->GetPlane().Point().PeekCoordinates());
//        clipDat[3] = grr.Dot(ccp->GetPlane().Normal());
//        clipCol[0] = static_cast<float>(ccp->GetColour()[0]) / 255.0f;
//        clipCol[1] = static_cast<float>(ccp->GetColour()[1]) / 255.0f;
//        clipCol[2] = static_cast<float>(ccp->GetColour()[2]) / 255.0f;
//        clipCol[3] = static_cast<float>(ccp->GetColour()[3]) / 255.0f;
//
//    } else {
//        clipDat[0] = clipDat[1] = clipDat[2] = clipDat[3] = 0.0f;
//        clipCol[0] = clipCol[1] = clipCol[2] = 0.75f;
//        clipCol[3] = 1.0f;
//    }
//
//    glDisable(GL_BLEND);
//    glEnable(GL_DEPTH_TEST);
//    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
//
//    float viewportStuff[4];
//    ::glGetFloatv(GL_VIEWPORT, viewportStuff);
//    glPointSize(vislib::math::Max(viewportStuff[2], viewportStuff[3]));
//    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
//    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
//    viewportStuff[2] = 2.0f / viewportStuff[2];
//    viewportStuff[3] = 2.0f / viewportStuff[3];
//
//    this->sphereShader.Enable();
//
//    glUniform4fvARB(this->sphereShader.ParameterLocation("viewAttr"),
//        1, viewportStuff);
//    glUniform3fvARB(this->sphereShader.ParameterLocation("camIn"),
//        1, cr->GetCameraParameters()->Front().PeekComponents());
//    glUniform3fvARB(this->sphereShader.ParameterLocation("camRight"),
//        1, cr->GetCameraParameters()->Right().PeekComponents());
//    glUniform3fvARB(this->sphereShader.ParameterLocation("camUp"),
//        1, cr->GetCameraParameters()->Up().PeekComponents());
//
//    glUniform4fvARB(this->sphereShader.ParameterLocation("clipDat"), 1, clipDat);
//    glUniform4fvARB(this->sphereShader.ParameterLocation("clipCol"), 1, clipCol);
//
//    glScalef(scaling, scaling, scaling);
//
//    if (c2 != NULL) {
//        unsigned int cial = glGetAttribLocationARB(this->sphereShader, "colIdx");
//
//        for (unsigned int i = 0; i < c2->GetParticleListCount(); i++) {
//            MultiParticleDataCall::Particles &parts = c2->AccessParticles(i);
//            float minC = 0.0f, maxC = 0.0f;
//            unsigned int colTabSize = 0;
//
//            // colour
//            switch (parts.GetColourDataType()) {
//                case MultiParticleDataCall::Particles::COLDATA_NONE:
//                    glColor3ubv(parts.GetGlobalColour());
//                    break;
//                case MultiParticleDataCall::Particles::COLDATA_UINT8_RGB:
//                    glEnableClientState(GL_COLOR_ARRAY);
//                    glColorPointer(3, GL_UNSIGNED_BYTE,
//                        parts.GetColourDataStride(), parts.GetColourData());
//                    break;
//                case MultiParticleDataCall::Particles::COLDATA_UINT8_RGBA:
//                    glEnableClientState(GL_COLOR_ARRAY);
//                    glColorPointer(4, GL_UNSIGNED_BYTE,
//                        parts.GetColourDataStride(), parts.GetColourData());
//                    break;
//                case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGB:
//                    glEnableClientState(GL_COLOR_ARRAY);
//                    glColorPointer(3, GL_FLOAT,
//                        parts.GetColourDataStride(), parts.GetColourData());
//                    break;
//                case MultiParticleDataCall::Particles::COLDATA_FLOAT_RGBA:
//                    glEnableClientState(GL_COLOR_ARRAY);
//                    glColorPointer(4, GL_FLOAT,
//                        parts.GetColourDataStride(), parts.GetColourData());
//                    break;
//                case MultiParticleDataCall::Particles::COLDATA_FLOAT_I: {
//                    glEnableVertexAttribArrayARB(cial);
//                    glVertexAttribPointerARB(cial, 1, GL_FLOAT, GL_FALSE,
//                        parts.GetColourDataStride(), parts.GetColourData());
//
//                    glEnable(GL_TEXTURE_1D);
//
//                    view::CallGetTransferFunction *cgtf = this->getTFSlot.CallAs<view::CallGetTransferFunction>();
//                    if ((cgtf != NULL) && ((*cgtf)())) {
//                        glBindTexture(GL_TEXTURE_1D, cgtf->OpenGLTexture());
//                        colTabSize = cgtf->TextureSize();
//                    } else {
//                        glBindTexture(GL_TEXTURE_1D, this->greyTF);
//                        colTabSize = 2;
//                    }
//
//                    glUniform1iARB(this->sphereShader.ParameterLocation("colTab"), 0);
//                    minC = parts.GetMinColourIndexValue();
//                    maxC = parts.GetMaxColourIndexValue();
//                    glColor3ub(127, 127, 127);
//                } break;
//                default:
//                    glColor3ub(127, 127, 127);
//                    break;
//            }
//
//            // radius and position
//            switch (parts.GetVertexDataType()) {
//                case MultiParticleDataCall::Particles::VERTDATA_NONE:
//                    continue;
//                case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
//                    glEnableClientState(GL_VERTEX_ARRAY);
//                    glUniform4fARB(this->sphereShader.ParameterLocation("inConsts1"),
//                        parts.GetGlobalRadius(), minC, maxC, float(colTabSize));
//                    glVertexPointer(3, GL_FLOAT,
//                        parts.GetVertexDataStride(), parts.GetVertexData());
//                    break;
//                case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
//                    glEnableClientState(GL_VERTEX_ARRAY);
//                    glUniform4fARB(this->sphereShader.ParameterLocation("inConsts1"),
//                        -1.0f, minC, maxC, float(colTabSize));
//                    glVertexPointer(4, GL_FLOAT,
//                        parts.GetVertexDataStride(), parts.GetVertexData());
//                    break;
//                default:
//                    continue;
//            }
//
//            glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));
//
//            glDisableClientState(GL_COLOR_ARRAY);
//            glDisableClientState(GL_VERTEX_ARRAY);
//            glDisableVertexAttribArrayARB(cial);
//            glDisable(GL_TEXTURE_1D);
//        }
//
//        c2->Unlock();
//
//    }
//
//    this->sphereShader.Disable();
//
//    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);
//
//    return true;
//}

bool moldyn::AbstractSimpleSphereRenderer::isTimeForced(void) const {
    return this->forceTimeSlot.Param<param::BoolParam>()->Value();
}
