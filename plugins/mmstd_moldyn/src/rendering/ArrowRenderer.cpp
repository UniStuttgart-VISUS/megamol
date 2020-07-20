/*
 * ArrowRenderer.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ArrowRenderer.h"


using namespace megamol::core;
using namespace megamol::core::moldyn;
using namespace megamol::stdplugin::moldyn::rendering;


ArrowRenderer::ArrowRenderer(void) : view::Renderer3DModule_2()
    , getDataSlot("getdata", "Connects to the data source")
    , getTFSlot("gettransferfunction", "Connects to the transfer function module")
    , getFlagsSlot("getflags", "connects to a FlagStorage")
    , getClipPlaneSlot("getclipplane", "Connects to a clipping plane module")
    , lengthScaleSlot("lengthScale", ""), lengthFilterSlot("lengthFilter", "Filters the arrows by length")
    , arrowShader()
    , greyTF(0) {

    this->getDataSlot.SetCompatibleCall<MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->getTFSlot.SetCompatibleCall<view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->getTFSlot);

    this->getFlagsSlot.SetCompatibleCall<FlagCallDescription>();
    this->MakeSlotAvailable(&this->getFlagsSlot);

    this->getClipPlaneSlot.SetCompatibleCall<view::CallClipPlaneDescription>();
    this->MakeSlotAvailable(&this->getClipPlaneSlot);
    
    this->lengthScaleSlot << new param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->lengthScaleSlot);
    
    this->lengthFilterSlot << new param::FloatParam( 0.0f, 0.0);
    this->MakeSlotAvailable(&this->lengthFilterSlot);
}


ArrowRenderer::~ArrowRenderer(void) {

    this->Release();
}


bool ArrowRenderer::create(void) {

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


bool ArrowRenderer::GetExtents(view::CallRender3D_2& call) {

    MultiParticleDataCall* c2 = this->getDataSlot.CallAs<MultiParticleDataCall>();
    if ((c2 != nullptr) && ((*c2)(1))) {
        call.SetTimeFramesCount(c2->FrameCount());
        call.AccessBoundingBoxes() = c2->AccessBoundingBoxes();

    } else {
        call.SetTimeFramesCount(1);
        call.AccessBoundingBoxes().Clear();
    }

    return true;
}


void ArrowRenderer::release(void) {

    this->arrowShader.Release();
    glDeleteTextures(1, &this->greyTF);
}


bool ArrowRenderer::Render(view::CallRender3D_2& call) {

    MultiParticleDataCall *c2 = this->getDataSlot.CallAs<MultiParticleDataCall>();
    if (c2 != nullptr) {
        c2->SetFrameID(static_cast<unsigned int>(call.Time()));
        if (!(*c2)(1)) return false;
        c2->SetFrameID(static_cast<unsigned int>(call.Time()));
        if (!(*c2)(0)) return false;
    } else {
        return false;
    }

    auto* cflags = this->getFlagsSlot.CallAs<FlagCall>();
  
    float lengthScale = this->lengthScaleSlot.Param<param::FloatParam>()->Value();
    float lengthFilter = this->lengthFilterSlot.Param<param::FloatParam>()->Value();

    // Clipping
    auto ccp = this->getClipPlaneSlot.CallAs<view::CallClipPlane>();
    float clipDat[4];
    float clipCol[4];
    if ((ccp != nullptr) && (*ccp)()) {
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

    // Camera
    view::Camera_2 cam;
    call.GetCamera(cam);
    cam_type::snapshot_type snapshot;
    cam_type::matrix_type viewTemp, projTemp;
    cam.calc_matrices(snapshot, viewTemp, projTemp, thecam::snapshot_content::all);
    glm::vec4 cam_view = snapshot.view_vector;
    glm::vec4 cam_right = snapshot.right_vector;
    glm::vec4 cam_up = snapshot.up_vector;

    // Matrices
    glm::mat4 view = viewTemp;
    glm::mat4 proj = projTemp;
    glm::mat4 MVinv = glm::inverse(view);
    glm::mat4 MVtransp = glm::transpose(view);
    glm::mat4 MVP = proj * view;
    glm::mat4 MVPinv = glm::inverse(MVP);
    glm::mat4 MVPtransp = glm::transpose(MVP);

    // Viewport
    auto viewport = call.GetViewport();
    glm::vec4 viewportStuff;
    viewportStuff[0] = 0.0f;
    viewportStuff[1] = 0.0f;
    viewportStuff[2] = static_cast<float>(viewport.Width());
    viewportStuff[3] = static_cast<float>(viewport.Height());
    if (viewportStuff[2] < 1.0f) viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f) viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    // Lights
    this->GetLights();
    glm::vec4 curlightDir = { 0.0f, 0.0f, 0.0f, 1.0f };
    if (this->lightMap.size() > 1) {
        vislib::sys::Log::DefaultLog.WriteWarn("ArrowRenderer: Only one single 'Distant Light' source is supported by this renderer");
    }
    for (auto light : this->lightMap) {
        if (light.second.lightType != core::view::light::DISTANTLIGHT) {
            vislib::sys::Log::DefaultLog.WriteWarn("ArrowRenderer: Only single 'Distant Light' source is supported by this renderer");
        }
        else {
            auto use_eyedir = light.second.dl_eye_direction;
            if (use_eyedir) {
                curlightDir = -cam_view;
            }
            else {
                auto lightDir = light.second.dl_direction;
                if (lightDir.size() == 3) {
                    curlightDir[0] = lightDir[0];
                    curlightDir[1] = lightDir[1];
                    curlightDir[2] = lightDir[2];
                }
                if (lightDir.size() == 4) {
                    curlightDir[3] = lightDir[3];
                }
                /// View Space Lighting. Comment line to change to Object Space Lighting.
                //curlightDir = MVtransp * curlightDir;
            }
/// TODO Implement missing distant light parameters:
            //light.second.dl_angularDiameter;
            //light.second.lightColor;
            //light.second.lightIntensity;
        }
    }

    glDisable(GL_BLEND);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glPointSize(vislib::math::Max(viewportStuff[2], viewportStuff[3]));

    this->arrowShader.Enable();

    glUniformMatrix4fv(this->arrowShader.ParameterLocation("MVinv"), 1, GL_FALSE, glm::value_ptr(MVinv));
    glUniformMatrix4fv(this->arrowShader.ParameterLocation("MVtransp"), 1, GL_FALSE, glm::value_ptr(MVtransp));
    glUniformMatrix4fv(this->arrowShader.ParameterLocation("MVP"), 1, GL_FALSE, glm::value_ptr(MVP));
    glUniformMatrix4fv(this->arrowShader.ParameterLocation("MVPinv"), 1, GL_FALSE, glm::value_ptr(MVPinv));
    glUniformMatrix4fv(this->arrowShader.ParameterLocation("MVPtransp"), 1, GL_FALSE, glm::value_ptr(MVPtransp));
    glUniform4fv(this->arrowShader.ParameterLocation("viewAttr"), 1, glm::value_ptr(viewportStuff));
    glUniform3fv(this->arrowShader.ParameterLocation("camIn"), 1, glm::value_ptr(cam_view));
    glUniform3fv(this->arrowShader.ParameterLocation("camRight"), 1, glm::value_ptr(cam_right));
    glUniform3fv(this->arrowShader.ParameterLocation("camUp"), 1, glm::value_ptr(cam_up));
    glUniform4fv(this->arrowShader.ParameterLocation("lightDir"), 1, glm::value_ptr(curlightDir));
    this->arrowShader.SetParameter("lengthScale", lengthScale);
    this->arrowShader.SetParameter("lengthFilter", lengthFilter);
    glUniform4fv(this->arrowShader.ParameterLocation("clipDat"), 1, clipDat);
    glUniform3fv(this->arrowShader.ParameterLocation("clipCol"), 1, clipCol);

    if (c2 != nullptr) {
        unsigned int cial = glGetAttribLocationARB(this->arrowShader, "colIdx");
        unsigned int tpal = glGetAttribLocationARB(this->arrowShader, "dir");
        bool useFlags = false;

        if (cflags != nullptr) {
            if (c2->GetParticleListCount() > 1) {
                vislib::sys::Log::DefaultLog.WriteWarn(
                    "ArrowRenderer: Cannot use FlagStorage together with multiple particle lists!");
            } else {
                useFlags = true;
            }
        }

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
                case MultiParticleDataCall::Particles::COLDATA_DOUBLE_I:
                case MultiParticleDataCall::Particles::COLDATA_FLOAT_I: {
                    glEnableVertexAttribArrayARB(cial);
                    if (parts.GetColourDataType() == MultiParticleDataCall::Particles::COLDATA_FLOAT_I) {
                        glVertexAttribPointerARB(
                            cial, 1, GL_FLOAT, GL_FALSE, parts.GetColourDataStride(), parts.GetColourData());
                    } else {
                        glVertexAttribPointerARB(
                            cial, 1, GL_DOUBLE, GL_FALSE, parts.GetColourDataStride(), parts.GetColourData());
                    }

                    glEnable(GL_TEXTURE_1D);
                    view::CallGetTransferFunction *cgtf = this->getTFSlot.CallAs<view::CallGetTransferFunction>();
                    if ((cgtf != nullptr) && ((*cgtf)())) {
                        glBindTexture(GL_TEXTURE_1D, cgtf->OpenGLTexture());
                        colTabSize = cgtf->TextureSize();
                    } else {
                        glBindTexture(GL_TEXTURE_1D, this->greyTF);
                        colTabSize = 2;
                    }

                    glUniform1i(this->arrowShader.ParameterLocation("colTab"), 0);
                    minC = parts.GetMinColourIndexValue();
                    maxC = parts.GetMaxColourIndexValue();
                    //glColor3ub(127, 127, 127);
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
                case MultiParticleDataCall::Particles::VERTDATA_DOUBLE_XYZ:
                    glEnableClientState(GL_VERTEX_ARRAY);
                    glUniform4f(this->arrowShader.ParameterLocation("inConsts1"), -1.0f, minC, maxC, float(colTabSize));
                    glVertexPointer(3, GL_DOUBLE, parts.GetVertexDataStride(), parts.GetVertexData());
                default:
                    continue;
            }

            // direction
            switch (parts.GetDirDataType()) {
                case MultiParticleDataCall::Particles::DIRDATA_FLOAT_XYZ:
                    glEnableVertexAttribArrayARB(tpal);
                    glVertexAttribPointerARB(tpal, 3, GL_FLOAT, GL_FALSE, parts.GetDirDataStride(), parts.GetDirData());
                    break;
                default:
                    vislib::sys::Log::DefaultLog.WriteWarn(
                        "ArrowRenderer: cannot render arrows without directional data!");
                    continue;
            }

            std::shared_ptr<FlagStorage::FlagVectorType> flags;
            unsigned int fal = 0;
            if (useFlags) {
                (*cflags)(core::FlagCall::CallMapFlags);
                cflags->validateFlagsCount(parts.GetCount());
                flags = cflags->GetFlags();
                fal = glGetAttribLocationARB(this->arrowShader, "flags");
                glEnableVertexAttribArrayARB(fal);
                glVertexAttribIPointer(fal, 1, GL_UNSIGNED_INT, 0, flags->data());
            }
            glUniform1ui(this->arrowShader.ParameterLocation("flagsAvailable"), useFlags ? 1 : 0);

            glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));

            if (useFlags) {
                glDisableVertexAttribArrayARB(fal);
                cflags->SetFlags(flags);
                (*cflags)(core::FlagCall::CallUnmapFlags);
                glVertexAttribIPointer(fal, 4, GL_FLOAT, 0, nullptr);
                glDisableVertexAttribArrayARB(fal);
            }

            glColorPointer(4, GL_FLOAT, 0, nullptr);
            glVertexPointer(4, GL_FLOAT, 0, nullptr);
            glDisableClientState(GL_COLOR_ARRAY);
            glDisableClientState(GL_VERTEX_ARRAY);

            if (parts.GetColourDataType() == MultiParticleDataCall::Particles::COLDATA_DOUBLE_I ||
                parts.GetColourDataType() == MultiParticleDataCall::Particles::COLDATA_FLOAT_I) {
                glVertexAttribPointerARB(cial, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
                glDisableVertexAttribArrayARB(cial);
            }
            glVertexAttribPointerARB(tpal, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
            glDisableVertexAttribArrayARB(tpal);
            glDisable(GL_TEXTURE_1D);
        }

        c2->Unlock();

    }

    this->arrowShader.Disable();

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);

    return true;
}
