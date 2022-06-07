/*
 * ArrowRenderer.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "ArrowRenderer.h"
#include "stdafx.h"

#include "mmcore/view/light/DistantLight.h"

#include <glm/ext.hpp>


#include "mmcore_gl/utility/ShaderSourceFactory.h"
#include "vislib_gl/graphics/gl/ShaderSource.h"

#include "OpenGL_Context.h"

#include "mmcore_gl/utility/ShaderFactory.h"

using namespace megamol::core;
using namespace megamol::geocalls;
using namespace megamol::moldyn_gl::rendering;


ArrowRenderer::ArrowRenderer(void)
        : core_gl::view::Renderer3DModuleGL()
        , getDataSlot("getdata", "Connects to the data source")
        , getTFSlot("gettransferfunction", "Connects to the transfer function module")
        , getFlagsSlot("getflags", "connects to a FlagStorage")
        , getClipPlaneSlot("getclipplane", "Connects to a clipping plane module")
        , lengthScaleSlot("lengthScale", "")
        , lengthFilterSlot("lengthFilter", "Filters the arrows by length")
        , arrowShader()
        , greyTF(0)
        , getLightsSlot("lights", "Lights are retrieved over this slot.") {

    this->getDataSlot.SetCompatibleCall<MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->getDataSlot);

    this->getTFSlot.SetCompatibleCall<core_gl::view::CallGetTransferFunctionGLDescription>();
    this->MakeSlotAvailable(&this->getTFSlot);

    this->getFlagsSlot.SetCompatibleCall<core_gl::FlagCallRead_GLDescription>();
    this->MakeSlotAvailable(&this->getFlagsSlot);

    this->getClipPlaneSlot.SetCompatibleCall<view::CallClipPlaneDescription>();
    this->MakeSlotAvailable(&this->getClipPlaneSlot);

    this->getLightsSlot.SetCompatibleCall<core::view::light::CallLightDescription>();
    this->MakeSlotAvailable(&this->getLightsSlot);

    this->lengthScaleSlot << new param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->lengthScaleSlot);

    this->lengthFilterSlot << new param::FloatParam(0.0f, 0.0);
    this->MakeSlotAvailable(&this->lengthFilterSlot);
}


ArrowRenderer::~ArrowRenderer(void) {

    this->Release();
}


bool ArrowRenderer::create(void) {
    auto const& ogl_ctx = frontend_resources.get<frontend_resources::OpenGL_Context>();
    if (!ogl_ctx.areExtAvailable(vislib_gl::graphics::gl::GLSLShader::RequiredExtensions()))
        return false;

#ifdef PROFILING
    auto& pm = const_cast<frontend_resources::PerformanceManager&>(
        frontend_resources.get<frontend_resources::PerformanceManager>());
    frontend_resources::PerformanceManager::basic_timer_config upload_timer, render_timer;
    upload_timer.name = "upload";
    upload_timer.api = frontend_resources::PerformanceManager::query_api::OPENGL;
    render_timer.name = "render";
    render_timer.api = frontend_resources::PerformanceManager::query_api::OPENGL;
    timing_handles_ = pm.add_timers(this, {upload_timer, render_timer});
#endif

    try {
        auto shdr_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());
        arrowShader_ = core::utility::make_glowl_shader("ArrowShader", shdr_options,
            std::filesystem::path("arrow/arrow.vert.glsl"), std::filesystem::path("arrow/arrow.frag.glsl"));
    } catch (glowl::GLSLProgramException const& ex) {
        core::utility::log::Log::DefaultLog.WriteError("[ArrowRenderer]: %s", ex.what());
        return false;
    } catch (...) {
        core::utility::log::Log::DefaultLog.WriteError("[ArrowRenderer] Failed to create program");
        return false;
    }

    vislib_gl::graphics::gl::ShaderSource vert, frag;
    auto ssf = std::make_shared<core_gl::utility::ShaderSourceFactory>(instance()->Configuration().ShaderDirectories());
    if (!ssf->MakeShaderSource("arrow::vertex", vert)) {
        return false;
    }
    if (!ssf->MakeShaderSource("arrow::fragment", frag)) {
        return false;
    }

    try {
        if (!this->arrowShader.Create(vert.Code(), vert.Count(), frag.Code(), frag.Count())) {
            megamol::core::utility::log::Log::DefaultLog.WriteMsg(
                megamol::core::utility::log::Log::LEVEL_ERROR, "Unable to compile arrow shader: Unknown error\n");
            return false;
        }

    } catch (vislib_gl::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(megamol::core::utility::log::Log::LEVEL_ERROR,
            "Unable to compile arrow shader (@%s): %s\n",
            vislib_gl::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
            ce.GetMsgA());
        return false;
    } catch (vislib::Exception e) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "Unable to compile arrow shader: %s\n", e.GetMsgA());
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "Unable to compile arrow shader: Unknown exception\n");
        return false;
    }

    glEnable(GL_TEXTURE_1D);
    glGenTextures(1, &this->greyTF);
    unsigned char tex[6] = {0, 0, 0, 255, 255, 255};
    glBindTexture(GL_TEXTURE_1D, this->greyTF);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, 2, 0, GL_RGB, GL_UNSIGNED_BYTE, tex);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glBindTexture(GL_TEXTURE_1D, 0);
    glDisable(GL_TEXTURE_1D);

    return true;
}


bool ArrowRenderer::GetExtents(core_gl::view::CallRender3DGL& call) {

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


bool ArrowRenderer::Render(core_gl::view::CallRender3DGL& call) {
#ifdef PROFILING
    auto& pm = const_cast<frontend_resources::PerformanceManager&>(
        frontend_resources.get<frontend_resources::PerformanceManager>());
#endif

    MultiParticleDataCall* c2 = this->getDataSlot.CallAs<MultiParticleDataCall>();
    if (c2 != nullptr) {
        c2->SetFrameID(static_cast<unsigned int>(call.Time()));
        if (!(*c2)(1))
            return false;
        c2->SetFrameID(static_cast<unsigned int>(call.Time()));
        if (!(*c2)(0))
            return false;

        if (in_data_hash_ != c2->DataHash() || in_frame_id_ != c2->FrameID()) {
#ifdef PROFILING
            pm.start_timer(timing_handles_[0], this->GetCoreInstance()->GetFrameID());
#endif
            loadData(*c2);
#ifdef PROFILING
            pm.stop_timer(timing_handles_[0]);
#endif
            in_data_hash_ = c2->DataHash();
            in_frame_id_ = c2->FrameID();
        }
    } else {
        return false;
    }

    auto* cflags = this->getFlagsSlot.CallAs<core_gl::FlagCallRead_GL>();

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
    core::view::Camera cam = call.GetCamera();
    auto cam_pose = cam.get<core::view::Camera::Pose>();
    glm::vec3 camView = cam_pose.direction;
    glm::vec3 camUp = cam_pose.up;
    glm::vec3 camRight = glm::cross(camView, camUp);
    auto fbo = call.GetFramebuffer();

    // Matrices
    auto view = cam.getViewMatrix();
    auto proj = cam.getProjectionMatrix();
    glm::mat4 MVinv = glm::inverse(view);
    glm::mat4 MVtransp = glm::transpose(view);
    glm::mat4 MVP = proj * view;
    glm::mat4 MVPinv = glm::inverse(MVP);
    glm::mat4 MVPtransp = glm::transpose(MVP);

    // Viewport
    glm::vec4 viewportStuff;
    viewportStuff[0] = 0.0f;
    viewportStuff[1] = 0.0f;
    viewportStuff[2] = static_cast<float>(fbo->getWidth());
    viewportStuff[3] = static_cast<float>(fbo->getHeight());
    if (viewportStuff[2] < 1.0f)
        viewportStuff[2] = 1.0f;
    if (viewportStuff[3] < 1.0f)
        viewportStuff[3] = 1.0f;
    viewportStuff[2] = 2.0f / viewportStuff[2];
    viewportStuff[3] = 2.0f / viewportStuff[3];

    // Lights
    glm::vec4 curlightDir = {0.0f, 0.0f, 0.0f, 1.0f};

    auto call_light = getLightsSlot.CallAs<core::view::light::CallLight>();
    if (call_light != nullptr) {
        if (!(*call_light)(0)) {
            return false;
        }

        auto lights = call_light->getData();
        auto distant_lights = lights.get<core::view::light::DistantLightType>();

        if (distant_lights.size() > 1) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "[ArrowRenderer] Only one single 'Distant Light' source is supported by this renderer");
        } else if (distant_lights.empty()) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn("[ArrowRenderer] No 'Distant Light' found");
        }

        for (auto const& light : distant_lights) {
            auto use_eyedir = light.eye_direction;
            if (use_eyedir) {
                curlightDir = glm::vec4(-camView, 1.0);
            } else {
                auto lightDir = light.direction;
                if (lightDir.size() == 3) {
                    curlightDir[0] = lightDir[0];
                    curlightDir[1] = lightDir[1];
                    curlightDir[2] = lightDir[2];
                }
                if (lightDir.size() == 4) {
                    curlightDir[3] = lightDir[3];
                }
                /// View Space Lighting. Comment line to change to Object Space Lighting.
                // this->curlightDir = this->curMVtransp * this->curlightDir;
            }
            /// TODO Implement missing distant light parameters:
            // light.second.dl_angularDiameter;
            // light.second.lightColor;
            // light.second.lightIntensity;
        }
    }

#ifdef PROFILING
    pm.set_transient_comment(timing_handles_[1], "std");
    if (this->GetCoreInstance()->GetFrameID() > 100)
        pm.start_timer(timing_handles_[1], this->GetCoreInstance()->GetFrameID());
#endif

    glDisable(GL_BLEND);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glPointSize(vislib::math::Max(viewportStuff[2], viewportStuff[3]));

    arrowShader_->use();
    //this->arrowShader.Enable();

    glUniformMatrix4fv(arrowShader_->getUniformLocation("MVinv"), 1, GL_FALSE, glm::value_ptr(MVinv));
    glUniformMatrix4fv(arrowShader_->getUniformLocation("MVtransp"), 1, GL_FALSE, glm::value_ptr(MVtransp));
    glUniformMatrix4fv(arrowShader_->getUniformLocation("MVP"), 1, GL_FALSE, glm::value_ptr(MVP));
    glUniformMatrix4fv(arrowShader_->getUniformLocation("MVPinv"), 1, GL_FALSE, glm::value_ptr(MVPinv));
    glUniformMatrix4fv(arrowShader_->getUniformLocation("MVPtransp"), 1, GL_FALSE, glm::value_ptr(MVPtransp));
    glUniform4fv(arrowShader_->getUniformLocation("viewAttr"), 1, glm::value_ptr(viewportStuff));
    glUniform3fv(arrowShader_->getUniformLocation("camIn"), 1, glm::value_ptr(camView));
    glUniform3fv(arrowShader_->getUniformLocation("camRight"), 1, glm::value_ptr(camRight));
    glUniform3fv(arrowShader_->getUniformLocation("camUp"), 1, glm::value_ptr(camUp));
    glUniform4fv(arrowShader_->getUniformLocation("lightDir"), 1, glm::value_ptr(curlightDir));
    arrowShader_->setUniform("lengthScale", lengthScale);
    arrowShader_->setUniform("lengthFilter", lengthFilter);
    glUniform4fv(arrowShader_->getUniformLocation("clipDat"), 1, clipDat);
    glUniform3fv(arrowShader_->getUniformLocation("clipCol"), 1, clipCol);

    if (c2 != nullptr) {
        unsigned int cial = glGetAttribLocationARB(this->arrowShader_->getHandle(), "colIdx");
        unsigned int tpal = glGetAttribLocationARB(this->arrowShader_->getHandle(), "dir");
        bool useFlags = false;

        if (cflags != nullptr) {
            if (c2->GetParticleListCount() > 1) {
                megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                    "ArrowRenderer: Cannot use FlagStorage together with multiple particle lists!");
            } else {
                useFlags = true;
            }
        }

        for (unsigned int i = 0; i < c2->GetParticleListCount(); i++) {
            MultiParticleDataCall::Particles& parts = c2->AccessParticles(i);
            float minC = 0.0f, maxC = 0.0f;
            unsigned int colTabSize = 0;

            // colour
            switch (parts.GetColourDataType()) {
            /*case MultiParticleDataCall::Particles::COLDATA_NONE:
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
                break;*/
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
                core_gl::view::CallGetTransferFunctionGL* cgtf =
                    this->getTFSlot.CallAs<core_gl::view::CallGetTransferFunctionGL>();
                if ((cgtf != nullptr) && ((*cgtf)())) {
                    glBindTexture(GL_TEXTURE_1D, cgtf->OpenGLTexture());
                    colTabSize = cgtf->TextureSize();
                } else {
                    glBindTexture(GL_TEXTURE_1D, this->greyTF);
                    colTabSize = 2;
                }

                glUniform1i(this->arrowShader_->getUniformLocation("colTab"), 0);
                minC = parts.GetMinColourIndexValue();
                maxC = parts.GetMaxColourIndexValue();
                //glColor3ub(127, 127, 127);
            } break;
            default:
                glColor3ub(127, 127, 127);
                break;
            }

            //// radius and position
            //switch (parts.GetVertexDataType()) {
            //case MultiParticleDataCall::Particles::VERTDATA_NONE:
            //    continue;
            //case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZ:
            //    glEnableClientState(GL_VERTEX_ARRAY);
            //    glUniform4f(this->arrowShader_->getUniformLocation("inConsts1"), parts.GetGlobalRadius(), minC, maxC,
            //        float(colTabSize));
            //    glVertexPointer(3, GL_FLOAT, parts.GetVertexDataStride(), parts.GetVertexData());
            //    break;
            //case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
            //    glEnableClientState(GL_VERTEX_ARRAY);
            //    glUniform4f(arrowShader_->getUniformLocation("inConsts1"), -1.0f, minC, maxC, float(colTabSize));
            //    glVertexPointer(4, GL_FLOAT, parts.GetVertexDataStride(), parts.GetVertexData());
            //    break;
            //case MultiParticleDataCall::Particles::VERTDATA_DOUBLE_XYZ:
            //    glEnableClientState(GL_VERTEX_ARRAY);
            //    glUniform4f(arrowShader_->getUniformLocation("inConsts1"), -1.0f, minC, maxC, float(colTabSize));
            //    glVertexPointer(3, GL_DOUBLE, parts.GetVertexDataStride(), parts.GetVertexData());
            //default:
            //    continue;
            //}

            //// direction
            //switch (parts.GetDirDataType()) {
            //case MultiParticleDataCall::Particles::DIRDATA_FLOAT_XYZ:
            //    glEnableVertexAttribArrayARB(tpal);
            //    glVertexAttribPointerARB(tpal, 3, GL_FLOAT, GL_FALSE, parts.GetDirDataStride(), parts.GetDirData());
            //    break;
            //default:
            //    megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            //        "ArrowRenderer: cannot render arrows without directional data!");
            //    continue;
            //}

            unsigned int fal = 0;
            if (useFlags) {
                (*cflags)(core_gl::FlagCallRead_GL::CallGetData);
                cflags->getData()->validateFlagCount(parts.GetCount());
                auto flags = cflags->getData();
                fal = glGetAttribLocationARB(this->arrowShader, "flags");
                glEnableVertexAttribArrayARB(fal);
                // TODO highly unclear whether this works fine
                glBindBuffer(GL_ARRAY_BUFFER, flags->flags->getName());
                //flags->flags->bindAs(GL_ARRAY_BUFFER);
                glVertexAttribIPointer(fal, 1, GL_UNSIGNED_INT, 0, nullptr);
            }
            glUniform1ui(arrowShader_->getUniformLocation("flagsAvailable"), useFlags ? 1 : 0);

            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, data_buf_[i]);
            glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));

            if (useFlags) {
                glDisableVertexAttribArrayARB(fal);
                //cflags->SetFlags(flags);
                //(*cflags)(core::FlagCall::CallUnmapFlags);
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

    glUseProgram(0);
    //this->arrowShader.Disable();

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);

#ifdef PROFILING
    if (this->GetCoreInstance()->GetFrameID() > 100)
        pm.stop_timer(timing_handles_[1]);
#endif

    return true;
}


void megamol::moldyn_gl::rendering::ArrowRenderer::loadData(geocalls::MultiParticleDataCall& in_data) {
    auto const pl_count = in_data.GetParticleListCount();

    glDeleteBuffers(data_buf_.size(), data_buf_.data());
    data_buf_.resize(pl_count);
    glCreateBuffers(data_buf_.size(), data_buf_.data());

    for (std::decay_t<decltype(pl_count)> pl_idx = 0; pl_idx < pl_count; ++pl_idx) {
        auto const& parts = in_data.AccessParticles(pl_idx);

        auto const xAcc = parts.GetParticleStore().GetXAcc();
        auto const yAcc = parts.GetParticleStore().GetYAcc();
        auto const zAcc = parts.GetParticleStore().GetZAcc();
        auto const radAcc = parts.GetParticleStore().GetRAcc();

        auto const crAcc = parts.GetParticleStore().GetCRAcc();
        auto const cgAcc = parts.GetParticleStore().GetCGAcc();
        auto const cbAcc = parts.GetParticleStore().GetCBAcc();
        auto const caAcc = parts.GetParticleStore().GetCAAcc();

        auto const dxAcc = parts.GetParticleStore().GetDXAcc();
        auto const dyAcc = parts.GetParticleStore().GetDYAcc();
        auto const dzAcc = parts.GetParticleStore().GetDZAcc();

        auto const p_count = parts.GetCount();

        std::vector<float> data;
        data.reserve(p_count * 8);

        for (std::decay_t<decltype(p_count)> p_idx = 0; p_idx < p_count; ++p_idx) {
            data.push_back(xAcc->Get_f(p_idx));
            data.push_back(yAcc->Get_f(p_idx));
            data.push_back(zAcc->Get_f(p_idx));
            data.push_back(radAcc->Get_f(p_idx));
            unsigned int col = glm::packUnorm4x8(
                glm::vec4(crAcc->Get_f(p_idx), cgAcc->Get_f(p_idx), cbAcc->Get_f(p_idx), caAcc->Get_f(p_idx)));
            data.push_back(*reinterpret_cast<float*>(&col));
            data.push_back(dxAcc->Get_f(p_idx));
            data.push_back(dyAcc->Get_f(p_idx));
            data.push_back(dzAcc->Get_f(p_idx));
        }

        glNamedBufferStorage(
            data_buf_[pl_idx], data.size() * sizeof(std::decay_t<decltype(data)>::value_type), data.data(), 0);
    }
}
