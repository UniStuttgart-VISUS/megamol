/*
 * ArrowRenderer.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "ArrowRenderer.h"

#include "mmcore/view/light/DistantLight.h"

#include <glm/ext.hpp>

#include "OpenGL_Context.h"

#ifdef PROFILING
#include "PerformanceManager.h"
#endif

using namespace megamol::core;
using namespace megamol::geocalls;
using namespace megamol::moldyn_gl::rendering;


ArrowRenderer::ArrowRenderer(void)
        : core_gl::view::Renderer3DModuleGL()
        , get_data_slot_("getdata", "Connects to the data source")
        , get_tf_slot_("gettransferfunction", "Connects to the transfer function module")
        , get_flags_slot_("getflags", "connects to a FlagStorage")
        , get_clip_plane_slot_("getclipplane", "Connects to a clipping plane module")
        , length_scale_slot_("length_scale", "")
        , length_filter_slot_("length_filter", "Filters the arrows by length")
        , arrow_pgrm_()
        , grey_tf_(0)
        , get_lights_slot_("lights", "Lights are retrieved over this slot.") {

    this->get_data_slot_.SetCompatibleCall<MultiParticleDataCallDescription>();
    this->MakeSlotAvailable(&this->get_data_slot_);

    this->get_tf_slot_.SetCompatibleCall<core_gl::view::CallGetTransferFunctionGLDescription>();
    this->MakeSlotAvailable(&this->get_tf_slot_);

    this->get_flags_slot_.SetCompatibleCall<core_gl::FlagCallRead_GLDescription>();
    this->MakeSlotAvailable(&this->get_flags_slot_);

    this->get_clip_plane_slot_.SetCompatibleCall<view::CallClipPlaneDescription>();
    this->MakeSlotAvailable(&this->get_clip_plane_slot_);

    this->get_lights_slot_.SetCompatibleCall<core::view::light::CallLightDescription>();
    this->MakeSlotAvailable(&this->get_lights_slot_);

    this->length_scale_slot_ << new param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->length_scale_slot_);

    this->length_filter_slot_ << new param::FloatParam(0.0f, 0.0);
    this->MakeSlotAvailable(&this->length_filter_slot_);
}


ArrowRenderer::~ArrowRenderer(void) {

    this->Release();
}


bool ArrowRenderer::create(void) {
#ifdef PROFILING
    perf_manager_ = const_cast<frontend_resources::PerformanceManager*>(
        &frontend_resources.get<frontend_resources::PerformanceManager>());

    frontend_resources::PerformanceManager::basic_timer_config render_timer;
    render_timer.name = "render";
    render_timer.api = frontend_resources::PerformanceManager::query_api::OPENGL;
    timers_ = perf_manager_->add_timers(this, {render_timer});
#endif

    auto const& ogl_ctx = frontend_resources.get<frontend_resources::OpenGL_Context>();
    if (!ogl_ctx.areExtAvailable(vislib_gl::graphics::gl::GLSLShader::RequiredExtensions()))
        return false;

    // create shader programs
    auto const shader_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());

    try {
        // TODO: use std::filesystem::path?
        arrow_pgrm_ = core::utility::make_glowl_shader(
            "arrow", shader_options, "arrow_renderer/arrow.vert.glsl", "arrow_renderer/arrow.frag.glsl");
    } catch (std::exception& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "Unable to compile arrow shader: %s. [%s, %s, line %d]\n", std::string(e.what()).c_str(),
            __FILE__, __FUNCTION__, __LINE__);

        return false;
    }


    glEnable(GL_TEXTURE_1D);
    glGenTextures(1, &this->grey_tf_);
    unsigned char tex[6] = {0, 0, 0, 255, 255, 255};
    glBindTexture(GL_TEXTURE_1D, this->grey_tf_);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, 2, 0, GL_RGB, GL_UNSIGNED_BYTE, tex);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glBindTexture(GL_TEXTURE_1D, 0);
    glDisable(GL_TEXTURE_1D);

    return true;
}


bool ArrowRenderer::GetExtents(core_gl::view::CallRender3DGL& call) {

    MultiParticleDataCall* c2 = this->get_data_slot_.CallAs<MultiParticleDataCall>();
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
    glDeleteTextures(1, &this->grey_tf_);

#ifdef PROFILING
    perf_manager_->remove_timers(timers_);
#endif
}


bool ArrowRenderer::Render(core_gl::view::CallRender3DGL& call) {

    MultiParticleDataCall* c2 = this->get_data_slot_.CallAs<MultiParticleDataCall>();
    if (c2 != nullptr) {
        c2->SetFrameID(static_cast<unsigned int>(call.Time()));
        if (!(*c2)(1))
            return false;
        c2->SetFrameID(static_cast<unsigned int>(call.Time()));
        if (!(*c2)(0))
            return false;
    } else {
        return false;
    }

    auto* cflags = this->get_flags_slot_.CallAs<core_gl::FlagCallRead_GL>();

    float length_scale = this->length_scale_slot_.Param<param::FloatParam>()->Value();
    float length_filter = this->length_filter_slot_.Param<param::FloatParam>()->Value();

    // Clipping
    auto ccp = this->get_clip_plane_slot_.CallAs<view::CallClipPlane>();
    float clip_dat[4];
    float clip_col[4];
    if ((ccp != nullptr) && (*ccp)()) {
        clip_dat[0] = ccp->GetPlane().Normal().X();
        clip_dat[1] = ccp->GetPlane().Normal().Y();
        clip_dat[2] = ccp->GetPlane().Normal().Z();
        vislib::math::Vector<float, 3> grr(ccp->GetPlane().Point().PeekCoordinates());
        clip_dat[3] = grr.Dot(ccp->GetPlane().Normal());
        clip_col[0] = static_cast<float>(ccp->GetColour()[0]) / 255.0f;
        clip_col[1] = static_cast<float>(ccp->GetColour()[1]) / 255.0f;
        clip_col[2] = static_cast<float>(ccp->GetColour()[2]) / 255.0f;
        clip_col[3] = static_cast<float>(ccp->GetColour()[3]) / 255.0f;
    } else {
        clip_dat[0] = clip_dat[1] = clip_dat[2] = clip_dat[3] = 0.0f;
        clip_col[0] = clip_col[1] = clip_col[2] = 0.75f;
        clip_col[3] = 1.0f;
    }

    // Camera
    core::view::Camera cam = call.GetCamera();
    auto cam_pose = cam.get<core::view::Camera::Pose>();
    glm::vec3 cam_view = cam_pose.direction;
    glm::vec3 cam_up = cam_pose.up;
    glm::vec3 cam_right = glm::cross(cam_view, cam_up);
    auto fbo = call.GetFramebuffer();

    // Matrices
    auto view = cam.getViewMatrix();
    auto proj = cam.getProjectionMatrix();
    glm::mat4 mv_inv = glm::inverse(view);
    glm::mat4 mv_transp = glm::transpose(view);
    glm::mat4 mvp = proj * view;
    glm::mat4 mvp_inv = glm::inverse(mvp);
    glm::mat4 mvp_transp = glm::transpose(mvp);

    // Viewport
    glm::vec4 viewport_stuff;
    viewport_stuff[0] = 0.0f;
    viewport_stuff[1] = 0.0f;
    viewport_stuff[2] = static_cast<float>(fbo->getWidth());
    viewport_stuff[3] = static_cast<float>(fbo->getHeight());
    if (viewport_stuff[2] < 1.0f)
        viewport_stuff[2] = 1.0f;
    if (viewport_stuff[3] < 1.0f)
        viewport_stuff[3] = 1.0f;
    viewport_stuff[2] = 2.0f / viewport_stuff[2];
    viewport_stuff[3] = 2.0f / viewport_stuff[3];

    // Lights
    glm::vec4 cur_light_dir = {0.0f, 0.0f, 0.0f, 1.0f};

    auto call_light = get_lights_slot_.CallAs<core::view::light::CallLight>();
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
                cur_light_dir = glm::vec4(-cam_view, 1.0);
            } else {
                auto light_dir = light.direction;
                if (light_dir.size() == 3) {
                    cur_light_dir[0] = light_dir[0];
                    cur_light_dir[1] = light_dir[1];
                    cur_light_dir[2] = light_dir[2];
                }
                if (light_dir.size() == 4) {
                    cur_light_dir[3] = light_dir[3];
                }
                /// View Space Lighting. Comment line to change to Object Space Lighting.
                // this->cur_light_dir = this->curMVtransp * this->cur_light_dir;
            }
            /// TODO Implement missing distant light parameters:
            // light.second.dl_angularDiameter;
            // light.second.lightColor;
            // light.second.lightIntensity;
        }
    }

    glDisable(GL_BLEND);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glPointSize(vislib::math::Max(viewport_stuff[2], viewport_stuff[3]));

    this->arrow_pgrm_->use();

    glUniformMatrix4fv(this->arrow_pgrm_->getUniformLocation("MVinv"), 1, GL_FALSE, glm::value_ptr(mv_inv));
    glUniformMatrix4fv(this->arrow_pgrm_->getUniformLocation("MVtransp"), 1, GL_FALSE, glm::value_ptr(mv_transp));
    glUniformMatrix4fv(this->arrow_pgrm_->getUniformLocation("MVP"), 1, GL_FALSE, glm::value_ptr(mvp));
    glUniformMatrix4fv(this->arrow_pgrm_->getUniformLocation("MVPinv"), 1, GL_FALSE, glm::value_ptr(mvp_inv));
    glUniformMatrix4fv(this->arrow_pgrm_->getUniformLocation("MVPtransp"), 1, GL_FALSE, glm::value_ptr(mvp_transp));
    glUniform4fv(this->arrow_pgrm_->getUniformLocation("viewAttr"), 1, glm::value_ptr(viewport_stuff));
    glUniform3fv(this->arrow_pgrm_->getUniformLocation("camIn"), 1, glm::value_ptr(cam_view));
    glUniform3fv(this->arrow_pgrm_->getUniformLocation("camRight"), 1, glm::value_ptr(cam_right));
    glUniform3fv(this->arrow_pgrm_->getUniformLocation("camUp"), 1, glm::value_ptr(cam_up));
    glUniform4fv(this->arrow_pgrm_->getUniformLocation("lightDir"), 1, glm::value_ptr(cur_light_dir));
    glUniform1f(this->arrow_pgrm_->getUniformLocation("lengthScale"), length_scale);
    glUniform1f(this->arrow_pgrm_->getUniformLocation("lengthFilter"), length_filter);
    glUniform4fv(this->arrow_pgrm_->getUniformLocation("clipDat"), 1, clip_dat);
    glUniform3fv(this->arrow_pgrm_->getUniformLocation("clipCol"), 1, clip_col);

    if (c2 != nullptr) {
        unsigned int cial = glGetAttribLocationARB(this->arrow_pgrm_->getHandle(), "colIdx");
        unsigned int tpal = glGetAttribLocationARB(this->arrow_pgrm_->getHandle(), "dir");
        bool use_flags = false;

        if (cflags != nullptr) {
            if (c2->GetParticleListCount() > 1) {
                megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                    "ArrowRenderer: Cannot use FlagStorage together with multiple particle lists!");
            } else {
                use_flags = true;
            }
        }

        for (unsigned int i = 0; i < c2->GetParticleListCount(); i++) {
            MultiParticleDataCall::Particles& parts = c2->AccessParticles(i);
            float min_c = 0.0f, max_c = 0.0f;
            unsigned int col_tab_size = 0;

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
                core_gl::view::CallGetTransferFunctionGL* cgtf =
                    this->get_tf_slot_.CallAs<core_gl::view::CallGetTransferFunctionGL>();
                if ((cgtf != nullptr) && ((*cgtf)())) {
                    glBindTexture(GL_TEXTURE_1D, cgtf->OpenGLTexture());
                    col_tab_size = cgtf->TextureSize();
                } else {
                    glBindTexture(GL_TEXTURE_1D, this->grey_tf_);
                    col_tab_size = 2;
                }

                glUniform1i(this->arrow_pgrm_->getUniformLocation("colTab"), 0);
                min_c = parts.GetMinColourIndexValue();
                max_c = parts.GetMaxColourIndexValue();
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
                glUniform4f(this->arrow_pgrm_->getUniformLocation("inConsts1"), parts.GetGlobalRadius(), min_c, max_c,
                    float(col_tab_size));
                glVertexPointer(3, GL_FLOAT, parts.GetVertexDataStride(), parts.GetVertexData());
                break;
            case MultiParticleDataCall::Particles::VERTDATA_FLOAT_XYZR:
                glEnableClientState(GL_VERTEX_ARRAY);
                glUniform4f(this->arrow_pgrm_->getUniformLocation("inConsts1"), -1.0f, min_c, max_c, float(col_tab_size));
                glVertexPointer(4, GL_FLOAT, parts.GetVertexDataStride(), parts.GetVertexData());
                break;
            case MultiParticleDataCall::Particles::VERTDATA_DOUBLE_XYZ:
                glEnableClientState(GL_VERTEX_ARRAY);
                glUniform4f(this->arrow_pgrm_->getUniformLocation("inConsts1"), -1.0f, min_c, max_c, float(col_tab_size));
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
                megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                    "ArrowRenderer: cannot render arrows without directional data!");
                continue;
            }

            unsigned int fal = 0;
            if (use_flags) {
                (*cflags)(core_gl::FlagCallRead_GL::CallGetData);
                cflags->getData()->validateFlagCount(parts.GetCount());
                auto flags = cflags->getData();
                fal = glGetAttribLocationARB(this->arrow_pgrm_->getHandle(), "flags");
                glEnableVertexAttribArrayARB(fal);
                // TODO highly unclear whether this works fine
                flags->flags->bind(GL_ARRAY_BUFFER);
                //flags->flags->bindAs(GL_ARRAY_BUFFER);
                glVertexAttribIPointer(fal, 1, GL_UNSIGNED_INT, 0, nullptr);
            }
            glUniform1ui(this->arrow_pgrm_->getUniformLocation("flagsAvailable"), use_flags ? 1 : 0);

#ifdef PROFILING
            perf_manager_->start_timer(timers_[0], this->GetCoreInstance()->GetFrameID());
#endif

            glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(parts.GetCount()));

#ifdef PROFILING
            perf_manager_->stop_timer(timers_[0]);
#endif

            if (use_flags) {
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

    glDisable(GL_DEPTH_TEST);
    glDisable(GL_VERTEX_PROGRAM_POINT_SIZE);

    return true;
}
