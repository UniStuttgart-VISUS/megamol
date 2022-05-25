/*
 * RaycastVolumeRenderer.h
 *
 * Copyright (C) 2018-2020 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "RaycastVolumeRenderer.h"

#include "OpenGL_Context.h"

#include "geometry_calls//VolumetricDataCall.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmstd_gl/renderer/CallGetTransferFunctionGL.h"

#include "vislib_gl/graphics/gl/ShaderSource.h"

#include "glowl/Texture.hpp"
#include "glowl/Texture2D.hpp"
#include "glowl/Texture3D.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include <glm/ext.hpp>

#include "mmcore/view/light/DistantLight.h"

using namespace megamol::volume_gl;

RaycastVolumeRenderer::RaycastVolumeRenderer()
        : mmstd_gl::Renderer3DModuleGL()
        , m_mode("mode", "Mode changing the behavior for the raycaster")
        , m_ray_step_ratio_param("ray step ratio", "Adjust sampling rate")
        , m_use_lighting_slot("lighting::use lighting", "Enable simple volumetric illumination")
        , m_ka_slot("lighting::ka", "Ambient part for Phong lighting")
        , m_kd_slot("lighting::kd", "Diffuse part for Phong lighting")
        , m_ks_slot("lighting::ks", "Specular part for Phong lighting")
        , m_shininess_slot("lighting::shininess", "Shininess for Phong lighting")
        , m_ambient_color("lighting::ambient color", "Ambient color")
        , m_specular_color("lighting::specular color", "Specular color")
        , m_material_color("lighting::material color", "Material color")
        , m_opacity_threshold("opacity threshold", "Opacity threshold for integrative rendering")
        , m_iso_value("isovalue", "Isovalue for isosurface rendering")
        , m_opacity("opacity", "Surface opacity for blending")
        , m_renderer_callerSlot("Renderer", "Renderer for chaining")
        , m_volumetricData_callerSlot("getData", "Connects the volume renderer with a voluemtric data source")
        , m_lights_callerSlot("lights", "Lights are retrieved over this slot.")
        , m_transferFunction_callerSlot(
              "getTransferFunction", "Connects the volume renderer with a transfer function") {

    this->m_renderer_callerSlot.SetCompatibleCall<mmstd_gl::CallRender3DGLDescription>();
    this->MakeSlotAvailable(&this->m_renderer_callerSlot);

    this->m_volumetricData_callerSlot.SetCompatibleCall<geocalls::VolumetricDataCallDescription>();
    this->MakeSlotAvailable(&this->m_volumetricData_callerSlot);

    this->m_lights_callerSlot.SetCompatibleCall<megamol::core::view::light::CallLightDescription>();
    this->MakeSlotAvailable(&this->m_lights_callerSlot);

    this->m_transferFunction_callerSlot.SetCompatibleCall<megamol::mmstd_gl::CallGetTransferFunctionGLDescription>();
    this->MakeSlotAvailable(&this->m_transferFunction_callerSlot);

    this->m_mode << new megamol::core::param::EnumParam(0);
    this->m_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(0, "Integration");
    this->m_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(1, "Isosurface");
    this->m_mode.Param<core::param::EnumParam>()->SetTypePair(2, "Aggregate");
    this->MakeSlotAvailable(&this->m_mode);

    this->m_ray_step_ratio_param << new megamol::core::param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->m_ray_step_ratio_param);

    this->m_opacity_threshold << new megamol::core::param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->m_opacity_threshold);

    this->m_iso_value << new megamol::core::param::FloatParam(0.5f);
    this->MakeSlotAvailable(&this->m_iso_value);

    this->m_opacity << new megamol::core::param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->m_opacity);

    this->m_use_lighting_slot << new core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->m_use_lighting_slot);

    this->m_ka_slot << new core::param::FloatParam(0.1f, 0.0f);
    this->MakeSlotAvailable(&this->m_ka_slot);

    this->m_kd_slot << new core::param::FloatParam(0.5f, 0.0f);
    this->MakeSlotAvailable(&this->m_kd_slot);

    this->m_ks_slot << new core::param::FloatParam(0.4f, 0.0f);
    this->MakeSlotAvailable(&this->m_ks_slot);

    this->m_shininess_slot << new core::param::FloatParam(10.0f, 0.0f);
    this->MakeSlotAvailable(&this->m_shininess_slot);

    this->m_ambient_color << new core::param::ColorParam(1.0f, 1.0f, 1.0f, 1.0f);
    this->MakeSlotAvailable(&this->m_ambient_color);

    this->m_specular_color << new core::param::ColorParam(1.0f, 1.0f, 1.0f, 1.0f);
    this->MakeSlotAvailable(&this->m_specular_color);

    this->m_material_color << new core::param::ColorParam(0.95f, 0.67f, 0.47f, 1.0f);
    this->MakeSlotAvailable(&this->m_material_color);
}

RaycastVolumeRenderer::~RaycastVolumeRenderer() {
    this->Release();
}

bool RaycastVolumeRenderer::create() {
    auto const& ogl_ctx = frontend_resources.get<frontend_resources::OpenGL_Context>();
    if (!ogl_ctx.isVersionGEQ(4, 3))
        return false;

    try {
        // create shader program
        auto const shdr_cp_options = msf::ShaderFactoryOptionsOpenGL(this->GetCoreInstance()->GetShaderPaths());

        rvc_dvr_shdr = core::utility::make_glowl_shader("RaycastVolumeRenderer-Compute", shdr_cp_options,
            std::filesystem::path("RaycastVolumeRenderer-Compute.comp.glsl"));
        rvc_iso_shdr = core::utility::make_glowl_shader("RaycastVolumeRenderer-Compute-Iso", shdr_cp_options,
            std::filesystem::path("RaycastVolumeRenderer-Compute-Iso.comp.glsl"));
        rvc_aggr_shdr = core::utility::make_glowl_shader("RaycastVolumeRenderer-Compute-Aggr", shdr_cp_options,
            std::filesystem::path("RaycastVolumeRenderer-Compute-Aggr.comp.glsl"));

        rtf_shdr = core::utility::make_glowl_shader("RaycastVolumeRenderer", shdr_cp_options,
            std::filesystem::path("RaycastVolumeRenderer-Vertex.vert.glsl"),
            std::filesystem::path("RaycastVolumeRenderer-Fragment.frag.glsl"));
        rtf_aggr_shdr = core::utility::make_glowl_shader("RaycastVolumeRenderer-Aggr", shdr_cp_options,
            std::filesystem::path("RaycastVolumeRenderer-Vertex.vert.glsl"),
            std::filesystem::path("RaycastVolumeRenderer-Fragment-Aggr.frag.glsl"));
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteMsg(
            megamol::core::utility::log::Log::LEVEL_ERROR, "Unable to compile shader: Unknown exception\n");
        return false;
    }

    return true;
}

void RaycastVolumeRenderer::release() {}

bool RaycastVolumeRenderer::GetExtents(mmstd_gl::CallRender3DGL& cr) {
    auto cd = m_volumetricData_callerSlot.CallAs<geocalls::VolumetricDataCall>();
    auto ci = m_renderer_callerSlot.CallAs<mmstd_gl::CallRender3DGL>();

    if (cd == nullptr)
        return false;

    // TODO Do something about time/framecount ?

    int const req_frame = static_cast<int>(cr.Time());

    cd->SetFrameID(req_frame);

    if (!(*cd)(geocalls::VolumetricDataCall::IDX_GET_EXTENTS))
        return false;
    if (!(*cd)(geocalls::VolumetricDataCall::IDX_GET_METADATA))
        return false;

    cr.SetTimeFramesCount(cd->FrameCount());

    auto bbox = cd->AccessBoundingBoxes().ObjectSpaceBBox();
    auto cbox = cd->AccessBoundingBoxes().ObjectSpaceClipBox();

    if (ci != nullptr) {
        *ci = cr;

        if (!(*ci)(mmstd_gl::CallRender3DGL::FnGetExtents))
            return false;

        bbox.Union(ci->AccessBoundingBoxes().BoundingBox());
        cbox.Union(ci->AccessBoundingBoxes().ClipBox());
    }

    cr.AccessBoundingBoxes().SetBoundingBox(bbox);
    cr.AccessBoundingBoxes().SetClipBox(cbox);

    return true;
}

bool RaycastVolumeRenderer::Render(mmstd_gl::CallRender3DGL& cr) {
    // Chain renderer
    auto ci = m_renderer_callerSlot.CallAs<mmstd_gl::CallRender3DGL>();

    // Camera
    core::view::Camera cam = cr.GetCamera();
    auto view = cam.getViewMatrix();
    auto proj = cam.getProjectionMatrix();
    auto cam_pose = cam.get<core::view::Camera::Pose>();
    auto cr_fbo = cr.GetFramebuffer();

    if (ci != nullptr) {
        auto cam = cr.GetCamera();
        ci->SetCamera(cam);

        if (this->m_mode.Param<core::param::EnumParam>()->Value() == 0 ||
            this->m_mode.Param<core::param::EnumParam>()->Value() == 2) {
            if (this->fbo.IsValid())
                this->fbo.Release();
            this->fbo.Create(cr_fbo->getWidth(), cr_fbo->getHeight(), GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE,
                vislib_gl::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE);
            this->fbo.Enable();
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        ci->SetTime(cr.Time());
        if (!(*ci)(mmstd_gl::CallRender3DGL::FnRender))
            return false;

        if (this->m_mode.Param<core::param::EnumParam>()->Value() == 0 ||
            this->m_mode.Param<core::param::EnumParam>()->Value() == 2) {
            this->fbo.Disable();
        }
    }

    // create render target texture
    if (this->m_render_target == nullptr || this->m_render_target->getWidth() != cr_fbo->getWidth() ||
        this->m_render_target->getHeight() != cr_fbo->getHeight()) {

        glowl::TextureLayout render_tgt_layout(GL_RGBA8, cr_fbo->getWidth(), cr_fbo->getHeight(), 1, GL_RGBA,
            GL_UNSIGNED_BYTE, 1,
            {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
                {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_MIN_FILTER, GL_LINEAR},
                {GL_TEXTURE_MAG_FILTER, GL_LINEAR}},
            {});
        m_render_target =
            std::make_unique<glowl::Texture2D>("raycast_volume_render_target", render_tgt_layout, nullptr);

        // create normal target texture
        glowl::TextureLayout normal_tgt_layout(GL_RGBA32F, cr_fbo->getWidth(), cr_fbo->getHeight(), 1, GL_RGBA,
            GL_FLOAT, 1,
            {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
                {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_MIN_FILTER, GL_LINEAR},
                {GL_TEXTURE_MAG_FILTER, GL_LINEAR}},
            {});
        m_normal_target =
            std::make_unique<glowl::Texture2D>("raycast_volume_normal_target", normal_tgt_layout, nullptr);

        // create depth target texture
        glowl::TextureLayout depth_tgt_layout(GL_R32F, cr_fbo->getWidth(), cr_fbo->getHeight(), 1, GL_R, GL_FLOAT, 1,
            {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
                {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_MIN_FILTER, GL_LINEAR},
                {GL_TEXTURE_MAG_FILTER, GL_LINEAR}},
            {});
        m_depth_target = std::make_unique<glowl::Texture2D>("raycast_volume_depth_target", depth_tgt_layout, nullptr);
    }

    // this is the apex of suck and must die
    // std::array<float, 4> light = {0.0f, 0.0f, 1.0f, 1.0f};
    // glGetLightfv(GL_LIGHT0, GL_POSITION, light.data());
    // end suck

    if (!updateVolumeData(cr.Time()))
        return false;

    // enable raycast volume rendering program
    glowl::GLSLProgram* compute_shdr;

    // pick shader based on selected mode
    if (this->m_mode.Param<core::param::EnumParam>()->Value() == 0) {
        if (!updateTransferFunction())
            return false;

        compute_shdr = rvc_dvr_shdr.get();
    } else if (this->m_mode.Param<core::param::EnumParam>()->Value() == 1) {
        compute_shdr = rvc_iso_shdr.get();
    } else if (this->m_mode.Param<core::param::EnumParam>()->Value() == 2) {
        if (!updateTransferFunction())
            return false;
        compute_shdr = rvc_aggr_shdr.get();
    } else {
        megamol::core::utility::log::Log::DefaultLog.WriteError("Unknown raycast mode.");
        return false;
    }

    // Lights
    auto curlight = core::view::light::DistantLightType{{{1.0f, 1.0f, 1.0f}, 1.0f}, {1.0f, 0.0f, 0.0f}, 0.0f, true};
    auto call_light = m_lights_callerSlot.CallAs<core::view::light::CallLight>();
    if (call_light != nullptr) {
        if (!(*call_light)(0)) {
            return false;
        }

        auto lights = call_light->getData();
        auto distant_lights = lights.get<core::view::light::DistantLightType>();

        if (!distant_lights.empty()) {
            for (auto& l : distant_lights) {
                const auto use_eyedir = l.eye_direction;
                if (use_eyedir) {
                    auto view_vec = cam_pose.direction;
                    l.direction[0] = view_vec[0];
                    l.direction[1] = view_vec[1];
                    l.direction[2] = view_vec[2];
                }
            }
            curlight = distant_lights[0];
        }

        if (distant_lights.size() > 1) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "[RaycastVolumeRenderer] Only one single 'Distant Light' source is supported by this renderer");
        } else if (distant_lights.empty()) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn("[RaycastVolumeRenderer] No 'Distant Light' found");
        }
    }

    // setup
    compute_shdr->use();

    compute_shdr->setUniform("view_mx", view);
    compute_shdr->setUniform("proj_mx", proj);

    glm::vec2 rt_resolution;
    rt_resolution[0] = static_cast<float>(m_render_target->getWidth());
    rt_resolution[1] = static_cast<float>(m_render_target->getHeight());
    compute_shdr->setUniform("rt_resolution", rt_resolution);

    glm::vec3 box_min;
    box_min[0] = m_volume_origin[0];
    box_min[1] = m_volume_origin[1];
    box_min[2] = m_volume_origin[2];
    glm::vec3 box_max;
    box_max[0] = m_volume_origin[0] + m_volume_extents[0];
    box_max[1] = m_volume_origin[1] + m_volume_extents[1];
    box_max[2] = m_volume_origin[2] + m_volume_extents[2];
    compute_shdr->setUniform("boxMin", box_min);
    compute_shdr->setUniform("boxMax", box_max);

    compute_shdr->setUniform("halfVoxelSize", 1.0f / (2.0f * (m_volume_resolution[0] - 1)),
        1.0f / (2.0f * (m_volume_resolution[1] - 1)), 1.0f / (2.0f * (m_volume_resolution[2] - 1)));
    auto const maxResolution =
        std::max(m_volume_resolution[0], std::max(m_volume_resolution[1], m_volume_resolution[2]));
    auto const maxExtents = std::max(m_volume_extents[0], std::max(m_volume_extents[1], m_volume_extents[2]));
    compute_shdr->setUniform("voxelSize", maxExtents / (maxResolution - 1.0f));

    compute_shdr->setUniform("valRange", valRange[0], valRange[1]);

    compute_shdr->setUniform("rayStepRatio", this->m_ray_step_ratio_param.Param<core::param::FloatParam>()->Value());

    compute_shdr->setUniform("use_lighting", this->m_use_lighting_slot.Param<core::param::BoolParam>()->Value());
    compute_shdr->setUniform("ka", this->m_ka_slot.Param<core::param::FloatParam>()->Value());
    compute_shdr->setUniform("kd", this->m_kd_slot.Param<core::param::FloatParam>()->Value());
    compute_shdr->setUniform("ks", this->m_ks_slot.Param<core::param::FloatParam>()->Value());
    compute_shdr->setUniform("shininess", this->m_shininess_slot.Param<core::param::FloatParam>()->Value());
    compute_shdr->setUniform("light", curlight.direction[0], curlight.direction[1], curlight.direction[2]);
    compute_shdr->setUniform("ambient_col", this->m_ambient_color.Param<core::param::ColorParam>()->Value()[0],
        this->m_ambient_color.Param<core::param::ColorParam>()->Value()[1],
        this->m_ambient_color.Param<core::param::ColorParam>()->Value()[2]);
    compute_shdr->setUniform("specular_col", this->m_specular_color.Param<core::param::ColorParam>()->Value()[0],
        this->m_specular_color.Param<core::param::ColorParam>()->Value()[1],
        this->m_specular_color.Param<core::param::ColorParam>()->Value()[2]);
    compute_shdr->setUniform("light_col", curlight.colour[0], curlight.colour[1], curlight.colour[2]);
    compute_shdr->setUniform("material_col", this->m_material_color.Param<core::param::ColorParam>()->Value()[0],
        this->m_material_color.Param<core::param::ColorParam>()->Value()[1],
        this->m_material_color.Param<core::param::ColorParam>()->Value()[2]);

    /*    auto const arv = std::dynamic_pointer_cast<core::view::AbstractView const>(cr.PeekCallerSlot()->Parent());
    std::array<float, 4> bgCol = {1.0f, 1.0f, 1.0f, 1.0f};
    if (arv != nullptr) {
        auto const ptr = arv->BackgroundColor();
        bgCol[0] = ptr[0];
        bgCol[1] = ptr[1];
        bgCol[2] = ptr[2];
        bgCol[3] = 1.0f;
    }*/
    compute_shdr->setUniform("background", cr.BackgroundColor());

    if (this->m_mode.Param<core::param::EnumParam>()->Value() == 0) {
        compute_shdr->setUniform(
            "opacityThreshold", this->m_opacity_threshold.Param<core::param::FloatParam>()->Value());
    } else if (this->m_mode.Param<core::param::EnumParam>()->Value() == 1) {
        compute_shdr->setUniform("isoValue", this->m_iso_value.Param<core::param::FloatParam>()->Value());

        compute_shdr->setUniform("opacity", this->m_opacity.Param<core::param::FloatParam>()->Value());
    }

    this->m_opacity_threshold.Parameter()->SetGUIVisible(this->m_mode.Param<core::param::EnumParam>()->Value() == 0);
    this->m_iso_value.Parameter()->SetGUIVisible(this->m_mode.Param<core::param::EnumParam>()->Value() == 1);
    this->m_opacity.Parameter()->SetGUIVisible(this->m_mode.Param<core::param::EnumParam>()->Value() == 1);

    // bind volume texture
    glActiveTexture(GL_TEXTURE0);
    m_volume_texture->bindTexture();
    compute_shdr->setUniform("volume_tx3D", 0);

    // bind the transfer function
    if (this->m_mode.Param<core::param::EnumParam>()->Value() == 0) {
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_1D, tf_texture);
        compute_shdr->setUniform("tf_tx1D", 1);

        if (ci != nullptr) {
            glActiveTexture(GL_TEXTURE2);
            this->fbo.BindColourTexture();
            compute_shdr->setUniform("color_tx2D", 2);

            glActiveTexture(GL_TEXTURE3);
            this->fbo.BindDepthTexture();
            compute_shdr->setUniform("depth_tx2D", 3);

            compute_shdr->setUniform("use_depth_tx", 1);
        } else {
            compute_shdr->setUniform("use_depth_tx", 0);
        }
    }

    if (this->m_mode.Param<core::param::EnumParam>()->Value() == 2) {
        if (ci != nullptr) {
            glActiveTexture(GL_TEXTURE2);
            this->fbo.BindColourTexture();
            compute_shdr->setUniform("color_tx2D", 2);

            glActiveTexture(GL_TEXTURE3);
            this->fbo.BindDepthTexture();
            compute_shdr->setUniform("depth_tx2D", 3);

            compute_shdr->setUniform("use_depth_tx", 1);
        } else {
            compute_shdr->setUniform("use_depth_tx", 0);
        }
    }

    // bind image texture
    m_render_target->bindImage(0, GL_WRITE_ONLY);

    if (this->m_mode.Param<core::param::EnumParam>()->Value() == 1) {
        m_normal_target->bindImage(1, GL_WRITE_ONLY);
        m_depth_target->bindImage(2, GL_WRITE_ONLY);
    }

    // dispatch compute
    glDispatchCompute(
        static_cast<int>(std::ceil(rt_resolution[0] / 8.0f)), static_cast<int>(std::ceil(rt_resolution[1] / 8.0f)), 1);

    if (this->m_mode.Param<core::param::EnumParam>()->Value() == 1) {
        glBindImageTexture(2, 0, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R8);
        glBindImageTexture(1, 0, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R8);
    }

    glBindImageTexture(0, 0, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R8);

    if (this->m_mode.Param<core::param::EnumParam>()->Value() == 0 ||
        this->m_mode.Param<core::param::EnumParam>()->Value() == 2) {
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, 0);
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    if (this->m_mode.Param<core::param::EnumParam>()->Value() == 0) {
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_1D, 0);
    }
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, 0);

    // compute_shdr->disable();

    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

    // read image back to determine min max
    float rndr_min = std::numeric_limits<float>::max();
    float rndr_max = std::numeric_limits<float>::lowest();
    if (this->m_mode.Param<core::param::EnumParam>()->Value() == 2) {
        glActiveTexture(GL_TEXTURE0);
        m_render_target->bindTexture();
        int width = 0;
        int height = 0;
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &width);
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &height);
        std::vector<float> tmp_data(width * height * 4);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, tmp_data.data());

        for (size_t idx = 0; idx < tmp_data.size() / 4; ++idx) {
            auto const val = tmp_data[idx * 4 + 3];
            if (val < rndr_min)
                rndr_min = val;
            if (val > rndr_max)
                rndr_max = val;
        }

        glBindTexture(GL_TEXTURE_2D, 0);
    }

    // copy image to framebuffer
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    bool state_depth_test = glIsEnabled(GL_DEPTH_TEST);
    bool state_blend = glIsEnabled(GL_BLEND);

    GLint state_blend_src_rgb, state_blend_src_alpha, state_blend_dst_rgb, state_blend_dst_alpha;
    glGetIntegerv(GL_BLEND_SRC_RGB, &state_blend_src_rgb);
    glGetIntegerv(GL_BLEND_SRC_ALPHA, &state_blend_src_alpha);
    glGetIntegerv(GL_BLEND_DST_RGB, &state_blend_dst_rgb);
    glGetIntegerv(GL_BLEND_DST_ALPHA, &state_blend_dst_alpha);

    if (this->m_mode.Param<core::param::EnumParam>()->Value() == 0 ||
        this->m_mode.Param<core::param::EnumParam>()->Value() == 2) {
        if (state_depth_test)
            glDisable(GL_DEPTH_TEST);
    } else if (this->m_mode.Param<core::param::EnumParam>()->Value() == 1) {
        if (!state_depth_test)
            glEnable(GL_DEPTH_TEST);
    }

    if (!state_blend)
        glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    auto fbo_shdr = rtf_shdr.get();
    if (this->m_mode.Param<core::param::EnumParam>()->Value() == 2) {
        fbo_shdr = rtf_aggr_shdr.get();
    }

    fbo_shdr->use();

    glActiveTexture(GL_TEXTURE0);
    m_render_target->bindTexture();
    fbo_shdr->setUniform("src_tx2D", 0);

    if (this->m_mode.Param<core::param::EnumParam>()->Value() == 1) {
        glActiveTexture(GL_TEXTURE1);
        m_normal_target->bindTexture();
        fbo_shdr->setUniform("normal_tx2D", 1);

        glActiveTexture(GL_TEXTURE2);
        m_depth_target->bindTexture();
        fbo_shdr->setUniform("depth_tx2D", 2);

        GLenum buffers[] = {GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT};
        glDrawBuffers(2, buffers);
    }

    if (this->m_mode.Param<core::param::EnumParam>()->Value() == 2) {
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_1D, tf_texture);
        fbo_shdr->setUniform("tf_tx1D", 1);

        fbo_shdr->setUniform("valRange", rndr_min, rndr_max);
    }

    glDrawArrays(GL_TRIANGLES, 0, 6);

    if (this->m_mode.Param<core::param::EnumParam>()->Value() == 1) {
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, 0);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    if (this->m_mode.Param<core::param::EnumParam>()->Value() == 2) {
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_1D, 0);
    }
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);

    // fbo_shdr->disable();

    glBlendFuncSeparate(state_blend_src_rgb, state_blend_dst_rgb, state_blend_src_alpha, state_blend_dst_alpha);
    if (!state_blend)
        glDisable(GL_BLEND);
    if (state_depth_test)
        glEnable(GL_DEPTH_TEST);
    else
        glDisable(GL_DEPTH_TEST);

    glUseProgram(0);

    return true;
}

bool RaycastVolumeRenderer::updateVolumeData(const unsigned int frameID) {
    auto* cd = this->m_volumetricData_callerSlot.CallAs<geocalls::VolumetricDataCall>();

    if (cd == nullptr)
        return false;

    // Use the force
    cd->SetFrameID(frameID, true);
    do {
        if (!(*cd)(geocalls::VolumetricDataCall::IDX_GET_EXTENTS))
            return false;
        if (!(*cd)(geocalls::VolumetricDataCall::IDX_GET_METADATA))
            return false;
        if (!(*cd)(geocalls::VolumetricDataCall::IDX_GET_DATA))
            return false;
    } while (cd->FrameID() != frameID);

    // TODO check time and frame id or whatever else
    if (this->m_volume_datahash != cd->DataHash()) {
        valRangeNeedsUpdate = true;
    }
    if (this->m_volume_datahash != cd->DataHash() || this->m_frame_id != cd->FrameID()) {
        this->m_volume_datahash = cd->DataHash();
        this->m_frame_id = cd->FrameID();
    } else {
        return true;
    }

    auto const metadata = cd->GetMetadata();

    if (!metadata->GridType == geocalls::CARTESIAN) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "RaycastVolumeRenderer only works with cartesian grids (for now)");
        return false;
    }

    m_volume_origin[0] = metadata->Origin[0];
    m_volume_origin[1] = metadata->Origin[1];
    m_volume_origin[2] = metadata->Origin[2];
    m_volume_extents[0] = metadata->Extents[0];
    m_volume_extents[1] = metadata->Extents[1];
    m_volume_extents[2] = metadata->Extents[2];
    m_volume_resolution[0] = metadata->Resolution[0];
    m_volume_resolution[1] = metadata->Resolution[1];
    m_volume_resolution[2] = metadata->Resolution[2];

    valRange[0] = 0.0f;
    valRange[1] = 1.0f;

    GLenum internal_format;
    GLenum format;
    GLenum type;

    switch (metadata->ScalarType) {
    case geocalls::FLOATING_POINT:
        if (metadata->ScalarLength == 4) {
            internal_format = GL_R32F;
            format = GL_RED;
            type = GL_FLOAT;
            // this only makes sense here, all other data types are normalized anyway
            valRange[0] = metadata->MinValues[0];
            valRange[1] = metadata->MaxValues[0];
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Floating point values with a length != 4 byte are invalid.");
            return false;
        }
        break;
    case geocalls::UNSIGNED_INTEGER:
        if (metadata->ScalarLength == 1) {
            internal_format = GL_R8;
            format = GL_RED;
            type = GL_UNSIGNED_BYTE;
        } else if (metadata->ScalarLength == 2) {
            internal_format = GL_R16;
            format = GL_RED;
            type = GL_UNSIGNED_SHORT;
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "Unsigned integers with a length greater than 2 are invalid.");
            return false;
        }
        break;
    case geocalls::SIGNED_INTEGER:
        if (metadata->ScalarLength == 2) {
            internal_format = GL_R16;
            format = GL_RED;
            type = GL_SHORT;
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteError("Integers with a length != 2 are invalid.");
            return false;
        }
        break;
    case geocalls::BITS:
        megamol::core::utility::log::Log::DefaultLog.WriteError("Invalid datatype.");
        return false;
        break;
    }

    auto const volumedata = cd->GetData();

    // TODO if/else data already on GPU

    glowl::TextureLayout volume_layout(internal_format, metadata->Resolution[0], metadata->Resolution[1],
        metadata->Resolution[2], format, type, 1,
        {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
            {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_MIN_FILTER, GL_LINEAR},
            {GL_TEXTURE_MAG_FILTER, GL_LINEAR}},
        {});

    GLint unpackAlignmentOrig = 0;
    glGetIntegerv(GL_UNPACK_ALIGNMENT, &unpackAlignmentOrig);

    // Pixel data rows must be aligned to 4 bytes by default, this is may not guarantied by all datasets.
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // for upload to GPU
    //glPixelStorei(GL_PACK_ALIGNMENT, 1); // for download from GPU

    m_volume_texture = std::make_unique<glowl::Texture3D>("raycast_volume_texture", volume_layout, volumedata);

    glPixelStorei(GL_UNPACK_ALIGNMENT, unpackAlignmentOrig);

    return true;
}

bool RaycastVolumeRenderer::updateTransferFunction() {
    mmstd_gl::CallGetTransferFunctionGL* ct =
        this->m_transferFunction_callerSlot.CallAs<mmstd_gl::CallGetTransferFunctionGL>();
    if (valRangeNeedsUpdate) {
        ct->SetRange(valRange);
        valRangeNeedsUpdate = false;
    }
    if (ct != NULL && ((*ct)())) {
        tf_texture = ct->OpenGLTexture();
        valRange = ct->Range();
    }

    return true;
}
