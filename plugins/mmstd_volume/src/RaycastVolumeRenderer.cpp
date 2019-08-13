/*
 * RaycastVolumeRenderer.h
 *
 * Copyright (C) 2018 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "RaycastVolumeRenderer.h"

#include <array>

#include "vislib/graphics/gl/ShaderSource.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/misc/VolumetricDataCall.h"
#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/utility/ScaledBoundingBoxes.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "mmcore/view/CallRender3D.h"

#include "linmath.h"
#include "mmcore/view/AbstractRenderingView.h"

using namespace megamol::stdplugin::volume;

RaycastVolumeRenderer::RaycastVolumeRenderer()
    : Renderer3DModule()
    , m_mode("mode", "Mode changing the behavior for the raycaster")
    , m_ray_step_ratio_param("ray step ratio", "Adjust sampling rate")
    , m_use_lighting_slot("use_lighting", "Enable simple volumetric illumination")
    , m_ka_slot("ka", "Ambient part for Phong lighting")
    , m_kd_slot("kd", "Diffuse part for Phong lighting")
    , m_ks_slot("ks", "Specular part for Phong lighting")
    , m_shininess_slot("shininess", "Shininess for Phong lighting")
    , m_ambient_color("ambient color", "Ambient color")
    , m_specular_color("specular color", "Specular color")
    , m_light_color("light color", "Light color")
    , m_material_color("material color", "Material color")
    , m_opacity_threshold("opacity threshold", "Opacity threshold for integrative rendering")
    , m_iso_value("isovalue", "Isovalue for isosurface rendering")
    , paramOverride("override::enable", "Enable override of range")
    , paramMinOverride("override::min", "Override the minimum value provided by the data set")
    , paramMaxOverride("override::max", "Override the maximum value provided by the data set")
    , m_renderer_callerSlot("Renderer", "Renderer for chaining")
    , m_volumetricData_callerSlot("getData", "Connects the volume renderer with a voluemtric data source")
    , m_transferFunction_callerSlot("getTranfserFunction", "Connects the volume renderer with a transfer function") {

    this->m_renderer_callerSlot.SetCompatibleCall<megamol::core::view::CallRender3DDescription>();
    this->MakeSlotAvailable(&this->m_renderer_callerSlot);

    this->m_volumetricData_callerSlot.SetCompatibleCall<megamol::core::misc::VolumetricDataCallDescription>();
    this->MakeSlotAvailable(&this->m_volumetricData_callerSlot);

    this->m_transferFunction_callerSlot.SetCompatibleCall<megamol::core::view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->m_transferFunction_callerSlot);

    this->m_mode << new megamol::core::param::EnumParam(0);
    this->m_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(0, "Integration");
    this->m_mode.Param<megamol::core::param::EnumParam>()->SetTypePair(1, "Isosurface");
    this->m_mode.Param<core::param::EnumParam>()->SetTypePair(2, "Aggregate");
    this->MakeSlotAvailable(&this->m_mode);

    this->m_ray_step_ratio_param << new megamol::core::param::FloatParam(1.0);
    this->MakeSlotAvailable(&this->m_ray_step_ratio_param);

    this->m_opacity_threshold << new megamol::core::param::FloatParam(1.0);
    this->MakeSlotAvailable(&this->m_opacity_threshold);

    this->m_iso_value << new megamol::core::param::FloatParam(0.5);
    this->MakeSlotAvailable(&this->m_iso_value);

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

    this->m_light_color << new core::param::ColorParam(1.0f, 1.0f, 1.0f, 1.0f);
    this->MakeSlotAvailable(&this->m_light_color);

    this->m_material_color << new core::param::ColorParam(0.95f, 0.67f, 0.47f, 1.0f);
    this->MakeSlotAvailable(&this->m_material_color);

    this->paramOverride << new megamol::core::param::BoolParam(false);
    this->MakeSlotAvailable(&this->paramOverride);

    this->paramMinOverride << new megamol::core::param::FloatParam(0.0f);
    this->MakeSlotAvailable(&this->paramMinOverride);

    this->paramMaxOverride << new megamol::core::param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->paramMaxOverride);
}

RaycastVolumeRenderer::~RaycastVolumeRenderer() { this->Release(); }

bool RaycastVolumeRenderer::create() {
    try {
        // create shader program
        m_raycast_volume_compute_shdr = std::make_unique<vislib::graphics::gl::GLSLComputeShader>();
        m_raycast_volume_compute_iso_shdr = std::make_unique<vislib::graphics::gl::GLSLComputeShader>();
        m_raycast_volume_compute_aggr_shdr = std::make_unique<vislib::graphics::gl::GLSLComputeShader>();
        m_render_to_framebuffer_shdr = std::make_unique<vislib::graphics::gl::GLSLShader>();
        m_render_to_framebuffer_aggr_shdr = std::make_unique<vislib::graphics::gl::GLSLShader>();

        vislib::graphics::gl::ShaderSource compute_shader_src;
        vislib::graphics::gl::ShaderSource compute_iso_shader_src;
        vislib::graphics::gl::ShaderSource compute_aggr_shader_src;
        vislib::graphics::gl::ShaderSource vertex_shader_src;
        vislib::graphics::gl::ShaderSource fragment_shader_src;
        vislib::graphics::gl::ShaderSource fragment_shader_aggr_src;

        if (!instance()->ShaderSourceFactory().MakeShaderSource("RaycastVolumeRenderer::compute", compute_shader_src))
            return false;
        if (!m_raycast_volume_compute_shdr->Compile(compute_shader_src.Code(), compute_shader_src.Count()))
            return false;
        if (!m_raycast_volume_compute_shdr->Link()) return false;

        if (!instance()->ShaderSourceFactory().MakeShaderSource(
                "RaycastVolumeRenderer::compute_iso", compute_iso_shader_src))
            return false;
        if (!m_raycast_volume_compute_iso_shdr->Compile(compute_iso_shader_src.Code(), compute_iso_shader_src.Count()))
            return false;
        if (!m_raycast_volume_compute_iso_shdr->Link()) return false;

        if (!instance()->ShaderSourceFactory().MakeShaderSource(
                "RaycastVolumeRenderer::compute_aggr", compute_aggr_shader_src))
            return false;
        if (!m_raycast_volume_compute_aggr_shdr->Compile(compute_aggr_shader_src.Code(), compute_aggr_shader_src.Count()))
            return false;
        if (!m_raycast_volume_compute_aggr_shdr->Link()) return false;

        if (!instance()->ShaderSourceFactory().MakeShaderSource("RaycastVolumeRenderer::vert", vertex_shader_src))
            return false;
        if (!instance()->ShaderSourceFactory().MakeShaderSource("RaycastVolumeRenderer::frag", fragment_shader_src))
            return false;
        if (!instance()->ShaderSourceFactory().MakeShaderSource("RaycastVolumeRenderer::frag_aggr", fragment_shader_aggr_src))
            return false;
        if (!m_render_to_framebuffer_shdr->Compile(vertex_shader_src.Code(), vertex_shader_src.Count(),
                fragment_shader_src.Code(), fragment_shader_src.Count()))
            return false;
        if (!m_render_to_framebuffer_shdr->Link()) return false;

        if (!m_render_to_framebuffer_aggr_shdr->Compile(vertex_shader_src.Code(), vertex_shader_src.Count(),
                fragment_shader_aggr_src.Code(), fragment_shader_aggr_src.Count()))
            return false;
        if (!m_render_to_framebuffer_aggr_shdr->Link()) return false;
    } catch (vislib::graphics::gl::AbstractOpenGLShader::CompileException ce) {
        vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "Unable to compile shader (@%s): %s\n",
            vislib::graphics::gl::AbstractOpenGLShader::CompileException::CompileActionName(ce.FailedAction()),
            ce.GetMsgA());
        return false;
    } catch (vislib::Exception e) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "Unable to compile shader: %s\n", e.GetMsgA());
        return false;
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteMsg(
            vislib::sys::Log::LEVEL_ERROR, "Unable to compile shader: Unknown exception\n");
        return false;
    }

    // create render target texture
    TextureLayout render_tgt_layout(GL_RGBA8, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, 1,
        {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
            {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_MIN_FILTER, GL_LINEAR},
            {GL_TEXTURE_MAG_FILTER, GL_LINEAR}},
        {});
    m_render_target = std::make_unique<Texture2D>("raycast_volume_render_target", render_tgt_layout, nullptr);

    // create normal target texture
    TextureLayout normal_tgt_layout(GL_RGBA8, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, 1,
        {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
            {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_MIN_FILTER, GL_LINEAR},
            {GL_TEXTURE_MAG_FILTER, GL_LINEAR}},
        {});
    m_normal_target = std::make_unique<Texture2D>("raycast_volume_normal_target", normal_tgt_layout, nullptr);

    // create depth target texture
    TextureLayout depth_tgt_layout(GL_R8, 1, 1, 1, GL_R, GL_UNSIGNED_BYTE, 1,
        {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
            {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_MIN_FILTER, GL_LINEAR},
            {GL_TEXTURE_MAG_FILTER, GL_LINEAR}},
        {});
    m_depth_target = std::make_unique<Texture2D>("raycast_volume_depth_target", depth_tgt_layout, nullptr);

    // create empty volume texture
    TextureLayout volume_layout(GL_R32F, 1, 1, 1, GL_RED, GL_FLOAT, 1,
        {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
            {GL_TEXTURE_MIN_FILTER, GL_LINEAR}, {GL_TEXTURE_MAG_FILTER, GL_LINEAR}},
        {});
    m_volume_texture = std::make_unique<Texture3D>("raycast_volume_texture", volume_layout, nullptr);

    // create empty transfer function texture
    TextureLayout tf(GL_RGBA8, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, 1,
        {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
            {GL_TEXTURE_MIN_FILTER, GL_LINEAR}, {GL_TEXTURE_MAG_FILTER, GL_LINEAR}},
        {});
    m_transfer_function = std::make_unique<Texture2D>("raycast_volume_texture", tf, nullptr);

    return true;
}

void RaycastVolumeRenderer::release() {
    m_raycast_volume_compute_shdr.reset(nullptr);
    m_raycast_volume_compute_iso_shdr.reset(nullptr);
    m_raycast_volume_compute_aggr_shdr.reset(nullptr);
    m_render_target.reset(nullptr);
    m_normal_target.reset(nullptr);
    m_depth_target.reset(nullptr);
    if (this->fbo.IsValid()) this->fbo.Release();
}

bool RaycastVolumeRenderer::GetExtents(megamol::core::Call& call) {
    auto cr = dynamic_cast<core::view::CallRender3D*>(&call);
    auto cd = m_volumetricData_callerSlot.CallAs<megamol::core::misc::VolumetricDataCall>();
    auto ci = m_renderer_callerSlot.CallAs<megamol::core::view::CallRender3D>();

    if (cr == nullptr) return false;
    if (cd == nullptr) return false;

    // TODO Do something about time/framecount ?

    int const req_frame = static_cast<int>(cr->Time());

    cd->SetFrameID(req_frame);

    if (!(*cd)(core::misc::VolumetricDataCall::IDX_GET_EXTENTS)) return false;
    if (!(*cd)(core::misc::VolumetricDataCall::IDX_GET_METADATA)) return false;

    cr->SetTimeFramesCount(cd->FrameCount());

    std::vector<core::BoundingBoxes> bbs{cd->GetBoundingBoxes()};

    if (ci != nullptr) {
        *ci = *cr;

        if (!(*ci)(core::view::CallRender3D::FnGetExtents)) return false;

        bbs.push_back(ci->GetBoundingBoxes());
    }

    cr->AccessBoundingBoxes() = core::utility::combineAndMagicScaleBoundingBoxes(bbs);

    return true;
}

bool RaycastVolumeRenderer::Render(megamol::core::Call& call) {
    megamol::core::view::CallRender3D* cr = dynamic_cast<core::view::CallRender3D*>(&call);
    if (cr == NULL) return false;

    // Chain renderer
    auto ci = m_renderer_callerSlot.CallAs<megamol::core::view::CallRender3D>();

    if (ci != nullptr) {
        ci->SetCameraParameters(cr->GetCameraParameters());

        if (this->m_mode.Param<core::param::EnumParam>()->Value() == 0 ||
            this->m_mode.Param<core::param::EnumParam>()->Value() == 2) {
            if (this->fbo.IsValid()) this->fbo.Release();
            this->fbo.Create(ci->GetViewport().Width(), ci->GetViewport().Height(), GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE,
                vislib::graphics::gl::FramebufferObject::ATTACHMENT_TEXTURE);
            this->fbo.Enable();

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        }

        ci->SetTime(cr->Time());
        if (!(*ci)(core::view::CallRender3D::FnRender)) return false;

        if (this->m_mode.Param<core::param::EnumParam>()->Value() == 0 ||
            this->m_mode.Param<core::param::EnumParam>()->Value() == 2) {
            this->fbo.Disable();
        }
    }

    // create render target texture
    TextureLayout render_tgt_layout(GL_RGBA8, cr->GetViewport().Width(), cr->GetViewport().Height(), 1, GL_RGBA,
        GL_UNSIGNED_BYTE, 1,
        {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
            {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_MIN_FILTER, GL_LINEAR},
            {GL_TEXTURE_MAG_FILTER, GL_LINEAR}},
        {});
    m_render_target->reload(render_tgt_layout, nullptr);

    // create normal target texture
    TextureLayout normal_tgt_layout(GL_RGBA8, cr->GetViewport().Width(), cr->GetViewport().Height(), 1, GL_RGBA,
        GL_UNSIGNED_BYTE, 1,
        {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
            {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_MIN_FILTER, GL_LINEAR},
            {GL_TEXTURE_MAG_FILTER, GL_LINEAR}},
        {});
    m_normal_target->reload(normal_tgt_layout, nullptr);

    // create depth target texture
    TextureLayout depth_tgt_layout(GL_R8, cr->GetViewport().Width(), cr->GetViewport().Height(), 1, GL_R,
        GL_UNSIGNED_BYTE, 1,
        {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
            {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_MIN_FILTER, GL_LINEAR},
            {GL_TEXTURE_MAG_FILTER, GL_LINEAR}},
        {});
    m_depth_target->reload(depth_tgt_layout, nullptr);

    // this is the apex of suck and must die
    core::utility::glMagicScale scaling;
    scaling.apply(cr->GetBoundingBoxes());

    GLfloat modelViewMatrix_column[16];
    glGetFloatv(GL_MODELVIEW_MATRIX, modelViewMatrix_column);
    GLfloat projMatrix_column[16];
    glGetFloatv(GL_PROJECTION_MATRIX, projMatrix_column);

    std::array<float, 4> light = {0.0f, 0.0f, 1.0f, 1.0f};
    glGetLightfv(GL_LIGHT0, GL_POSITION, light.data());
    // end suck

    if (!updateVolumeData(cr->Time())) return false;

    // enable raycast volume rendering program
    vislib::graphics::gl::GLSLComputeShader* compute_shdr;

    // pick shader based on selected mode
    if (this->m_mode.Param<core::param::EnumParam>()->Value() == 0) {
        if (!updateTransferFunction()) return false;

        compute_shdr = this->m_raycast_volume_compute_shdr.get();
    } else if (this->m_mode.Param<core::param::EnumParam>()->Value() == 1) {
        compute_shdr = this->m_raycast_volume_compute_iso_shdr.get();
    } else if (this->m_mode.Param<core::param::EnumParam>()->Value() == 2) {
        if (!updateTransferFunction()) return false;
        compute_shdr = this->m_raycast_volume_compute_aggr_shdr.get();
    } else {
        vislib::sys::Log::DefaultLog.WriteError("Unknown raycast mode.");
        return false;
    }

    // setup
    compute_shdr->Enable();

    glUniformMatrix4fv(compute_shdr->ParameterLocation("view_mx"), 1, GL_FALSE, modelViewMatrix_column);
    glUniformMatrix4fv(compute_shdr->ParameterLocation("proj_mx"), 1, GL_FALSE, projMatrix_column);

    vec2 rt_resolution;
    rt_resolution[0] = static_cast<float>(m_render_target->getWidth());
    rt_resolution[1] = static_cast<float>(m_render_target->getHeight());
    glUniform2fv(compute_shdr->ParameterLocation("rt_resolution"), 1, rt_resolution);

    vec3 box_min;
    box_min[0] = m_volume_origin[0];
    box_min[1] = m_volume_origin[1];
    box_min[2] = m_volume_origin[2];
    vec3 box_max;
    box_max[0] = m_volume_origin[0] + m_volume_extents[0];
    box_max[1] = m_volume_origin[1] + m_volume_extents[1];
    box_max[2] = m_volume_origin[2] + m_volume_extents[2];
    glUniform3fv(compute_shdr->ParameterLocation("boxMin"), 1, box_min);
    glUniform3fv(compute_shdr->ParameterLocation("boxMax"), 1, box_max);

    glUniform3f(compute_shdr->ParameterLocation("halfVoxelSize"), 1.0f / (2.0f * (m_volume_resolution[0] - 1)),
        1.0f / (2.0f * (m_volume_resolution[1] - 1)), 1.0f / (2.0f * (m_volume_resolution[2] - 1)));
    auto const maxResolution =
        std::fmax(m_volume_resolution[0], std::fmax(m_volume_resolution[1], m_volume_resolution[2]));
    auto const maxExtents = std::fmax(m_volume_extents[0], std::fmax(m_volume_extents[1], m_volume_extents[2]));
    glUniform1f(compute_shdr->ParameterLocation("voxelSize"), maxExtents / (maxResolution - 1.0f));
    glUniform2fv(compute_shdr->ParameterLocation("valRange"), 1, valRange.data());
    glUniform1f(compute_shdr->ParameterLocation("rayStepRatio"),
        this->m_ray_step_ratio_param.Param<core::param::FloatParam>()->Value());

    glUniform1i(compute_shdr->ParameterLocation("use_lighting"),
        this->m_use_lighting_slot.Param<core::param::BoolParam>()->Value());
    glUniform1f(compute_shdr->ParameterLocation("ka"),
        this->m_ka_slot.Param<core::param::FloatParam>()->Value());
    glUniform1f(compute_shdr->ParameterLocation("kd"),
        this->m_kd_slot.Param<core::param::FloatParam>()->Value());
    glUniform1f(compute_shdr->ParameterLocation("ks"),
        this->m_ks_slot.Param<core::param::FloatParam>()->Value());
    glUniform1f(compute_shdr->ParameterLocation("shininess"),
        this->m_shininess_slot.Param<core::param::FloatParam>()->Value());
    glUniform3fv(compute_shdr->ParameterLocation("light"), 1, light.data());
    glUniform3fv(compute_shdr->ParameterLocation("ambient_col"), 1,
        this->m_ambient_color.Param<core::param::ColorParam>()->Value().data());
    glUniform3fv(compute_shdr->ParameterLocation("specular_col"), 1,
        this->m_specular_color.Param<core::param::ColorParam>()->Value().data());
    glUniform3fv(compute_shdr->ParameterLocation("light_col"), 1,
        this->m_light_color.Param<core::param::ColorParam>()->Value().data());
    glUniform3fv(compute_shdr->ParameterLocation("material_col"), 1,
        this->m_material_color.Param<core::param::ColorParam>()->Value().data());

    auto const arv = std::dynamic_pointer_cast<core::view::AbstractRenderingView const>(cr->PeekCallerSlot()->Parent());
    std::array<float, 3> bkgndCol = {1.0f, 1.0f, 1.0f};
    if (arv != nullptr) {
        auto const ptr = arv->BkgndColour();
        bkgndCol[0] = ptr[0];
        bkgndCol[1] = ptr[1];
        bkgndCol[2] = ptr[2];
    }
    glUniform3fv(compute_shdr->ParameterLocation("background"), 1, bkgndCol.data());

    if (this->m_mode.Param<core::param::EnumParam>()->Value() == 0) {
        glUniform1f(compute_shdr->ParameterLocation("opacityThreshold"),
            this->m_opacity_threshold.Param<core::param::FloatParam>()->Value());
    } else if (this->m_mode.Param<core::param::EnumParam>()->Value() == 1) {
        glUniform1f(
            compute_shdr->ParameterLocation("isoValue"), this->m_iso_value.Param<core::param::FloatParam>()->Value());
    }

    // bind volume texture
    glActiveTexture(GL_TEXTURE0);
    m_volume_texture->bindTexture();
    glUniform1i(compute_shdr->ParameterLocation("volume_tx3D"), 0);

    // bind the transfer function
    if (this->m_mode.Param<core::param::EnumParam>()->Value() == 0) {
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_1D, tf_texture);
        glUniform1i(compute_shdr->ParameterLocation("tf_tx1D"), 1);

        if (ci != nullptr) {
            glActiveTexture(GL_TEXTURE2);
            this->fbo.BindColourTexture();
            glUniform1i(compute_shdr->ParameterLocation("color_tx2D"), 2);

            glActiveTexture(GL_TEXTURE3);
            this->fbo.BindDepthTexture();
            glUniform1i(compute_shdr->ParameterLocation("depth_tx2D"), 3);

            glUniform1i(compute_shdr->ParameterLocation("use_depth_tx"), 1);
        } else {
            glUniform1i(compute_shdr->ParameterLocation("use_depth_tx"), 0);
        }
    }

    if (this->m_mode.Param<core::param::EnumParam>()->Value() == 2) {
        if (ci != nullptr) {
            glActiveTexture(GL_TEXTURE2);
            this->fbo.BindColourTexture();
            glUniform1i(compute_shdr->ParameterLocation("color_tx2D"), 2);

            glActiveTexture(GL_TEXTURE3);
            this->fbo.BindDepthTexture();
            glUniform1i(compute_shdr->ParameterLocation("depth_tx2D"), 3);

            glUniform1i(compute_shdr->ParameterLocation("use_depth_tx"), 1);
        } else {
            glUniform1i(compute_shdr->ParameterLocation("use_depth_tx"), 0);
        }
    }

    // bind image texture
    m_render_target->bindImage(0, GL_WRITE_ONLY);

    if (this->m_mode.Param<core::param::EnumParam>()->Value() == 1) {
        m_normal_target->bindImage(1, GL_WRITE_ONLY);
        m_depth_target->bindImage(2, GL_WRITE_ONLY);
    }

    // dispatch compute
    compute_shdr->Dispatch(
        static_cast<int>(std::ceil(rt_resolution[0] / 8.0f)), static_cast<int>(std::ceil(rt_resolution[1] / 8.0f)), 1);

    compute_shdr->Disable();
   
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_1D, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, 0);

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
        std::vector<float> tmp_data(width*height*4);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, tmp_data.data());

        for (size_t idx = 0; idx < tmp_data.size() / 4; ++idx) {
            auto const val = tmp_data[idx * 4 + 3];
            if (val < rndr_min) rndr_min = val;
            if (val > rndr_max) rndr_max = val;
        }

        glBindTexture(GL_TEXTURE_2D, 0);
    }

    // copy image to framebuffer
    // TODO query gl state and reset to previous state?
    if (this->m_mode.Param<core::param::EnumParam>()->Value() == 0 ||
        this->m_mode.Param<core::param::EnumParam>()->Value() == 2) {
        glDisable(GL_DEPTH_TEST);
    } else if (this->m_mode.Param<core::param::EnumParam>()->Value() == 1) {
        glEnable(GL_DEPTH_TEST);
    }

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    auto fbo_shdr = m_render_to_framebuffer_shdr.get();
    if (this->m_mode.Param<core::param::EnumParam>()->Value() == 2) {
        fbo_shdr = m_render_to_framebuffer_aggr_shdr.get();
    }

    fbo_shdr->Enable();

    glActiveTexture(GL_TEXTURE0);
    m_render_target->bindTexture();
    glUniform1i(fbo_shdr->ParameterLocation("src_tx2D"), 0);

    if (this->m_mode.Param<core::param::EnumParam>()->Value() == 1) {
        glActiveTexture(GL_TEXTURE1);
        m_normal_target->bindTexture();
        glUniform1i(fbo_shdr->ParameterLocation("normal_tx2D"), 1);

        glActiveTexture(GL_TEXTURE2);
        m_depth_target->bindTexture();
        glUniform1i(fbo_shdr->ParameterLocation("depth_tx2D"), 2);

        GLenum buffers[] = {GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT};
        glDrawBuffers(2, buffers);
    }

    if (this->m_mode.Param<core::param::EnumParam>()->Value() == 2) {
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_1D, tf_texture);
        glUniform1i(fbo_shdr->ParameterLocation("tf_tx1D"), 1);

        glUniform2f(fbo_shdr->ParameterLocation("valRange"), rndr_min, rndr_max);
    }

    glDrawArrays(GL_TRIANGLES, 0, 6);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);

    fbo_shdr->Disable();

    // cleanup
    glUseProgram(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);

    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    return true;
}

bool RaycastVolumeRenderer::updateVolumeData(const unsigned int frameID) {
    auto* cd = this->m_volumetricData_callerSlot.CallAs<megamol::core::misc::VolumetricDataCall>();

    if (cd == nullptr) return false;

    // Use the force
    cd->SetFrameID(frameID, true);
    do {
        if (!(*cd)(core::misc::VolumetricDataCall::IDX_GET_EXTENTS)) return false;
        if (!(*cd)(core::misc::VolumetricDataCall::IDX_GET_METADATA)) return false;
        if (!(*cd)(core::misc::VolumetricDataCall::IDX_GET_DATA)) return false;
    } while (cd->FrameID() != frameID);

    // TODO check time and frame id or whatever else
    if (this->m_volume_datahash != cd->DataHash() || this->m_frame_id != cd->FrameID()) {
        this->m_volume_datahash = cd->DataHash();
        this->m_frame_id = cd->FrameID();
    } else {
        return true;
    }

    auto const metadata = cd->GetMetadata();

    if (!metadata->GridType == core::misc::CARTESIAN) {
        vislib::sys::Log::DefaultLog.WriteError("RaycastVolumeRenderer only works with cartesian grids (for now)");
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

    if (this->paramOverride.Param<core::param::BoolParam>()->Value()) {
        valRange[0] = this->paramMinOverride.Param<core::param::FloatParam>()->Value();
        valRange[1] = this->paramMaxOverride.Param<core::param::FloatParam>()->Value();
    } else {
    valRange[0] = metadata->MinValues[0];
    valRange[1] = metadata->MaxValues[0];
    }

    GLenum internal_format;
    GLenum format;
    GLenum type;

    switch (metadata->ScalarType) {
    case core::misc::FLOATING_POINT:
        if (metadata->ScalarLength == 4) {
            internal_format = GL_R32F;
            format = GL_RED;
            type = GL_FLOAT;
        } else {
            vislib::sys::Log::DefaultLog.WriteError("Floating point values with a length != 4 byte are invalid.");
            return false;
        }
        break;
    case core::misc::UNSIGNED_INTEGER:
        if (metadata->ScalarLength == 1) {
            internal_format = GL_R8;
            format = GL_RED;
            type = GL_UNSIGNED_BYTE;
        } else if (metadata->ScalarLength == 2) {
            internal_format = GL_R16UI;
            format = GL_RED;
            type = GL_UNSIGNED_SHORT;
        } else {
            vislib::sys::Log::DefaultLog.WriteError("Unsigned integers with a length greater than 2 are invalid.");
            return false;
        }
        break;
    case core::misc::SIGNED_INTEGER:
        if (metadata->ScalarLength == 2) {
            internal_format = GL_R16I;
            format = GL_RED;
            type = GL_SHORT;
        } else {
            vislib::sys::Log::DefaultLog.WriteError("Integers with a length != 2 are invalid.");
            return false;
        }
        break;
    case core::misc::BITS:
        vislib::sys::Log::DefaultLog.WriteError("Invalid datatype.");
        return false;
        break;
    }

    auto const volumedata = cd->GetData();

    // TODO if/else data already on GPU

    // vislib::sys::Log::DefaultLog.WriteInfo(L"Volume frame %u is being uploaded.", cd->FrameID());
    TextureLayout volume_layout(internal_format, metadata->Resolution[0], metadata->Resolution[1],
        metadata->Resolution[2], format, type, 1,
        {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
            {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_MIN_FILTER, GL_LINEAR},
            {GL_TEXTURE_MAG_FILTER, GL_LINEAR}},
        {});

    m_volume_texture->reload(volume_layout, volumedata);

    return true;
}

bool RaycastVolumeRenderer::updateTransferFunction() {
    core::view::CallGetTransferFunction* ct =
        this->m_transferFunction_callerSlot.CallAs<core::view::CallGetTransferFunction>();

    if (ct != NULL && ((*ct)())) {
        tf_texture = ct->OpenGLTexture();
    }

    return true;
}
