/*
 * RaycastVolumeRenderer.cpp
 *
 * Copyright (C) 2018-2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "RaycastVolumeRenderer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>

#include "vislib/Exception.h"
#include "vislib/graphics/gl/GLSLComputeShader.h"
#include "vislib/graphics/gl/GLSLShader.h"
#include "vislib/graphics/gl/ShaderSource.h"
#include "vislib/sys/Log.h"

#include "mmcore/Call.h"
#include "mmcore/CoreInstance.h"
#include "mmcore/misc/VolumetricDataCall.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/view/CallGetTransferFunction.h"
#include "mmcore/view/CallRender3D_2.h"
#include "mmcore/view/Camera_2.h"
#include "mmcore/view/Renderer3DModule_2.h"

#include "glowl/Texture.hpp"
#include "glowl/Texture2D.hpp"
#include "glowl/Texture3D.hpp"

using namespace megamol::stdplugin::volume;

RaycastVolumeRenderer::RaycastVolumeRenderer()
    : Renderer3DModule_2()
    , m_volumetricData_callerSlot("getData", "Connects the volume renderer with a voluemtric data source")
    , m_transferFunction_callerSlot("getTranfserFunction", "Connects the volume renderer with a transfer function")
    , m_ray_step_ratio_param("ray step ratio", "Adjust sampling rate") {

    this->m_volumetricData_callerSlot.SetCompatibleCall<core::misc::VolumetricDataCallDescription>();
    this->MakeSlotAvailable(&this->m_volumetricData_callerSlot);

    this->m_transferFunction_callerSlot.SetCompatibleCall<core::view::CallGetTransferFunctionDescription>();
    this->MakeSlotAvailable(&this->m_transferFunction_callerSlot);

    this->m_ray_step_ratio_param << new core::param::FloatParam(1.0f);
    this->MakeSlotAvailable(&this->m_ray_step_ratio_param);
}

RaycastVolumeRenderer::~RaycastVolumeRenderer() { this->Release(); }

bool RaycastVolumeRenderer::create() {
    try {
        // create shader program
        m_raycast_volume_compute_shdr = std::make_unique<vislib::graphics::gl::GLSLComputeShader>();
        m_render_to_framebuffer_shdr = std::make_unique<vislib::graphics::gl::GLSLShader>();

        vislib::graphics::gl::ShaderSource compute_shader_src;
        vislib::graphics::gl::ShaderSource vertex_shader_src;
        vislib::graphics::gl::ShaderSource fragment_shader_src;

        if (!instance()->ShaderSourceFactory().MakeShaderSource("RaycastVolumeRenderer::compute", compute_shader_src))
            return false;
        if (!m_raycast_volume_compute_shdr->Compile(compute_shader_src.Code(), compute_shader_src.Count()))
            return false;
        if (!m_raycast_volume_compute_shdr->Link()) return false;

        if (!instance()->ShaderSourceFactory().MakeShaderSource("RaycastVolumeRenderer::vert", vertex_shader_src))
            return false;
        if (!instance()->ShaderSourceFactory().MakeShaderSource("RaycastVolumeRenderer::frag", fragment_shader_src))
            return false;
        if (!m_render_to_framebuffer_shdr->Compile(vertex_shader_src.Code(), vertex_shader_src.Count(),
                fragment_shader_src.Code(), fragment_shader_src.Count()))
            return false;
        if (!m_render_to_framebuffer_shdr->Link()) return false;
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
    glowl::TextureLayout render_tgt_layout(GL_RGBA8, 1920, 1080, 1, GL_RGBA, GL_UNSIGNED_BYTE, 1,
        {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
            {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_MIN_FILTER, GL_LINEAR},
            {GL_TEXTURE_MAG_FILTER, GL_LINEAR}},
        {});
    m_render_target = std::make_unique<glowl::Texture2D>("raycast_volume_render_target", render_tgt_layout, nullptr);

    // create empty volume texture
    glowl::TextureLayout volume_layout(GL_R32F, 1, 1, 1, GL_RED, GL_FLOAT, 1,
        {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
            {GL_TEXTURE_MIN_FILTER, GL_LINEAR}, {GL_TEXTURE_MAG_FILTER, GL_LINEAR}},
        {});
    m_volume_texture = std::make_unique<glowl::Texture3D>("raycast_volume_texture", volume_layout, nullptr);

    return true;
}

void RaycastVolumeRenderer::release() {
    m_raycast_volume_compute_shdr.reset(nullptr);
    m_render_target.reset(nullptr);
}

bool RaycastVolumeRenderer::GetExtents(megamol::core::view::CallRender3D_2& cr) {
    auto cd = m_volumetricData_callerSlot.CallAs<core::misc::VolumetricDataCall>();

    if (cd == nullptr) return false;

    // TODO Do something about time/framecount ?

    int const req_frame = static_cast<int>(cr.Time());

    cd->SetFrameID(req_frame);

    if (!(*cd)(core::misc::VolumetricDataCall::IDX_GET_EXTENTS)) return false;
    if (!(*cd)(core::misc::VolumetricDataCall::IDX_GET_METADATA)) return false;

    cr.SetTimeFramesCount(cd->FrameCount());
    cr.AccessBoundingBoxes() = cd->GetBoundingBoxes();

    return true;
}

bool RaycastVolumeRenderer::Render(megamol::core::view::CallRender3D_2& cr) {
    if (!updateVolumeData()) return false;

    auto ct = this->m_transferFunction_callerSlot.CallAs<core::view::CallGetTransferFunction>();
    if (ct == nullptr || !(*ct)()) return false;

    // get camera
    core::view::Camera_2 cam;
    cr.GetCamera(cam);

    cam_type::matrix_type view, proj;
    cam.calc_matrices(view, proj);

    // enable raycast volume rendering program
    m_raycast_volume_compute_shdr->Enable();

    glUniformMatrix4fv(m_raycast_volume_compute_shdr->ParameterLocation("view_mx"), 1, GL_FALSE,
        glm::value_ptr(static_cast<glm::mat4>(view)));
    glUniformMatrix4fv(m_raycast_volume_compute_shdr->ParameterLocation("proj_mx"), 1, GL_FALSE,
        glm::value_ptr(static_cast<glm::mat4>(proj)));

    std::array<float, 2> rt_resolution;
    rt_resolution[0] = static_cast<float>(m_render_target->getWidth());
    rt_resolution[1] = static_cast<float>(m_render_target->getHeight());
    glUniform2fv(m_raycast_volume_compute_shdr->ParameterLocation("rt_resolution"), 1, rt_resolution.data());

    // bbox sizes
    glm::vec3 box_min;
    box_min[0] = m_volume_origin[0];
    box_min[1] = m_volume_origin[1];
    box_min[2] = m_volume_origin[2];
    glm::vec3 box_max;
    box_max[0] = m_volume_origin[0] + m_volume_extents[0];
    box_max[1] = m_volume_origin[1] + m_volume_extents[1];
    box_max[2] = m_volume_origin[2] + m_volume_extents[2];
    glUniform3fv(m_raycast_volume_compute_shdr->ParameterLocation("boxMin"), 1, glm::value_ptr(box_min));
    glUniform3fv(m_raycast_volume_compute_shdr->ParameterLocation("boxMax"), 1, glm::value_ptr(box_max));

    glUniform3f(m_raycast_volume_compute_shdr->ParameterLocation("halfVoxelSize"),
        1.0f / (2.0f * (m_volume_resolution[0] - 1)), 1.0f / (2.0f * (m_volume_resolution[1] - 1)),
        1.0f / (2.0f * (m_volume_resolution[2] - 1)));
    auto const maxResolution =
        std::max(m_volume_resolution[0], std::max(m_volume_resolution[1], m_volume_resolution[2]));
    auto const maxExtents = std::max(m_volume_extents[0], std::max(m_volume_extents[1], m_volume_extents[2]));
    glUniform1f(m_raycast_volume_compute_shdr->ParameterLocation("voxelSize"), maxExtents / (maxResolution - 1.0f));
    glUniform2fv(m_raycast_volume_compute_shdr->ParameterLocation("valRange"), 1, this->valRange.data());
    glUniform1f(m_raycast_volume_compute_shdr->ParameterLocation("rayStepRatio"),
        this->m_ray_step_ratio_param.Param<core::param::FloatParam>()->Value());
    glUniform1f(m_raycast_volume_compute_shdr->ParameterLocation("opacityThreshold"), 1.0);

    // bind volume texture
    glActiveTexture(GL_TEXTURE0);
    m_volume_texture->bindTexture();
    glUniform1i(m_raycast_volume_compute_shdr->ParameterLocation("volume_tx3D"), 0);
    // bind the transfer function
    ct->BindConvenience(*m_raycast_volume_compute_shdr, GL_TEXTURE1, 1);

    // bind image texture
    m_render_target->bindImage(0, GL_WRITE_ONLY);

    // dispatch compute
    m_raycast_volume_compute_shdr->Dispatch(
        static_cast<int>(std::ceil(rt_resolution[0] / 8.0f)), static_cast<int>(std::ceil(rt_resolution[1] / 8.0f)), 1);

    // unbind image texture
    glBindImageTexture(0, 0, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R);

    // unbind the transfer function
    ct->UnbindConvenience();
    // unbind volume texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, 0);

    m_raycast_volume_compute_shdr->Disable();

    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

    // store state
    bool state_depth_test = glIsEnabled(GL_DEPTH_TEST);
    bool state_blend = glIsEnabled(GL_BLEND);

    GLint state_blend_src_rgb, state_blend_src_alpha, state_blend_dst_rgb, state_blend_dst_alpha;
    glGetIntegerv(GL_BLEND_SRC_RGB, &state_blend_src_rgb);
    glGetIntegerv(GL_BLEND_SRC_ALPHA, &state_blend_src_alpha);
    glGetIntegerv(GL_BLEND_DST_RGB, &state_blend_dst_rgb);
    glGetIntegerv(GL_BLEND_DST_ALPHA, &state_blend_dst_alpha);

    if (state_depth_test) glDisable(GL_DEPTH_TEST);
    if (!state_blend) glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    m_render_to_framebuffer_shdr->Enable();

    // bind texture
    glActiveTexture(GL_TEXTURE0);
    m_render_target->bindTexture();
    glUniform1i(m_render_to_framebuffer_shdr->ParameterLocation("src_tx2D"), 0);

    // render
    glDrawArrays(GL_TRIANGLES, 0, 6);

    // unbind texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);

    m_render_to_framebuffer_shdr->Disable();

    // restore state
    glBlendFuncSeparate(state_blend_src_rgb, state_blend_dst_rgb, state_blend_src_alpha, state_blend_dst_alpha);
    if (!state_blend) glDisable(GL_BLEND);
    if (state_depth_test)
        glEnable(GL_DEPTH_TEST);
    else
        glDisable(GL_DEPTH_TEST);

    return true;
}

bool RaycastVolumeRenderer::updateVolumeData() {
    auto* cd = this->m_volumetricData_callerSlot.CallAs<core::misc::VolumetricDataCall>();

    if (cd == nullptr) return false;

    if (!(*cd)(core::misc::VolumetricDataCall::IDX_GET_EXTENTS)) return false;
    if (!(*cd)(core::misc::VolumetricDataCall::IDX_GET_METADATA)) return false;
    if (!(*cd)(core::misc::VolumetricDataCall::IDX_GET_DATA)) return false;

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

    valRange[0] = metadata->MinValues[0];
    valRange[1] = metadata->MaxValues[0];

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

    // debug using dummy-data
    std::array<uint8_t, 8> debug_volume_data = {255, 0, 255, 0, 255, 255, 0, 0};
    glowl::TextureLayout debug_volume_layout(GL_R8, 2, 2, 2, GL_RED, GL_UNSIGNED_BYTE, 1,
        {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
            {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_MIN_FILTER, GL_NEAREST},
            {GL_TEXTURE_MAG_FILTER, GL_NEAREST}},
        {});

    glowl::TextureLayout volume_layout(internal_format, metadata->Resolution[0], metadata->Resolution[1],
        metadata->Resolution[2], format, type, 1,
        {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
            {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_MIN_FILTER, GL_LINEAR},
            {GL_TEXTURE_MAG_FILTER, GL_LINEAR}},
        {});

    m_volume_texture->reload(volume_layout, volumedata);

    return true;
}
