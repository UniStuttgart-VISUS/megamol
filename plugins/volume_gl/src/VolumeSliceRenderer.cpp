/*
 * VolumeSliceRenderer.cpp
 *
 * Copyright (C) 2012-2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "VolumeSliceRenderer.h"

#include <array>
#include <cmath>

#include <glm/ext.hpp>

#include "geometry_calls/VolumetricDataCall.h"
#include "mmcore/Call.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore_gl/utility/ShaderFactory.h"
#include "mmstd/renderer/CallClipPlane.h"
#include "mmstd_gl/renderer/CallGetTransferFunctionGL.h"
#include "mmstd_gl/renderer/CallRender3DGL.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"
#include "vislib/math/Plane.h"
#include "vislib/math/Point.h"

using megamol::core::utility::log::Log;

/*
 * VolumeSliceRenderer::VolumeSliceRenderer
 */
megamol::volume_gl::VolumeSliceRenderer::VolumeSliceRenderer()
        : mmstd_gl::Renderer3DModuleGL()
        , getVolSlot("getVol", "The call for data")
        , getTFSlot("gettransferfunction", "The call for Transfer function")
        , getClipPlaneSlot("getclipplane", "The call for clipping plane") {

    this->getVolSlot.SetCompatibleCall<geocalls::VolumetricDataCallDescription>();
    this->MakeSlotAvailable(&this->getVolSlot);

    this->getTFSlot.SetCompatibleCall<mmstd_gl::CallGetTransferFunctionGLDescription>();
    this->MakeSlotAvailable(&this->getTFSlot);

    this->getClipPlaneSlot.SetCompatibleCall<core::view::CallClipPlaneDescription>();
    this->MakeSlotAvailable(&this->getClipPlaneSlot);
}


/*
 * VolumeSliceRenderer::VolumeSliceRenderer
 */
megamol::volume_gl::VolumeSliceRenderer::~VolumeSliceRenderer() {
    this->Release();
}


/*
 * VolumeSliceRenderer::VolumeSliceRenderer
 */
bool megamol::volume_gl::VolumeSliceRenderer::create() {
    auto const shaderOptions =
        core::utility::make_path_shader_options(frontend_resources.get<megamol::frontend_resources::RuntimeConfig>());

    try {
        this->compute_shader = core::utility::make_glowl_shader(
            "compute_shader", shaderOptions, "volume_gl/VolumeSliceRenderer.comp.glsl");
        this->render_shader = core::utility::make_glowl_shader("render_shader", shaderOptions,
            "volume_gl/VolumeSliceRenderer.vert.glsl", "volume_gl/VolumeSliceRenderer.frag.glsl");

    } catch (std::exception& e) {
        Log::DefaultLog.WriteError(("VolumeSliceRenderer: " + std::string(e.what())).c_str());
        return false;
    }

    return true;
}


/*
 * VolumeSliceRenderer::VolumeSliceRenderer
 */
bool megamol::volume_gl::VolumeSliceRenderer::GetExtents(mmstd_gl::CallRender3DGL& cr) {
    auto* vdc = this->getVolSlot.CallAs<geocalls::VolumetricDataCall>();

    vdc->SetFrameID(static_cast<unsigned int>(cr.Time()));

    if (vdc == nullptr || !(*vdc)(geocalls::VolumetricDataCall::IDX_GET_EXTENTS))
        return false;
    if (vdc == nullptr || !(*vdc)(geocalls::VolumetricDataCall::IDX_GET_METADATA))
        return false;

    cr.SetTimeFramesCount(vdc->FrameCount());
    cr.AccessBoundingBoxes() = vdc->AccessBoundingBoxes();

    return true;
}


/*
 * VolumeSliceRenderer::VolumeSliceRenderer
 */
void megamol::volume_gl::VolumeSliceRenderer::release() {}


/*
 * VolumeSliceRenderer::VolumeSliceRenderer
 */
bool megamol::volume_gl::VolumeSliceRenderer::Render(mmstd_gl::CallRender3DGL& cr) {
    // get volume data
    auto* vdc = this->getVolSlot.CallAs<geocalls::VolumetricDataCall>();
    if (vdc == nullptr || !(*vdc)(geocalls::VolumetricDataCall::IDX_GET_EXTENTS))
        return false;
    if (vdc == nullptr || !(*vdc)(geocalls::VolumetricDataCall::IDX_GET_METADATA))
        return false;
    if (vdc == nullptr || !(*vdc)(geocalls::VolumetricDataCall::IDX_GET_DATA))
        return false;

    auto const metadata = vdc->GetMetadata();

    GLenum internal_format;
    GLenum format;
    GLenum type;

    switch (metadata->ScalarType) {
    case geocalls::FLOATING_POINT:
        if (metadata->ScalarLength == 4) {
            internal_format = GL_R32F;
            format = GL_RED;
            type = GL_FLOAT;
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
            internal_format = GL_R16UI;
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
            internal_format = GL_R16I;
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

    glowl::TextureLayout volume_layout(internal_format, metadata->Resolution[0], metadata->Resolution[1],
        metadata->Resolution[2], format, type, 1,
        {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
            {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_MIN_FILTER, GL_LINEAR},
            {GL_TEXTURE_MAG_FILTER, GL_LINEAR}},
        {});
    glowl::Texture3D vol_tex("volume_texture", volume_layout, vdc->GetData());

    // get clip plane
    core::view::CallClipPlane* ccp = this->getClipPlaneSlot.CallAs<core::view::CallClipPlane>();
    if (ccp == nullptr || !(*ccp)())
        return false;

    const auto slice = ccp->GetPlane();

    // get transfer function
    mmstd_gl::CallGetTransferFunctionGL* cgtf = this->getTFSlot.CallAs<mmstd_gl::CallGetTransferFunctionGL>();
    if (cgtf == nullptr || !(*cgtf)())
        return false;

    // get camera
    core::view::Camera cam = cr.GetCamera();
    auto view = cam.getViewMatrix();
    auto proj = cam.getProjectionMatrix();
    auto cr_fbo = cr.GetFramebuffer();

    // create render target
    glowl::TextureLayout render_tgt_layout(GL_RGBA8, cr_fbo->getWidth(), cr_fbo->getHeight(), 1, GL_RGBA,
        GL_UNSIGNED_BYTE, 1,
        {{GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER},
            {GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER}, {GL_TEXTURE_MIN_FILTER, GL_LINEAR},
            {GL_TEXTURE_MAG_FILTER, GL_LINEAR}},
        {});
    glowl::Texture2D render_target("render_target", render_tgt_layout, nullptr);

    // compute slice
    this->compute_shader->use();

    glUniformMatrix4fv(
        this->compute_shader->getUniformLocation("view_mx"), 1, GL_FALSE, glm::value_ptr(static_cast<glm::mat4>(view)));
    glUniformMatrix4fv(
        this->compute_shader->getUniformLocation("proj_mx"), 1, GL_FALSE, glm::value_ptr(static_cast<glm::mat4>(proj)));

    std::array<float, 2> rt_resolution;
    rt_resolution[0] = static_cast<float>(render_target.getWidth());
    rt_resolution[1] = static_cast<float>(render_target.getHeight());
    glUniform2fv(this->compute_shader->getUniformLocation("rt_resolution"), 1, rt_resolution.data());

    glm::vec3 box_min;
    box_min[0] = vdc->GetBoundingBoxes().ObjectSpaceBBox().Left();
    box_min[1] = vdc->GetBoundingBoxes().ObjectSpaceBBox().Bottom();
    box_min[2] = vdc->GetBoundingBoxes().ObjectSpaceBBox().Back();
    glm::vec3 box_max;
    box_max[0] = vdc->GetBoundingBoxes().ObjectSpaceBBox().Right();
    box_max[1] = vdc->GetBoundingBoxes().ObjectSpaceBBox().Top();
    box_max[2] = vdc->GetBoundingBoxes().ObjectSpaceBBox().Front();
    glUniform3fv(this->compute_shader->getUniformLocation("boxMin"), 1, glm::value_ptr(box_min));
    glUniform3fv(this->compute_shader->getUniformLocation("boxMax"), 1, glm::value_ptr(box_max));

    std::array<float, 2> valueRange;
    valueRange[0] = static_cast<float>(vdc->GetMetadata()->MinValues[0]);
    valueRange[1] = static_cast<float>(vdc->GetMetadata()->MaxValues[0]);
    glUniform2fv(this->compute_shader->getUniformLocation("valRange"), 1, valueRange.data());

    std::array<float, 4> plane;
    plane[0] = slice.GetA();
    plane[1] = slice.GetB();
    plane[2] = slice.GetC();
    plane[3] = std::abs(slice.GetD());
    glUniform4fv(this->compute_shader->getUniformLocation("slice"), 1, plane.data());

    glActiveTexture(GL_TEXTURE0);
    vol_tex.bindTexture();
    glUniform1i(this->compute_shader->getUniformLocation("volume_tx3D"), 0);

    cgtf->BindConvenience(this->compute_shader, GL_TEXTURE1, 1);

    render_target.bindImage(0, GL_WRITE_ONLY);

    glDispatchCompute(
        static_cast<int>(std::ceil(rt_resolution[0] / 8.0f)), static_cast<int>(std::ceil(rt_resolution[1] / 8.0f)), 1);

    glBindImageTexture(0, 0, 0, GL_TRUE, 0, GL_WRITE_ONLY, GL_R);

    cgtf->UnbindConvenience();

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_3D, 0);

    glUseProgram(0);

    glMemoryBarrier(GL_TEXTURE_FETCH_BARRIER_BIT);

    // render
    bool state_depth_test = glIsEnabled(GL_DEPTH_TEST);
    bool state_blend = glIsEnabled(GL_BLEND);

    GLint state_blend_src_rgb, state_blend_src_alpha, state_blend_dst_rgb, state_blend_dst_alpha;
    glGetIntegerv(GL_BLEND_SRC_RGB, &state_blend_src_rgb);
    glGetIntegerv(GL_BLEND_SRC_ALPHA, &state_blend_src_alpha);
    glGetIntegerv(GL_BLEND_DST_RGB, &state_blend_dst_rgb);
    glGetIntegerv(GL_BLEND_DST_ALPHA, &state_blend_dst_alpha);

    if (state_depth_test)
        glDisable(GL_DEPTH_TEST);
    if (!state_blend)
        glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    this->render_shader->use();

    glActiveTexture(GL_TEXTURE0);
    render_target.bindTexture();
    glUniform1i(this->render_shader->getUniformLocation("src_tx2D"), 0);

    glDrawArrays(GL_TRIANGLES, 0, 6);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);

    glUseProgram(0);

    glBlendFuncSeparate(state_blend_src_rgb, state_blend_dst_rgb, state_blend_src_alpha, state_blend_dst_alpha);
    if (!state_blend)
        glDisable(GL_BLEND);
    if (state_depth_test)
        glEnable(GL_DEPTH_TEST);
    else
        glDisable(GL_DEPTH_TEST);

    return true;
}
