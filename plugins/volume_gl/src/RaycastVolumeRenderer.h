/*
 * RaycastVolumeRenderer.h
 *
 * Copyright (C) 2018-2020 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef RAYCAST_VOLUME_RENDERER_H_INCLUDED
#define RAYCAST_VOLUME_RENDERER_H_INCLUDED
#pragma once

#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore_gl/view/Renderer3DModuleGL.h"

#include "mmcore_gl/utility/ShaderFactory.h"

#include "vislib_gl/graphics/gl/FramebufferObject.h"

#include "glowl/Texture2D.hpp"
#include "glowl/Texture3D.hpp"

#include <array>
#include <limits>
#include <memory>

namespace megamol {
namespace volume_gl {

class RaycastVolumeRenderer : public megamol::core_gl::view::Renderer3DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "RaycastVolumeRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Modern compute-based raycast renderer for volumetric datasets.";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
#ifdef _WIN32
#if defined(DEBUG) || defined(_DEBUG)
        HDC dc = ::wglGetCurrentDC();
        HGLRC rc = ::wglGetCurrentContext();
        ASSERT(dc != NULL);
        ASSERT(rc != NULL);
#endif // DEBUG || _DEBUG
#endif // _WIN32
        return true;
    }

    RaycastVolumeRenderer();
    ~RaycastVolumeRenderer();

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create();

    /**
     * Implementation of 'Release'.
     */
    void release();

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool GetExtents(core_gl::view::CallRender3DGL& call) override;

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool Render(core_gl::view::CallRender3DGL& call) override;

    bool updateVolumeData(const unsigned int frameID);

    bool updateTransferFunction();

private:
    std::unique_ptr<glowl::GLSLProgram> rvc_dvr_shdr;
    std::unique_ptr<glowl::GLSLProgram> rvc_iso_shdr;
    std::unique_ptr<glowl::GLSLProgram> rvc_aggr_shdr;
    std::unique_ptr<glowl::GLSLProgram> rtf_shdr;
    std::unique_ptr<glowl::GLSLProgram> rtf_aggr_shdr;

    std::unique_ptr<glowl::Texture2D> m_render_target;
    std::unique_ptr<glowl::Texture2D> m_normal_target;
    std::unique_ptr<glowl::Texture2D> m_depth_target;

    std::unique_ptr<glowl::Texture3D> m_volume_texture;

    GLuint tf_texture;

    size_t m_volume_datahash = std::numeric_limits<size_t>::max();
    int m_frame_id = -1;

    float m_volume_origin[3];
    float m_volume_extents[3];
    float m_volume_resolution[3];

    /** Parameters for changing the behavior */
    core::param::ParamSlot m_mode;

    core::param::ParamSlot m_ray_step_ratio_param;
    core::param::ParamSlot m_opacity_threshold;
    core::param::ParamSlot m_iso_value;
    core::param::ParamSlot m_opacity;

    core::param::ParamSlot m_use_lighting_slot;
    core::param::ParamSlot m_ka_slot;
    core::param::ParamSlot m_kd_slot;
    core::param::ParamSlot m_ks_slot;
    core::param::ParamSlot m_shininess_slot;
    core::param::ParamSlot m_ambient_color;
    core::param::ParamSlot m_specular_color;
    core::param::ParamSlot m_material_color;

    /** caller slot */
    megamol::core::CallerSlot m_renderer_callerSlot;
    megamol::core::CallerSlot m_volumetricData_callerSlot;
    megamol::core::CallerSlot m_lights_callerSlot;
    megamol::core::CallerSlot m_transferFunction_callerSlot;

    std::array<float, 2> valRange;
    bool valRangeNeedsUpdate = false;

    /** FBO for chaining renderers */
    vislib_gl::graphics::gl::FramebufferObject fbo;
};

} // namespace volume_gl
} // namespace megamol

#endif
