/*
 * RaycastVolumeRenderer.h
 *
 * Copyright (C) 2018-2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#ifndef RAYCAST_VOLUME_RENDERER_H_INCLUDED
#define RAYCAST_VOLUME_RENDERER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "vislib/graphics/gl/GLSLComputeShader.h"
#include "vislib/graphics/gl/GLSLShader.h"

#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/CallRender3D_2.h"
#include "mmcore/view/Renderer3DModule_2.h"

#include "glowl/Texture2D.hpp"
#include "glowl/Texture3D.hpp"

namespace megamol {
namespace stdplugin {
namespace volume {

class RaycastVolumeRenderer : public core::view::Renderer3DModule_2 {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "RaycastVolumeRenderer"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Modern compute-based raycast renderer for volumetric datasets."; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
#ifdef _WIN32
#    if defined(DEBUG) || defined(_DEBUG)
        HDC dc = ::wglGetCurrentDC();
        HGLRC rc = ::wglGetCurrentContext();
        ASSERT(dc != NULL);
        ASSERT(rc != NULL);
#    endif // DEBUG || _DEBUG
#endif     // _WIN32
        return vislib::graphics::gl::GLSLShader::AreExtensionsAvailable() && ogl_IsVersionGEQ(4, 3);
    }

    RaycastVolumeRenderer();
    ~RaycastVolumeRenderer();

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create() override;

    /**
     * Implementation of 'Release'.
     */
    virtual void release() override;

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetExtents(core::view::CallRender3D_2& call) override;

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool Render(core::view::CallRender3D_2& call) override;

    /**
     * Get and update data by calling input modules
     */
    bool updateVolumeData();

private:
    std::unique_ptr<vislib::graphics::gl::GLSLComputeShader> m_raycast_volume_compute_shdr;
    std::unique_ptr<vislib::graphics::gl::GLSLShader> m_render_to_framebuffer_shdr;

    std::unique_ptr<glowl::Texture2D> m_render_target;

    std::unique_ptr<glowl::Texture3D> m_volume_texture;

    std::size_t m_volume_datahash = std::numeric_limits<std::size_t>::max();
    int m_frame_id = -1;

    glm::vec3 m_volume_origin;
    glm::vec3 m_volume_extents;
    glm::vec3 m_volume_resolution;

    /** caller slot */
    core::CallerSlot m_volumetricData_callerSlot;
    core::CallerSlot m_transferFunction_callerSlot;

    core::param::ParamSlot m_ray_step_ratio_param;

    std::array<float, 2> valRange;
};

} // namespace volume
} // namespace stdplugin
} // namespace megamol

#endif