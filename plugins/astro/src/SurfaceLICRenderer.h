/*
 * SurfaceLICRenderer.h
 *
 * Copyright (C) 2019 by MegaMol team
 * Alle Rechte vorbehalten.
 */

#ifndef ASTRO_SURFACELICRENDERER_H_INCLUDED
#define ASTRO_SURFACELICRENDERER_H_INCLUDED
#pragma once

#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/Renderer3DModule.h"

#include "vislib/graphics/gl/FramebufferObject.h"
#include "vislib/graphics/gl/GLSLComputeShader.h"
#include "vislib/graphics/gl/GLSLShader.h"

#include "glowl/Texture2D.hpp"
#include "glowl/Texture3D.hpp"

#include <memory>
#include <vector>

namespace megamol {
namespace astro {

class SurfaceLICRenderer : public megamol::core::view::Renderer3DModule {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) { return "SurfaceLICRenderer"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) { return "Renderer for surface LIC"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) { return true; }

    SurfaceLICRenderer();
    ~SurfaceLICRenderer();

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
    virtual bool GetExtents(core::Call& call) override;

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool Render(core::Call& call) override;

private:
    /** caller slot */
    core::CallerSlot m_input_renderer;
    core::CallerSlot m_input_velocities;
    core::CallerSlot m_input_transfer_function;

    /** Parameters */
    core::param::ParamSlot stencil_size;
    core::param::ParamSlot arc_length;
    core::param::ParamSlot num_advections;
    core::param::ParamSlot epsilon;
    core::param::ParamSlot coloring;

    /** Shader */
    std::unique_ptr<vislib::graphics::gl::GLSLComputeShader> m_lic_compute_shdr;
    std::unique_ptr<vislib::graphics::gl::GLSLShader> m_render_to_framebuffer_shdr;

    /** Textures */
    std::unique_ptr<glowl::Texture2D> m_render_target;
    std::unique_ptr<glowl::Texture3D> m_velocity_texture;
    std::unique_ptr<glowl::Texture2D> m_noise_texture;

    /** FBO for input */
    vislib::graphics::gl::FramebufferObject fbo;

    /** Noise texture data */
    std::vector<float> noise;
};

}
}

#endif /* ASTRO_SURFACELICRENDERER_H_INCLUDED */