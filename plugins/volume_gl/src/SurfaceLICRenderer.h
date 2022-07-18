/*
 * SurfaceLICRenderer.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#pragma once

#include "mmcore/Call.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"

#include "vislib_gl/graphics/gl/GLSLComputeShader.h"
#include "vislib_gl/graphics/gl/GLSLShader.h"

#include "glowl/FramebufferObject.hpp"
#include "glowl/Texture2D.hpp"
#include "glowl/Texture3D.hpp"

#include <memory>
#include <vector>

namespace megamol {
namespace volume_gl {

class SurfaceLICRenderer : public mmstd_gl::Renderer3DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName(void) {
        return "SurfaceLICRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Renderer for surface LIC";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

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
    virtual bool GetExtents(mmstd_gl::CallRender3DGL& call) override;

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool Render(mmstd_gl::CallRender3DGL& call) override;

private:
    /** utility functions */
    vislib::math::Cuboid<float> combineBoundingBoxes(std::vector<vislib::math::Cuboid<float>> boxVec);

    /** caller slot */
    core::CallerSlot input_renderer;
    core::CallerSlot input_velocities;
    core::CallerSlot input_transfer_function;

    /** Parameters */
    core::param::ParamSlot arc_length;
    core::param::ParamSlot num_advections;
    core::param::ParamSlot epsilon;
    core::param::ParamSlot noise_bands;
    core::param::ParamSlot noise_scale;
    core::param::ParamSlot coloring;

    core::param::ParamSlot ka;
    core::param::ParamSlot kd;
    core::param::ParamSlot ks;
    core::param::ParamSlot shininess;
    core::param::ParamSlot ambient_color;
    core::param::ParamSlot specular_color;
    core::param::ParamSlot light_color;

    /** Input data hash */
    SIZE_T hash;

    /** Shader */
    vislib_gl::graphics::gl::GLSLComputeShader pre_compute_shdr;
    vislib_gl::graphics::gl::GLSLComputeShader lic_compute_shdr;
    vislib_gl::graphics::gl::GLSLShader render_to_framebuffer_shdr;

    /** Textures */
    std::unique_ptr<glowl::Texture2D> velocity_target;
    std::unique_ptr<glowl::Texture2D> render_target;
    std::unique_ptr<glowl::Texture3D> velocity_texture;
    std::unique_ptr<glowl::Texture3D> noise_texture;

    /** FBO for input */
    std::shared_ptr<glowl::FramebufferObject> fbo;

    /** Noise texture data */
    std::vector<float> noise;
};

} // namespace volume_gl
} // namespace megamol
