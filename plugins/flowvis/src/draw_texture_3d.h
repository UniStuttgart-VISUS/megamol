/*
 * draw_texture_3d.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
#pragma once

#include "mmcore/Call.h"
#include "mmcore/CalleeSlot.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/view/CallRender3D_2.h"
#include "mmcore/view/Renderer3DModule_2.h"

#include "vislib/graphics/gl/GLSLShader.h"

#include "glowl/Texture2D.hpp"

#include "glm/mat4x4.hpp"

#include <memory>

namespace megamol {
namespace flowvis {

/**
 * Module for drawing a two-dimensional texture on a quad in 3D space.
 *
 * @author Alexander Straub
 */
class draw_texture_3d : public core::view::Renderer3DModule_2 {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static inline const char* ClassName() { return "draw_texture_3d"; }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char* Description() { return "Draw a 2D texture in 3D space"; }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static inline bool IsAvailable() { return true; }

    /**
     * Initialises a new instance.
     */
    draw_texture_3d();

    /**
     * Finalises an instance.
     */
    virtual ~draw_texture_3d();

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

    /** Callbacks for the computed streamlines */
    virtual bool GetExtents(core::view::CallRender3D_2& call) override;
    virtual bool Render(core::view::CallRender3D_2& call) override;

private:
    /** Input slot for getting the texture */
    core::CallerSlot texture_slot;

    /** Input slot for getting the model matrix */
    core::CallerSlot model_matrix_slot;

    /** Parameters for defining a transparent color */
    core::param::ParamSlot enable_transparency;
    core::param::ParamSlot transparent_color;

    /** Information for rendering */
    struct render_data_t {
        GLuint vs, fs, cs, prog, cs_prog;

        std::shared_ptr<glowl::Texture2D> texture;
        glm::mat4 model_matrix;

        bool initialized = false;
    } render_data;
};

} // namespace flowvis
} // namespace megamol
