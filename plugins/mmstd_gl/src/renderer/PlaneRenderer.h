/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <array>
#include <memory>

#include <glowl/GLSLProgram.hpp>

#include "mmcore/CallerSlot.h"
#include "mmstd_gl/renderer/CallRender3DGL.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"
#include "vislib/math/Plane.h"

namespace megamol::mmstd_gl {

/**
 * Module for rendering (clip) plane.
 *
 * @author Alexander Straub
 */
class PlaneRenderer : public mmstd_gl::Renderer3DModuleGL {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static inline const char* ClassName() {
        return "PlaneRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static inline const char* Description() {
        return "Render a (clip) plane";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static inline bool IsAvailable() {
        return true;
    }

    /**
     * Initialises a new instance.
     */
    PlaneRenderer();

    /**
     * Finalises an instance.
     */
    ~PlaneRenderer() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

    /** Callbacks for the computed streamlines */
    bool GetExtents(mmstd_gl::CallRender3DGL& call) override;
    bool Render(mmstd_gl::CallRender3DGL& call) override;

private:
    /** Call for getting the input plane */
    core::CallerSlot input_plane_slot;

    /** The plane color */
    std::array<float, 4> color;

    /** The (clip) plane */
    vislib::math::Plane<float> plane;

    /** Data needed for rendering */
    std::unique_ptr<glowl::GLSLProgram> render_data;

    /** Initialization status */
    bool initialized;
};

} // namespace megamol::mmstd_gl
