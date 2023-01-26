/*
 * QuartzTexRenderer.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <memory>

#include <glowl/glowl.h>

#include "AbstractTexQuartzRenderer.h"
#include "QuartzCrystalDataCall.h"
#include "QuartzParticleGridDataCall.h"
#include "mmcore/CallerSlot.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd_gl/renderer/CallRender3DGL.h"
#include "mmstd_gl/renderer/Renderer3DModuleGL.h"
#include "vislib_gl/graphics/gl/glfunctions.h"


namespace megamol {
namespace demos_gl {

/**
 * Module rendering gridded quarts particle data
 */
class QuartzTexRenderer : public mmstd_gl::Renderer3DModuleGL, public AbstractTexQuartzRenderer {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "QuartzTexRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Module rendering gridded quartz particles using GLSL ray casting shader";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /** Ctor */
    QuartzTexRenderer();

    /** Dtor */
    ~QuartzTexRenderer() override;

protected:
    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool GetExtents(mmstd_gl::CallRender3DGL& call) override;

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool Render(mmstd_gl::CallRender3DGL& call) override;

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

private:
    /** The crystalite shader */
    std::unique_ptr<glowl::GLSLProgram> cryShader;

    /** Shows/Hides the axes (x and y) of the clipping plane */
    core::param::ParamSlot showClipAxesSlot;

    // SSBO for multiple lights
    GLuint ssboLights;

    // vbo
    GLuint vbo;
};

} // namespace demos_gl
} /* end namespace megamol */
