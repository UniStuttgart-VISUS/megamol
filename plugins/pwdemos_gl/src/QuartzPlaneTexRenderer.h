/*
 * QuartzPlaneTexRenderer.h
 *
 * Copyright (C) 2018 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <memory>

#include <glowl/glowl.h>

#include "AbstractTexQuartzRenderer.h"
#include "mmcore/param/ParamSlot.h"
#include "mmstd_gl/renderer/CallRender2DGL.h"
#include "mmstd_gl/renderer/Renderer2DModuleGL.h"
#include "vislib_gl/graphics/gl/glfunctions.h"


namespace megamol {
namespace demos_gl {

/**
 * QuartzPlaneTexRenderer
 */
class QuartzPlaneTexRenderer : public mmstd_gl::Renderer2DModuleGL, public AbstractTexQuartzRenderer {
public:
    /**
     * Answer the name of this module.
     *
     * @return The name of this module.
     */
    static const char* ClassName() {
        return "QuartzPlaneTexRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description() {
        return "Module rendering gridded quartz particles onto a clipping plane";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable() {
        return true;
    }

    /**
     * Ctor
     */
    QuartzPlaneTexRenderer();

    /**
     * Dtor
     */
    ~QuartzPlaneTexRenderer() override;

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    bool create() override;

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool GetExtents(mmstd_gl::CallRender2DGL& call) override;

    /**
     * Implementation of 'Release'.
     */
    void release() override;

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    bool Render(mmstd_gl::CallRender2DGL& call) override;

private:
    /** The crystalite shader */
    std::unique_ptr<glowl::GLSLProgram> cryShader;

    /** Use clipping plane or grain colour for grains */
    core::param::ParamSlot useClipColSlot;
};

} // namespace demos_gl
} /* end namespace megamol */
