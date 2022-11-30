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
    static const char* ClassName(void) {
        return "QuartzPlaneTexRenderer";
    }

    /**
     * Answer a human readable description of this module.
     *
     * @return A human readable description of this module.
     */
    static const char* Description(void) {
        return "Module rendering gridded quartz particles onto a clipping plane";
    }

    /**
     * Answers whether this module is available on the current system.
     *
     * @return 'true' if the module is available, 'false' otherwise.
     */
    static bool IsAvailable(void) {
        return true;
    }

    /**
     * Ctor
     */
    QuartzPlaneTexRenderer(void);

    /**
     * Dtor
     */
    virtual ~QuartzPlaneTexRenderer(void);

protected:
    /**
     * Implementation of 'Create'.
     *
     * @return 'true' on success, 'false' otherwise.
     */
    virtual bool create(void);

    /**
     * The get extents callback. The module should set the members of
     * 'call' to tell the caller the extents of its data (bounding boxes
     * and times).
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool GetExtents(mmstd_gl::CallRender2DGL& call);

    /**
     * Implementation of 'Release'.
     */
    virtual void release(void);

    /**
     * The render callback.
     *
     * @param call The calling call.
     *
     * @return The return value of the function.
     */
    virtual bool Render(mmstd_gl::CallRender2DGL& call);

private:
    /** The crystalite shader */
    std::unique_ptr<glowl::GLSLProgram> cryShader;

    /** Use clipping plane or grain colour for grains */
    core::param::ParamSlot useClipColSlot;
};

} // namespace demos_gl
} /* end namespace megamol */
